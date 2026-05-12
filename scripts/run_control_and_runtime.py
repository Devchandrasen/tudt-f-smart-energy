from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "extended"

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import tudtf_core as p4  # noqa: E402
import run_rolling_validation as p5  # noqa: E402
import run_extended_analysis as mra  # noqa: E402


DEGRADED_CORE = ["missing", "delayed", "noisy", "security_anomaly"]
WINDOWS = p5.WINDOW_STARTS


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def constrained_dispatch(
    load_kw: float,
    renewable_kw: float,
    price_eur_kwh: float,
    soc: float,
    battery: p4.BatteryConfig,
    robust: bool = False,
) -> float:
    candidates = np.linspace(-battery.max_kw, battery.max_kw, 9)
    best_action = 0.0
    best_objective = float("inf")
    for action in candidates:
        next_soc, executed, battery_violated = p4.apply_battery(float(action), soc, battery)
        if battery_violated:
            continue
        grid_import = max(load_kw - renewable_kw - max(executed, 0.0) + max(-executed, 0.0), 0.0)
        export = max(renewable_kw + max(executed, 0.0) - load_kw - max(-executed, 0.0), 0.0)
        energy_cost = grid_import * price_eur_kwh - export * price_eur_kwh * 0.20
        degradation = 0.0015 * abs(executed)
        soc_regularization = 12.0 * (next_soc - battery.initial_soc) ** 2
        conservatism = 0.0005 * max(load_kw - renewable_kw, 0.0) if robust else 0.0
        objective = energy_cost + degradation + soc_regularization + conservatism
        if objective < best_objective:
            best_objective = objective
            best_action = executed
    return float(best_action)


def evaluate_control(
    controller: str,
    df: pd.DataFrame,
    pred_load: np.ndarray,
    pred_ren: np.ndarray,
    load_high: np.ndarray,
    ren_low: np.ndarray,
    scores: pd.DataFrame,
    cfg: p4.ExperimentConfig,
    battery: p4.BatteryConfig,
) -> Dict[str, float]:
    soc = battery.initial_soc
    net = p4.make_power_network()
    counts = {
        "recommendation": 0,
        "low_trust_automation": 0,
        "physical_violation": 0,
        "abstention": 0,
        "cautious": 0,
    }
    cost_total = 0.0
    trust_values: List[float] = []

    for i, row in enumerate(df.itertuples(index=False)):
        score_row = scores.iloc[i]
        full_trust = float(score_row["overall_trust_score"])
        trust_values.append(full_trust)
        price = float(row.price_eur_kwh)
        if controller == "opf_dispatch":
            requested = constrained_dispatch(float(pred_load[i]), float(pred_ren[i]), price, soc, battery)
        elif controller == "robust_opf_dispatch":
            requested = constrained_dispatch(float(load_high[i]), float(ren_low[i]), price, soc, battery, robust=True)
        elif controller == "opf_plus_tudt_gate":
            candidate = constrained_dispatch(float(load_high[i]), float(ren_low[i]), price, soc, battery, robust=True)
            if full_trust < cfg.trust_low:
                requested = 0.0
                counts["abstention"] += 1
            elif full_trust < cfg.trust_high:
                requested = 0.50 * candidate
                if abs(requested) > 1e-6:
                    counts["cautious"] += 1
            else:
                requested = candidate
        else:
            raise ValueError(controller)

        soc, executed, battery_violated = p4.apply_battery(requested, soc, battery)
        physics = p4.run_powerflow_feasibility(net, float(row.target_load_kw), float(row.target_renewable_kw), executed)
        grid_import = max(float(row.target_load_kw) - float(row.target_renewable_kw) - max(executed, 0.0) + max(-executed, 0.0), 0.0)
        export = max(float(row.target_renewable_kw) + max(executed, 0.0) - float(row.target_load_kw) - max(-executed, 0.0), 0.0)
        cost_total += grid_import * price - export * price * 0.20
        recommendation = abs(executed) > 1e-6
        physical_violation = bool(battery_violated) or physics["physics_feasible"] < 1.0
        low_trust_automation = bool(recommendation and full_trust < cfg.trust_high and controller != "opf_plus_tudt_gate")
        counts["recommendation"] += int(recommendation)
        counts["physical_violation"] += int(physical_violation)
        counts["low_trust_automation"] += int(low_trust_automation)

    n = len(df)
    return {
        "physical_infeasibility_rate": counts["physical_violation"] / n,
        "low_trust_automation_rate": counts["low_trust_automation"] / n,
        "recommendation_rate": counts["recommendation"] / n,
        "cautious_action_rate": counts["cautious"] / n,
        "abstention_rate": counts["abstention"] / n,
        "energy_cost_eur": cost_total,
        "mean_trust_score": float(np.mean(trust_values)),
    }


def control_baselines(df: pd.DataFrame) -> None:
    rows: List[Dict[str, float | str]] = []
    for w_idx, window_start in enumerate(WINDOWS):
        start = pd.Timestamp(window_start)
        train = df[df["timestamp"] < start].copy()
        test = df[df["timestamp"] >= start].head(168).copy()
        models, margins = p5.train_conformal_models(train)
        battery = p4.BatteryConfig()
        cfg = p4.ExperimentConfig(max_test_rows=168, test_start=window_start)
        for s_idx, scenario in enumerate(DEGRADED_CORE):
            scenario_df = p4.apply_scenario(test, scenario, 7200 + 17 * w_idx + s_idx)
            pred_load, load_low, load_high, pred_ren, ren_low, ren_high = p5.conformal_predictions(
                models, margins, scenario_df[p4.FEATURES]
            )
            scores = p4.compute_scores(
                scenario_df, pred_load, load_low, load_high, pred_ren, ren_low, ren_high
            )
            for controller in ["opf_dispatch", "robust_opf_dispatch", "opf_plus_tudt_gate"]:
                summary = evaluate_control(
                    controller,
                    scenario_df,
                    pred_load,
                    pred_ren,
                    load_high,
                    ren_low,
                    scores,
                    cfg,
                    battery,
                )
                rows.append(
                    {
                        "window_start": window_start,
                        "scenario": scenario,
                        "controller": controller,
                        **summary,
                    }
                )
    detail = pd.DataFrame(rows)
    detail.to_csv(OUTPUT_DIR / "control_baseline_rolling.csv", index=False)
    summary = (
        detail.groupby("controller")[
            [
                "physical_infeasibility_rate",
                "low_trust_automation_rate",
                "recommendation_rate",
                "cautious_action_rate",
                "abstention_rate",
                "energy_cost_eur",
            ]
        ]
        .mean()
        .reset_index()
    )
    summary.to_csv(OUTPUT_DIR / "control_baseline_summary.csv", index=False)


def runtime_summary(df: pd.DataFrame) -> None:
    window_start = "2020-04-01 00:00:00+00:00"
    start = pd.Timestamp(window_start)
    train = df[df["timestamp"] < start].copy()
    test = df[df["timestamp"] >= start].head(168).copy()
    models, margins = p5.train_conformal_models(train)
    scenario_df = p4.apply_scenario(test, "delayed", 9001)
    pred_load, load_low, load_high, pred_ren, ren_low, ren_high = p5.conformal_predictions(
        models, margins, scenario_df[p4.FEATURES]
    )

    t0 = time.perf_counter()
    scores = p4.compute_scores(scenario_df, pred_load, load_low, load_high, pred_ren, ren_low, ren_high)
    trust_ms_per_step = (time.perf_counter() - t0) * 1000.0 / len(scenario_df)

    step_times = []
    battery = p4.BatteryConfig()
    cfg = p4.ExperimentConfig(max_test_rows=168, test_start=window_start)
    soc = battery.initial_soc
    net = p4.make_power_network()
    for i, row in enumerate(scenario_df.itertuples(index=False)):
        t0 = time.perf_counter()
        candidate = p4.action_from_prediction(
            float(pred_load[i]),
            float(pred_ren[i]),
            float(row.price_eur_kwh),
            float(scenario_df["price_eur_kwh"].median()),
            battery,
            soc,
        )
        trust = float(scores.iloc[i]["overall_trust_score"])
        if trust < cfg.trust_low:
            requested = 0.0
        elif trust < cfg.trust_high:
            requested = 0.50 * candidate
        else:
            requested = candidate
        soc, executed, _ = p4.apply_battery(requested, soc, battery)
        p4.run_powerflow_feasibility(net, float(row.target_load_kw), float(row.target_renewable_kw), executed)
        step_times.append((time.perf_counter() - t0) * 1000.0)

    pd.DataFrame(
        [
            {
                "scenario": "delayed",
                "rows": len(scenario_df),
                "trust_score_vectorized_ms_per_step": trust_ms_per_step,
                "full_decision_mean_ms_per_step": float(np.mean(step_times)),
                "full_decision_p95_ms_per_step": float(np.quantile(step_times, 0.95)),
                "hardware_note": "local desktop Python run; includes battery update and pandapower load flow",
            }
        ]
    ).to_csv(OUTPUT_DIR / "runtime_summary.csv", index=False)


def main() -> None:
    ensure_dirs()
    df = p4.load_and_prepare_data()
    control_baselines(df)
    runtime_summary(df)


if __name__ == "__main__":
    main()
