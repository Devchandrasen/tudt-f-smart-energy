from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


ROOT = Path(__file__).resolve().parents[1]
STRENGTHENED_OUTPUTS = ROOT / "results" / "rolling_validation"
OUTPUT_DIR = ROOT / "results" / "extended"
RAW_FILE = ROOT / "data" / "raw" / "opsd_time_series_60min_singleindex.csv"

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import tudtf_core as p4  # noqa: E402
import run_rolling_validation as p5  # noqa: E402


DEGRADED_CORE = ["missing", "delayed", "noisy", "security_anomaly"]
CORE_BASELINES = [
    "hybrid_no_trust",
    "tudt_f",
    "tudt_no_sync",
    "tudt_no_data_reliability",
    "tudt_no_uncertainty",
    "tudt_no_security",
]


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def bootstrap_ci(values: np.ndarray, seed: int = 42, n_boot: int = 5000) -> Tuple[float, float]:
    if len(values) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means[i] = float(np.mean(sample))
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def holm_bonferroni(p_values: List[float]) -> List[float]:
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.zeros(m)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p_values[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted.tolist()


def decoupled_metric_tables(metrics: pd.DataFrame) -> None:
    rows: List[Dict[str, float | str]] = []
    for scenario in DEGRADED_CORE:
        for baseline in ["hybrid_no_trust", "tudt_f"]:
            subset = metrics[(metrics["scenario"] == scenario) & (metrics["baseline"] == baseline)]
            physical_rate = (
                (subset["battery_violation_count"] + subset["voltage_violation_count"] + subset["line_violation_count"])
                / 168.0
            ).mean()
            hybrid_cost = metrics[(metrics["scenario"] == scenario) & (metrics["baseline"] == "hybrid_no_trust")][
                "energy_cost_eur"
            ].mean()
            rows.append(
                {
                    "scenario": scenario,
                    "baseline": baseline,
                    "physical_infeasibility_rate": physical_rate,
                    "low_trust_automation_rate": subset["untrusted_automation_rate"].mean(),
                    "combined_physical_or_low_trust_rate": subset["unsafe_recommendation_rate"].mean(),
                    "recommendation_rate": subset["recommendation_rate"].mean(),
                    "abstention_rate": subset["abstention_rate"].mean(),
                    "mean_cost_eur": subset["energy_cost_eur"].mean(),
                    "cost_delta_vs_hybrid_eur": subset["energy_cost_eur"].mean() - hybrid_cost,
                    "mean_trust_score": subset["mean_trust_score"].mean(),
                }
            )
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "decoupled_metrics.csv", index=False)

    rows = []
    core = metrics[metrics["scenario"].isin(DEGRADED_CORE)]
    for baseline in ["hybrid_no_trust", "tudt_f"]:
        subset = core[core["baseline"] == baseline]
        rows.append(
            {
                "baseline": baseline,
                "physical_infeasibility_rate": float(
                    (
                        subset["battery_violation_count"]
                        + subset["voltage_violation_count"]
                        + subset["line_violation_count"]
                    ).sum()
                    / (168.0 * len(subset))
                ),
                "low_trust_automation_rate": subset["untrusted_automation_rate"].mean(),
                "combined_physical_or_low_trust_rate": subset["unsafe_recommendation_rate"].mean(),
                "recommendation_rate": subset["recommendation_rate"].mean(),
                "abstention_rate": subset["abstention_rate"].mean(),
                "mean_cost_eur": subset["energy_cost_eur"].mean(),
            }
        )
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "degraded_core_metric_summary.csv", index=False)


def corrected_stats(metrics: pd.DataFrame) -> None:
    rows: List[Dict[str, float | str]] = []
    subset = metrics[metrics["scenario"].isin(DEGRADED_CORE)]
    p_values: List[float] = []
    for comparator in [
        "hybrid_no_trust",
        "tudt_no_sync",
        "tudt_no_data_reliability",
        "tudt_no_uncertainty",
        "tudt_no_security",
    ]:
        pivot = subset.pivot_table(
            index=["window_start", "scenario"],
            columns="baseline",
            values="unsafe_recommendation_rate",
        ).dropna(subset=["tudt_f", comparator])
        diff = (pivot[comparator] - pivot["tudt_f"]).to_numpy()
        ci_low, ci_high = bootstrap_ci(diff)
        if stats is not None and np.any(np.abs(diff) > 1e-12):
            p_val = float(stats.wilcoxon(diff, alternative="greater").pvalue)
        else:
            p_val = 1.0
        p_values.append(p_val)
        rows.append(
            {
                "comparator": comparator,
                "n_pairs": len(diff),
                "mean_reduction": float(np.mean(diff)),
                "median_reduction": float(np.median(diff)),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "rank_biserial_effect_sign": float((diff > 1e-12).sum() / len(diff)),
                "wilcoxon_p": p_val,
            }
        )
    adjusted = holm_bonferroni(p_values)
    for row, p_adj in zip(rows, adjusted):
        row["holm_adjusted_p"] = p_adj
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "corrected_statistical_tests.csv", index=False)


def run_action_details(
    baseline: str,
    df: pd.DataFrame,
    pred_load: np.ndarray,
    pred_ren: np.ndarray,
    scores: pd.DataFrame,
    cfg: p4.ExperimentConfig,
    battery: p4.BatteryConfig,
    gate: str = "full",
) -> Dict[str, float]:
    soc = battery.initial_soc
    net = p4.make_power_network()
    median_price = float(df["price_eur_kwh"].median())
    counts = {
        "full_action": 0,
        "cautious_action": 0,
        "abstention": 0,
        "no_action": 0,
        "recommendation": 0,
        "low_trust_automation": 0,
        "physical_violation": 0,
    }
    cost_total = 0.0
    for i, row in enumerate(df.itertuples(index=False)):
        score_row = scores.iloc[i]
        full_trust = float(score_row["overall_trust_score"])
        if gate == "full":
            trust = full_trust
        elif gate == "uncertainty":
            trust = float(score_row["model_uncertainty_score"])
        elif gate == "data_quality":
            trust = float(score_row["data_reliability_score"])
        elif gate == "sync":
            trust = float(score_row["twin_synchronization_score"])
        elif gate == "security":
            trust = float(score_row["security_anomaly_score"])
        elif gate == "delay_detector":
            trust = 0.50 if float(row.delay_seconds) > 0 else 1.0
        elif gate == "anomaly_detector":
            trust = 0.50 if bool(row.anomaly_flag) else 1.0
        else:
            raise ValueError(gate)

        candidate = p4.action_from_prediction(
            float(pred_load[i]),
            float(pred_ren[i]),
            float(row.price_eur_kwh),
            median_price,
            battery,
            soc,
        )

        if baseline == "hybrid_no_trust":
            requested = candidate
            mode = "full_action" if abs(candidate) > 1e-6 else "no_action"
        elif trust < cfg.trust_low:
            requested = 0.0
            mode = "abstention"
        elif trust < cfg.trust_high:
            requested = 0.50 * candidate
            mode = "cautious_action" if abs(requested) > 1e-6 else "no_action"
        else:
            requested = candidate
            mode = "full_action" if abs(requested) > 1e-6 else "no_action"

        soc, executed, battery_violation = p4.apply_battery(requested, soc, battery)
        physics = p4.run_powerflow_feasibility(net, float(row.target_load_kw), float(row.target_renewable_kw), executed)
        grid_import = max(float(row.target_load_kw) - float(row.target_renewable_kw) - max(executed, 0.0) + max(-executed, 0.0), 0.0)
        export = max(float(row.target_renewable_kw) + max(executed, 0.0) - float(row.target_load_kw) - max(-executed, 0.0), 0.0)
        cost_total += grid_import * float(row.price_eur_kwh) - export * float(row.price_eur_kwh) * 0.20

        recommendation = abs(executed) > 1e-6
        physical_violation = bool(battery_violation) or physics["physics_feasible"] < 1.0
        low_trust_automation = bool(recommendation and full_trust < cfg.trust_high)
        counts[mode] += 1
        counts["recommendation"] += int(recommendation)
        counts["physical_violation"] += int(physical_violation)
        if baseline == "hybrid_no_trust" or gate != "full":
            counts["low_trust_automation"] += int(low_trust_automation)

    n = len(df)
    return {
        "full_action_rate": counts["full_action"] / n,
        "cautious_action_rate": counts["cautious_action"] / n,
        "abstention_rate": counts["abstention"] / n,
        "no_action_rate": counts["no_action"] / n,
        "recommendation_rate": counts["recommendation"] / n,
        "physical_infeasibility_rate": counts["physical_violation"] / n,
        "low_trust_automation_rate": counts["low_trust_automation"] / n,
        "energy_cost_eur": cost_total,
    }


def action_and_gate_baselines(df: pd.DataFrame) -> None:
    rows: List[Dict[str, float | str]] = []
    gate_specs = [
        ("hybrid_no_trust", "none"),
        ("tudt_f", "full"),
        ("uncertainty_only_gate", "uncertainty"),
        ("data_quality_only_gate", "data_quality"),
        ("sync_only_gate", "sync"),
        ("security_only_gate", "security"),
        ("delay_detector_gate", "delay_detector"),
        ("anomaly_detector_gate", "anomaly_detector"),
    ]
    for w_idx, window_start in enumerate(p5.WINDOW_STARTS):
        start = pd.Timestamp(window_start)
        train = df[df["timestamp"] < start].copy()
        test = df[df["timestamp"] >= start].head(168).copy()
        models, margins = p5.train_conformal_models(train)
        battery = p4.BatteryConfig()
        cfg = p4.ExperimentConfig(max_test_rows=168, test_start=window_start)
        for s_idx, scenario in enumerate(DEGRADED_CORE):
            scenario_df = p4.apply_scenario(test, scenario, 4200 + 17 * w_idx + s_idx)
            pred_load, load_low, load_high, pred_ren, ren_low, ren_high = p5.conformal_predictions(
                models, margins, scenario_df[p4.FEATURES]
            )
            scores = p4.compute_scores(
                scenario_df, pred_load, load_low, load_high, pred_ren, ren_low, ren_high
            )
            for baseline, gate in gate_specs:
                summary = run_action_details(
                    baseline,
                    scenario_df,
                    pred_load,
                    pred_ren,
                    scores,
                    cfg,
                    battery,
                    gate="full" if gate == "none" else gate,
                )
                rows.append(
                    {
                        "window_start": window_start,
                        "scenario": scenario,
                        "baseline": baseline,
                        **summary,
                    }
                )
    detail = pd.DataFrame(rows)
    detail.to_csv(OUTPUT_DIR / "action_mode_rolling_metrics.csv", index=False)
    summary = (
        detail.groupby(["scenario", "baseline"])[
            [
                "full_action_rate",
                "cautious_action_rate",
                "abstention_rate",
                "recommendation_rate",
                "physical_infeasibility_rate",
                "low_trust_automation_rate",
                "energy_cost_eur",
            ]
        ]
        .mean()
        .reset_index()
    )
    summary.to_csv(OUTPUT_DIR / "action_mode_summary.csv", index=False)


def threshold_sensitivity(df: pd.DataFrame) -> None:
    rows: List[Dict[str, float | str]] = []
    settings = [(0.55, 0.80), (0.65, 0.85), (0.75, 0.90)]
    for tau_l, tau_h in settings:
        for w_idx, window_start in enumerate(p5.WINDOW_STARTS):
            start = pd.Timestamp(window_start)
            train = df[df["timestamp"] < start].copy()
            test = df[df["timestamp"] >= start].head(168).copy()
            models, margins = p5.train_conformal_models(train)
            battery = p4.BatteryConfig()
            cfg = p4.ExperimentConfig(max_test_rows=168, test_start=window_start, trust_low=tau_l, trust_high=tau_h)
            for s_idx, scenario in enumerate(DEGRADED_CORE):
                scenario_df = p4.apply_scenario(test, scenario, 5200 + 17 * w_idx + s_idx)
                pred_load, load_low, load_high, pred_ren, ren_low, ren_high = p5.conformal_predictions(
                    models, margins, scenario_df[p4.FEATURES]
                )
                scores = p4.compute_scores(
                    scenario_df, pred_load, load_low, load_high, pred_ren, ren_low, ren_high
                )
                summary = run_action_details(
                    "tudt_f", scenario_df, pred_load, pred_ren, scores, cfg, battery, gate="full"
                )
                rows.append({"tau_low": tau_l, "tau_high": tau_h, "window_start": window_start, "scenario": scenario, **summary})
    detail = pd.DataFrame(rows)
    detail.to_csv(OUTPUT_DIR / "threshold_sensitivity_rolling.csv", index=False)
    summary = (
        detail.groupby(["tau_low", "tau_high"])[
            ["recommendation_rate", "cautious_action_rate", "abstention_rate", "low_trust_automation_rate", "energy_cost_eur"]
        ]
        .mean()
        .reset_index()
    )
    summary.to_csv(OUTPUT_DIR / "threshold_sensitivity_summary.csv", index=False)


def prepare_region(region: str) -> pd.DataFrame:
    if not RAW_FILE.exists():
        p4.download_dataset()
    cols = [
        "utc_timestamp",
        f"{region}_load_actual_entsoe_transparency",
        f"{region}_load_forecast_entsoe_transparency",
        f"{region}_price_day_ahead",
        f"{region}_solar_generation_actual",
        f"{region}_wind_generation_actual",
    ]
    raw = pd.read_csv(RAW_FILE, usecols=cols, parse_dates=["utc_timestamp"])
    raw = raw.rename(
        columns={
            "utc_timestamp": "timestamp",
            f"{region}_load_actual_entsoe_transparency": "load_mw",
            f"{region}_load_forecast_entsoe_transparency": "load_forecast_mw",
            f"{region}_price_day_ahead": "price_eur_mwh",
            f"{region}_solar_generation_actual": "solar_mw",
            f"{region}_wind_generation_actual": "wind_mw",
        }
    ).sort_values("timestamp")
    raw = raw[raw["timestamp"] >= pd.Timestamp("2015-01-01", tz="UTC")].copy()
    raw["solar_mw"] = raw["solar_mw"].fillna(0.0)
    raw["wind_mw"] = raw["wind_mw"].fillna(0.0)
    for col in ["load_mw", "load_forecast_mw", "price_eur_mwh"]:
        raw[col] = raw[col].interpolate(limit_direction="both")
    raw["price_eur_mwh"] = raw["price_eur_mwh"].fillna(raw["price_eur_mwh"].median())
    load_scale = raw["load_mw"].median()
    renewable_scale = (raw["solar_mw"] + raw["wind_mw"]).quantile(0.95)
    if renewable_scale <= 0:
        renewable_scale = 1.0
    out = pd.DataFrame()
    out["timestamp"] = raw["timestamp"]
    out["load_kw"] = raw["load_mw"] / load_scale * 600.0
    out["load_forecast_kw"] = raw["load_forecast_mw"] / load_scale * 600.0
    out["solar_kw"] = raw["solar_mw"] / renewable_scale * 260.0
    out["wind_kw"] = raw["wind_mw"] / renewable_scale * 140.0
    out["renewable_kw"] = out["solar_kw"] + out["wind_kw"]
    out["price_eur_kwh"] = raw["price_eur_mwh"] / 1000.0
    out["net_load_kw"] = out["load_kw"] - out["renewable_kw"]
    ts = out["timestamp"]
    out["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
    out["month_sin"] = np.sin(2 * np.pi * ts.dt.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * ts.dt.month / 12)
    for lag in [1, 24, 168]:
        out[f"load_lag_{lag}"] = out["load_kw"].shift(lag)
    for lag in [1, 24]:
        out[f"ren_lag_{lag}"] = out["renewable_kw"].shift(lag)
    out["load_roll_24"] = out["load_kw"].rolling(24).mean()
    out["ren_roll_24"] = out["renewable_kw"].rolling(24).mean()
    out["target_load_kw"] = out["load_kw"].shift(-1)
    out["target_renewable_kw"] = out["renewable_kw"].shift(-1)
    out["target_net_load_kw"] = out["target_load_kw"] - out["target_renewable_kw"]
    return out.dropna().reset_index(drop=True)


def external_region_check() -> None:
    try:
        dk = prepare_region("DK_1")
    except Exception as exc:
        pd.DataFrame([{"error": str(exc)}]).to_csv(OUTPUT_DIR / "external_region_dk1_summary.csv", index=False)
        return
    rows: List[Dict[str, float | str]] = []
    windows = [
        "2020-01-01 00:00:00+00:00",
        "2020-04-01 00:00:00+00:00",
        "2020-07-01 00:00:00+00:00",
    ]
    for w_idx, window_start in enumerate(windows):
        start = pd.Timestamp(window_start)
        train = dk[dk["timestamp"] < start].copy()
        test = dk[dk["timestamp"] >= start].head(168).copy()
        models, margins = p5.train_conformal_models(train)
        battery = p4.BatteryConfig()
        cfg = p4.ExperimentConfig(max_test_rows=168, test_start=window_start)
        for s_idx, scenario in enumerate(DEGRADED_CORE):
            scenario_df = p4.apply_scenario(test, scenario, 6200 + 19 * w_idx + s_idx)
            pred_load, load_low, load_high, pred_ren, ren_low, ren_high = p5.conformal_predictions(
                models, margins, scenario_df[p4.FEATURES]
            )
            scores = p4.compute_scores(
                scenario_df, pred_load, load_low, load_high, pred_ren, ren_low, ren_high
            )
            for baseline in ["hybrid_no_trust", "tudt_f"]:
                summary = run_action_details(
                    baseline,
                    scenario_df,
                    pred_load,
                    pred_ren,
                    scores,
                    cfg,
                    battery,
                    gate="full",
                )
                rows.append({"region": "DK_1", "window_start": window_start, "scenario": scenario, "baseline": baseline, **summary})
    detail = pd.DataFrame(rows)
    detail.to_csv(OUTPUT_DIR / "external_region_dk1_rolling.csv", index=False)
    summary = (
        detail.groupby(["region", "baseline"])[
            ["physical_infeasibility_rate", "low_trust_automation_rate", "recommendation_rate", "abstention_rate", "energy_cost_eur"]
        ]
        .mean()
        .reset_index()
    )
    summary.to_csv(OUTPUT_DIR / "external_region_dk1_summary.csv", index=False)


def main() -> None:
    ensure_dirs()
    metrics = pd.read_csv(STRENGTHENED_OUTPUTS / "rolling_window_metrics.csv")
    decoupled_metric_tables(metrics)
    corrected_stats(metrics)
    base_df = p4.load_and_prepare_data()
    action_and_gate_baselines(base_df)
    threshold_sensitivity(base_df)
    external_region_check()


if __name__ == "__main__":
    main()
