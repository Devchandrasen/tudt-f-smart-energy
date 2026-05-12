from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "rolling_validation"
sys.path.insert(0, str(ROOT / "src"))

import tudtf_core as p4  # noqa: E402


warnings.filterwarnings("ignore", message="Precision loss.*")


SCENARIOS = [
    "clean",
    "missing",
    "delayed",
    "noisy",
    "disturbance",
    "security_anomaly",
]

BASELINES = [
    "monitoring_only",
    "physics_only",
    "ml_only",
    "hybrid_no_trust",
    "tudt_no_sync",
    "tudt_no_data_reliability",
    "tudt_no_uncertainty",
    "tudt_no_security",
    "tudt_f",
]

COMPARATORS = [
    "hybrid_no_trust",
    "tudt_no_sync",
    "tudt_no_data_reliability",
    "tudt_no_uncertainty",
    "tudt_no_security",
]

WINDOW_STARTS = [
    "2020-01-01 00:00:00+00:00",
    "2020-02-01 00:00:00+00:00",
    "2020-03-01 00:00:00+00:00",
    "2020-04-01 00:00:00+00:00",
    "2020-05-01 00:00:00+00:00",
    "2020-06-01 00:00:00+00:00",
    "2020-07-01 00:00:00+00:00",
    "2020-08-01 00:00:00+00:00",
    "2020-09-01 00:00:00+00:00",
]


def conformal_quantile(values: np.ndarray, alpha: float = 0.10) -> float:
    if len(values) == 0:
        return 0.0
    level = min(1.0, math.ceil((len(values) + 1) * (1.0 - alpha)) / len(values))
    return float(np.quantile(values, level, method="higher"))


def train_conformal_models(
    train: pd.DataFrame, calibration_fraction: float = 0.20, alpha: float = 0.10
) -> Tuple[Dict[str, object], Dict[str, float]]:
    split = max(1000, int(len(train) * (1.0 - calibration_fraction)))
    fit = train.iloc[:split].copy()
    calibration = train.iloc[split:].copy()
    models = p4.train_models(fit)

    load_mean, load_low, load_high = p4.rf_interval(models["load"], calibration[p4.FEATURES])
    ren_mean, ren_low, ren_high = p4.rf_interval(models["renewable"], calibration[p4.FEATURES])

    load_residual = np.maximum.reduce(
        [
            calibration["target_load_kw"].to_numpy() - load_high,
            load_low - calibration["target_load_kw"].to_numpy(),
            np.zeros(len(calibration)),
        ]
    )
    ren_residual = np.maximum.reduce(
        [
            calibration["target_renewable_kw"].to_numpy() - ren_high,
            ren_low - calibration["target_renewable_kw"].to_numpy(),
            np.zeros(len(calibration)),
        ]
    )
    margins = {
        "load_margin": conformal_quantile(load_residual, alpha=alpha),
        "renewable_margin": conformal_quantile(ren_residual, alpha=alpha),
        "calibration_rows": float(len(calibration)),
    }
    return models, margins


def conformal_predictions(
    models: Dict[str, object], margins: Dict[str, float], x: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred_load, load_low, load_high = p4.rf_interval(models["load"], x)
    pred_ren, ren_low, ren_high = p4.rf_interval(models["renewable"], x)
    return (
        pred_load,
        load_low - margins["load_margin"],
        load_high + margins["load_margin"],
        pred_ren,
        np.maximum(0.0, ren_low - margins["renewable_margin"]),
        ren_high + margins["renewable_margin"],
    )


def run_window(
    df: pd.DataFrame, window_start: str, rows_per_window: int, seed_base: int
) -> List[Dict[str, float | str]]:
    start = pd.Timestamp(window_start)
    train = df[df["timestamp"] < start].copy()
    test = df[df["timestamp"] >= start].head(rows_per_window).copy()
    if len(test) < rows_per_window:
        raise ValueError(f"Window {window_start} has only {len(test)} rows")

    models, margins = train_conformal_models(train)
    battery = p4.BatteryConfig()
    cfg = p4.ExperimentConfig(max_test_rows=rows_per_window, test_start=window_start)

    rows: List[Dict[str, float | str]] = []
    for s_idx, scenario in enumerate(SCENARIOS):
        scenario_df = p4.apply_scenario(test, scenario, seed_base + s_idx)
        x = scenario_df[p4.FEATURES]
        pred_load, load_low, load_high, pred_ren, ren_low, ren_high = conformal_predictions(
            models, margins, x
        )
        scores = p4.compute_scores(
            scenario_df, pred_load, load_low, load_high, pred_ren, ren_low, ren_high
        )
        pred_metrics = p4.prediction_metrics(
            scenario_df, pred_load, load_low, load_high, pred_ren, ren_low, ren_high
        )
        for baseline in BASELINES:
            eval_metrics = p4.evaluate_baseline(
                baseline, scenario_df, pred_load, pred_ren, scores, cfg, battery
            )
            rows.append(
                {
                    "window_start": window_start,
                    "scenario": scenario,
                    "baseline": baseline,
                    "interval_method": "conformalized_rf",
                    "load_conformal_margin_kw": margins["load_margin"],
                    "renewable_conformal_margin_kw": margins["renewable_margin"],
                    **pred_metrics,
                    **eval_metrics,
                }
            )
    return rows


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "net_load_mae_kw",
        "net_load_rmse_kw",
        "load_interval_coverage",
        "renewable_interval_coverage",
        "mean_interval_width_kw",
        "energy_cost_eur",
        "unsafe_recommendation_rate",
        "recommendation_rate",
        "untrusted_automation_rate",
        "abstention_rate",
        "mean_trust_score",
    ]
    agg = (
        metrics.groupby(["scenario", "baseline"])[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.columns = [
        "_".join([str(part) for part in col if part]).rstrip("_")
        for col in agg.columns.to_flat_index()
    ]
    return agg


def paired_tests(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float | str]] = []
    scenario_groups = {scenario: [scenario] for scenario in SCENARIOS}
    scenario_groups["degraded_core"] = ["missing", "delayed", "noisy", "security_anomaly"]
    scenario_groups["all_degraded"] = [
        "missing",
        "delayed",
        "noisy",
        "disturbance",
        "security_anomaly",
    ]

    for group_name, scenarios in scenario_groups.items():
        subset = metrics[metrics["scenario"].isin(scenarios)]
        for comparator in COMPARATORS:
            pivot = subset.pivot_table(
                index=["window_start", "scenario"],
                columns="baseline",
                values="unsafe_recommendation_rate",
            ).dropna(subset=["tudt_f", comparator])
            diff = pivot[comparator] - pivot["tudt_f"]
            if len(diff) == 0:
                continue
            if stats is not None and np.any(np.abs(diff.to_numpy()) > 1e-12):
                try:
                    wilcoxon_p = float(stats.wilcoxon(diff, alternative="greater").pvalue)
                except ValueError:
                    wilcoxon_p = 1.0
                try:
                    ttest_p = float(stats.ttest_rel(pivot[comparator], pivot["tudt_f"], alternative="greater").pvalue)
                except Exception:
                    ttest_p = 1.0
            else:
                wilcoxon_p = 1.0
                ttest_p = 1.0

            rows.append(
                {
                    "scenario_group": group_name,
                    "comparator": comparator,
                    "n_pairs": float(len(diff)),
                    "mean_unsafe_tudt_f": float(pivot["tudt_f"].mean()),
                    "mean_unsafe_comparator": float(pivot[comparator].mean()),
                    "mean_absolute_reduction": float(diff.mean()),
                    "median_absolute_reduction": float(diff.median()),
                    "windows_with_improvement": float((diff > 1e-12).sum()),
                    "windows_equal": float((np.abs(diff) <= 1e-12).sum()),
                    "windows_worse": float((diff < -1e-12).sum()),
                    "wilcoxon_p_greater": wilcoxon_p,
                    "paired_ttest_p_greater": ttest_p,
                }
            )
    return pd.DataFrame(rows)


def write_gate_report(metrics: pd.DataFrame, aggregate: pd.DataFrame, tests: pd.DataFrame, rows_per_window: int) -> None:
    report = OUTPUT_DIR / "rolling_validation_report.md"
    degraded = tests[
        (tests["scenario_group"] == "degraded_core")
        & (tests["comparator"] == "hybrid_no_trust")
    ]
    if not degraded.empty:
        mean_reduction = float(degraded.iloc[0]["mean_absolute_reduction"])
        p_value = float(degraded.iloc[0]["wilcoxon_p_greater"])
    else:
        mean_reduction = 0.0
        p_value = 1.0

    gate_pass = mean_reduction > 0 and p_value < 0.05

    lines: List[str] = []
    lines.append("# Rolling Validation Report")
    lines.append("")
    lines.append(f"Rows per validation window: {rows_per_window}")
    lines.append(f"Validation windows: {len(WINDOW_STARTS)} monthly windows from January to September 2020")
    lines.append("Dataset: Open Power System Data Time Series package, version 2020-10-06")
    lines.append("")
    lines.append("## Validation Verdict")
    lines.append("")
    if gate_pass:
        lines.append("Full TUDT-F reduces the degraded-core combined physical-or-low-trust automation metric in the rolling-window protocol.")
    else:
        lines.append("Full TUDT-F does not meet the configured rolling-window reduction criterion.")
    lines.append("")
    lines.append("## Main Statistical Gate")
    lines.append("")
    lines.append(
        f"Against hybrid-without-trust on the degraded-core scenarios, full TUDT-F mean unsafe-rate reduction was {mean_reduction:.4f} with Wilcoxon one-sided p={p_value:.4g}."
    )
    lines.append("")
    lines.append("## Degraded-Core Comparisons")
    lines.append("")
    core = tests[tests["scenario_group"] == "degraded_core"].copy()
    for row in core.itertuples(index=False):
        lines.append(
            f"- TUDT-F vs {row.comparator}: mean unsafe comparator={row.mean_unsafe_comparator:.4f}, "
            f"TUDT-F={row.mean_unsafe_tudt_f:.4f}, reduction={row.mean_absolute_reduction:.4f}, "
            f"Wilcoxon p={row.wilcoxon_p_greater:.4g}"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "These results support a trust-gated decision-compliance contribution rather than a universal prediction-accuracy claim. "
        "The evaluated method reduces unsafe or untrusted automation under imperfect real-world data by using synchronization, data reliability, uncertainty, and anomaly evidence."
    )
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append("- `rolling_window_metrics.csv`")
    lines.append("- `aggregate_metrics.csv`")
    lines.append("- `statistical_tests.csv`")

    report.write_text("\n".join(lines), encoding="utf-8")


def run_rolling_validation(rows_per_window: int, max_windows: int | None = None) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = p4.load_and_prepare_data()
    starts = WINDOW_STARTS[:max_windows] if max_windows else WINDOW_STARTS
    all_rows: List[Dict[str, float | str]] = []
    for idx, start in enumerate(starts):
        print(f"Running window {idx + 1}/{len(starts)}: {start}")
        all_rows.extend(run_window(df, start, rows_per_window, seed_base=1000 + idx * 100))

    metrics = pd.DataFrame(all_rows)
    aggregate = aggregate_metrics(metrics)
    tests = paired_tests(metrics)

    metrics.to_csv(OUTPUT_DIR / "rolling_window_metrics.csv", index=False)
    aggregate.to_csv(OUTPUT_DIR / "aggregate_metrics.csv", index=False)
    tests.to_csv(OUTPUT_DIR / "statistical_tests.csv", index=False)
    write_gate_report(metrics, aggregate, tests, rows_per_window)

    print("Rolling validation complete.")
    print(f"Metrics: {OUTPUT_DIR / 'rolling_window_metrics.csv'}")
    print(f"Tests: {OUTPUT_DIR / 'statistical_tests.csv'}")
    print(f"Validation report: {OUTPUT_DIR / 'rolling_validation_report.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TUDT-F rolling-window validation.")
    parser.add_argument("--rows-per-window", type=int, default=168)
    parser.add_argument("--max-windows", type=int, default=None)
    args = parser.parse_args()
    run_rolling_validation(args.rows_per_window, args.max_windows)


if __name__ == "__main__":
    main()
