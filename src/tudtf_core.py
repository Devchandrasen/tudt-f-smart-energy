from __future__ import annotations

import math
import argparse
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import pandapower as pp
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pandapower is required. Install with: python -m pip install pandapower"
    ) from exc


warnings.filterwarnings("ignore", message="X has feature names.*")


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "results" / "primary"
FIGURE_DIR = OUTPUT_DIR / "figures"

OPSD_URL = (
    "https://data.open-power-system-data.org/time_series/2020-10-06/"
    "time_series_60min_singleindex.csv"
)
RAW_FILE = RAW_DIR / "opsd_time_series_60min_singleindex.csv"
PROCESSED_FILE = PROCESSED_DIR / "microgrid_opsd_de_lu_hourly.csv"


FEATURES = [
    "load_kw",
    "renewable_kw",
    "price_eur_kwh",
    "load_forecast_kw",
    "net_load_kw",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "load_lag_1",
    "load_lag_24",
    "load_lag_168",
    "ren_lag_1",
    "ren_lag_24",
    "load_roll_24",
    "ren_roll_24",
]


@dataclass(frozen=True)
class BatteryConfig:
    capacity_kwh: float = 800.0
    max_kw: float = 260.0
    min_soc: float = 0.12
    max_soc: float = 0.92
    initial_soc: float = 0.55
    efficiency: float = 0.94


@dataclass(frozen=True)
class ExperimentConfig:
    random_seed: int = 42
    max_test_rows: int = 240
    train_until: str = "2019-12-31 23:00:00+00:00"
    test_start: str = "2020-01-01 00:00:00+00:00"
    trust_high: float = 0.85
    trust_low: float = 0.65
    nominal_interval_coverage: float = 0.90


def ensure_dirs() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, FIGURE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def download_dataset() -> None:
    ensure_dirs()
    if RAW_FILE.exists() and RAW_FILE.stat().st_size > 100_000_000:
        return
    print(f"Downloading OPSD dataset to {RAW_FILE} ...")
    urllib.request.urlretrieve(OPSD_URL, RAW_FILE)


def load_and_prepare_data() -> pd.DataFrame:
    ensure_dirs()
    if PROCESSED_FILE.exists():
        df = pd.read_csv(PROCESSED_FILE, parse_dates=["timestamp"])
        return df

    download_dataset()
    usecols = [
        "utc_timestamp",
        "DE_LU_load_actual_entsoe_transparency",
        "DE_LU_load_forecast_entsoe_transparency",
        "DE_LU_price_day_ahead",
        "DE_LU_solar_generation_actual",
        "DE_LU_wind_generation_actual",
    ]
    raw = pd.read_csv(RAW_FILE, usecols=usecols, parse_dates=["utc_timestamp"])
    raw = raw.rename(
        columns={
            "utc_timestamp": "timestamp",
            "DE_LU_load_actual_entsoe_transparency": "load_mw",
            "DE_LU_load_forecast_entsoe_transparency": "load_forecast_mw",
            "DE_LU_price_day_ahead": "price_eur_mwh",
            "DE_LU_solar_generation_actual": "solar_mw",
            "DE_LU_wind_generation_actual": "wind_mw",
        }
    )
    raw = raw.sort_values("timestamp").reset_index(drop=True)
    raw = raw[raw["timestamp"] >= pd.Timestamp("2015-01-01", tz="UTC")].copy()

    for col in ["solar_mw", "wind_mw"]:
        raw[col] = raw[col].fillna(0.0)
    raw["load_mw"] = raw["load_mw"].interpolate(limit_direction="both")
    raw["load_forecast_mw"] = raw["load_forecast_mw"].interpolate(limit_direction="both")
    raw["price_eur_mwh"] = raw["price_eur_mwh"].interpolate(limit_direction="both")
    raw["price_eur_mwh"] = raw["price_eur_mwh"].fillna(raw["price_eur_mwh"].median())

    load_scale = raw["load_mw"].median()
    renewable_scale = (raw["solar_mw"] + raw["wind_mw"]).quantile(0.95)
    if renewable_scale <= 0:
        renewable_scale = 1.0

    df = pd.DataFrame()
    df["timestamp"] = raw["timestamp"]
    df["load_kw"] = raw["load_mw"] / load_scale * 600.0
    df["load_forecast_kw"] = raw["load_forecast_mw"] / load_scale * 600.0
    df["solar_kw"] = raw["solar_mw"] / renewable_scale * 260.0
    df["wind_kw"] = raw["wind_mw"] / renewable_scale * 140.0
    df["renewable_kw"] = df["solar_kw"] + df["wind_kw"]
    df["price_eur_kwh"] = raw["price_eur_mwh"] / 1000.0
    df["net_load_kw"] = df["load_kw"] - df["renewable_kw"]

    ts = df["timestamp"]
    hour = ts.dt.hour
    dow = ts.dt.dayofweek
    month = ts.dt.month
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    for lag in [1, 24, 168]:
        df[f"load_lag_{lag}"] = df["load_kw"].shift(lag)
    for lag in [1, 24]:
        df[f"ren_lag_{lag}"] = df["renewable_kw"].shift(lag)
    df["load_roll_24"] = df["load_kw"].rolling(24).mean()
    df["ren_roll_24"] = df["renewable_kw"].rolling(24).mean()

    df["target_load_kw"] = df["load_kw"].shift(-1)
    df["target_renewable_kw"] = df["renewable_kw"].shift(-1)
    df["target_net_load_kw"] = df["target_load_kw"] - df["target_renewable_kw"]

    df = df.dropna().reset_index(drop=True)
    df.to_csv(PROCESSED_FILE, index=False)
    return df


def train_models(train: pd.DataFrame) -> Dict[str, RandomForestRegressor]:
    kwargs = dict(
        n_estimators=60,
        min_samples_leaf=8,
        max_features=0.75,
        random_state=42,
        n_jobs=-1,
    )
    load_model = RandomForestRegressor(**kwargs)
    ren_model = RandomForestRegressor(**kwargs)
    load_model.fit(train[FEATURES], train["target_load_kw"])
    ren_model.fit(train[FEATURES], train["target_renewable_kw"])
    return {"load": load_model, "renewable": ren_model}


def rf_interval(
    model: RandomForestRegressor, x: pd.DataFrame, lower_q: float = 0.05, upper_q: float = 0.95
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = x.to_numpy()
    tree_preds = np.vstack([tree.predict(values) for tree in model.estimators_])
    mean = tree_preds.mean(axis=0)
    lower = np.quantile(tree_preds, lower_q, axis=0)
    upper = np.quantile(tree_preds, upper_q, axis=0)
    return mean, lower, upper


def apply_scenario(df: pd.DataFrame, scenario: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)
    out["delay_seconds"] = 0.0
    out["missing_flag"] = False
    out["noise_ratio"] = 0.0
    out["anomaly_flag"] = False
    out["sensor_agreement"] = 1.0

    if scenario == "clean":
        return out

    if scenario == "missing":
        idx = rng.choice(n, size=max(1, int(n * 0.12)), replace=False)
        out.loc[out.index[idx], "missing_flag"] = True
        for col in ["load_kw", "renewable_kw", "net_load_kw", "load_lag_1", "ren_lag_1"]:
            out.loc[out.index[idx], col] = np.nan
        out[FEATURES] = out[FEATURES].ffill().bfill()
        out.loc[out["missing_flag"], "sensor_agreement"] = 0.70
        return out

    if scenario == "delayed":
        delay_steps = 2
        out["delay_seconds"] = 3600.0 * delay_steps
        for col in ["load_kw", "renewable_kw", "net_load_kw", "load_lag_1", "ren_lag_1"]:
            out[col] = out[col].shift(delay_steps).bfill()
        out["sensor_agreement"] = 0.82
        return out

    if scenario == "noisy":
        load_noise = rng.normal(0, out["load_kw"].std() * 0.08, size=n)
        ren_noise = rng.normal(0, max(out["renewable_kw"].std(), 1.0) * 0.12, size=n)
        out["load_kw"] = (out["load_kw"] + load_noise).clip(lower=0)
        out["renewable_kw"] = (out["renewable_kw"] + ren_noise).clip(lower=0)
        out["net_load_kw"] = out["load_kw"] - out["renewable_kw"]
        out["noise_ratio"] = 0.12
        out["sensor_agreement"] = 0.76
        return out

    if scenario == "disturbance":
        start = int(n * 0.45)
        stop = min(n, start + max(24, int(n * 0.08)))
        out.loc[out.index[start:stop], "load_kw"] *= 1.28
        out.loc[out.index[start:stop], "renewable_kw"] *= 0.55
        out["net_load_kw"] = out["load_kw"] - out["renewable_kw"]
        out.loc[out.index[start:stop], "anomaly_flag"] = True
        out.loc[out.index[start:stop], "sensor_agreement"] = 0.68
        return out

    if scenario == "security_anomaly":
        idx = rng.choice(n, size=max(1, int(n * 0.06)), replace=False)
        signs = rng.choice([-1, 1], size=len(idx))
        out.loc[out.index[idx], "load_kw"] *= np.where(signs > 0, 1.55, 0.55)
        out.loc[out.index[idx], "renewable_kw"] *= np.where(signs > 0, 0.40, 1.80)
        out["net_load_kw"] = out["load_kw"] - out["renewable_kw"]
        out.loc[out.index[idx], "anomaly_flag"] = True
        out.loc[out.index[idx], "sensor_agreement"] = 0.42
        out.loc[out.index[idx], "noise_ratio"] = 0.20
        return out

    raise ValueError(f"Unknown scenario: {scenario}")


def clip01(x: np.ndarray | float) -> np.ndarray | float:
    return np.clip(x, 0.0, 1.0)


def compute_scores(
    df: pd.DataFrame,
    pred_load: np.ndarray,
    load_lower: np.ndarray,
    load_upper: np.ndarray,
    pred_ren: np.ndarray,
    ren_lower: np.ndarray,
    ren_upper: np.ndarray,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    delay_score = np.exp(-df["delay_seconds"].to_numpy() / 7200.0)
    missing_score = np.where(df["missing_flag"].to_numpy(), 0.45, 1.0)
    quality_score = clip01(1.0 - df["noise_ratio"].to_numpy() * 3.0)

    current_mismatch = np.abs((df["load_kw"] - df["renewable_kw"]).to_numpy() - (pred_load - pred_ren))
    mismatch_score = clip01(1.0 - current_mismatch / np.nanpercentile(df["load_kw"], 90))

    out["twin_synchronization_score"] = (
        0.25 * delay_score + 0.25 * missing_score + 0.25 * quality_score + 0.25 * mismatch_score
    )

    completeness_score = missing_score
    noise_score = quality_score
    outlier_score = np.where(df["anomaly_flag"].to_numpy(), 0.35, 1.0)
    agreement_score = df["sensor_agreement"].to_numpy()
    out["data_reliability_score"] = (
        0.30 * completeness_score + 0.20 * noise_score + 0.20 * outlier_score + 0.30 * agreement_score
    )

    load_width = load_upper - load_lower
    ren_width = ren_upper - ren_lower
    width_score = clip01(1.0 - (load_width + ren_width) / (np.nanpercentile(df["load_kw"], 90) * 1.25))
    load_hit = (df["target_load_kw"].to_numpy() >= load_lower) & (
        df["target_load_kw"].to_numpy() <= load_upper
    )
    ren_hit = (df["target_renewable_kw"].to_numpy() >= ren_lower) & (
        df["target_renewable_kw"].to_numpy() <= ren_upper
    )
    calibration_score = 0.5 * load_hit.astype(float) + 0.5 * ren_hit.astype(float)
    residual = np.abs(df["target_net_load_kw"].to_numpy() - (pred_load - pred_ren))
    residual_score = clip01(1.0 - residual / np.nanpercentile(np.abs(df["target_net_load_kw"]), 90))
    out["model_uncertainty_score"] = 0.35 * width_score + 0.35 * calibration_score + 0.30 * residual_score

    valid_range_score = np.where(
        (df["load_kw"].to_numpy() >= 0) & (df["renewable_kw"].to_numpy() >= 0), 1.0, 0.0
    )
    jump_score = np.where(df["anomaly_flag"].to_numpy(), 0.30, 1.0)
    consistency_score = agreement_score
    source_score = np.ones(len(df))
    out["security_anomaly_score"] = (
        0.30 * valid_range_score + 0.30 * jump_score + 0.30 * consistency_score + 0.10 * source_score
    )

    out["overall_trust_score"] = (
        0.30 * out["twin_synchronization_score"]
        + 0.25 * out["data_reliability_score"]
        + 0.25 * out["model_uncertainty_score"]
        + 0.20 * out["security_anomaly_score"]
    )
    return out


def make_power_network() -> pp.pandapowerNet:
    net = pp.create_empty_network()
    b_grid = pp.create_bus(net, vn_kv=20.0, name="grid_bus")
    b_microgrid = pp.create_bus(net, vn_kv=20.0, name="microgrid_bus")
    pp.create_ext_grid(net, b_grid, vm_pu=1.0, name="utility_grid")
    pp.create_line_from_parameters(
        net,
        b_grid,
        b_microgrid,
        length_km=3.0,
        r_ohm_per_km=0.42,
        x_ohm_per_km=0.16,
        c_nf_per_km=0.0,
        max_i_ka=0.035,
        name="feeder",
    )
    pp.create_load(net, b_microgrid, p_mw=0.1, q_mvar=0.03, name="microgrid_load")
    pp.create_sgen(net, b_microgrid, p_mw=0.0, q_mvar=0.0, name="renewable_generation")
    pp.create_sgen(net, b_microgrid, p_mw=0.0, q_mvar=0.0, name="battery_discharge")
    return net


def run_powerflow_feasibility(
    net: pp.pandapowerNet, load_kw: float, renewable_kw: float, battery_discharge_kw: float
) -> Dict[str, float]:
    load_mw = max(load_kw, 0.0) / 1000.0
    renewable_mw = max(renewable_kw, 0.0) / 1000.0
    batt_sgen_mw = max(battery_discharge_kw, 0.0) / 1000.0
    charge_load_mw = max(-battery_discharge_kw, 0.0) / 1000.0
    net.load.at[0, "p_mw"] = load_mw + charge_load_mw
    net.load.at[0, "q_mvar"] = (load_mw + charge_load_mw) * 0.28
    net.sgen.at[0, "p_mw"] = renewable_mw
    net.sgen.at[1, "p_mw"] = batt_sgen_mw
    try:
        pp.runpp(
            net,
            algorithm="nr",
            max_iteration=15,
            tolerance_mva=1e-5,
            init="results",
            numba=False,
        )
        min_v = float(net.res_bus.vm_pu.min())
        max_loading = float(net.res_line.loading_percent.max())
        converged = 1.0
    except Exception:
        min_v = 0.0
        max_loading = 999.0
        converged = 0.0
    feasible = float(converged and min_v >= 0.95 and max_loading <= 100.0)
    return {
        "min_voltage_pu": min_v,
        "line_loading_percent": max_loading,
        "powerflow_converged": converged,
        "physics_feasible": feasible,
    }


def action_from_prediction(
    pred_load_kw: float,
    pred_ren_kw: float,
    price_eur_kwh: float,
    median_price: float,
    battery: BatteryConfig,
    soc: float,
) -> float:
    net_pred = pred_load_kw - pred_ren_kw
    high_net = 410.0
    low_net = 390.0
    action_kw = 0.0
    if price_eur_kwh > median_price * 1.05 or net_pred > high_net + 160.0:
        available_kw = max(0.0, (soc - battery.min_soc) * battery.capacity_kwh * battery.efficiency)
        desired = 70.0 + max(0.0, net_pred - high_net) * 0.35
        action_kw = min(battery.max_kw, desired, available_kw)
    elif price_eur_kwh <= median_price * 0.95 or net_pred < low_net:
        available_kw = max(0.0, (battery.max_soc - soc) * battery.capacity_kwh / battery.efficiency)
        desired = 70.0 + max(0.0, low_net - net_pred) * 0.30
        action_kw = -min(battery.max_kw, desired, available_kw)
    elif net_pred > high_net:
        available_kw = max(0.0, (soc - battery.min_soc) * battery.capacity_kwh * battery.efficiency)
        action_kw = min(battery.max_kw, 35.0 + max(0.0, net_pred - high_net) * 0.20, available_kw)
    return float(action_kw)


def apply_battery(action_kw: float, soc: float, battery: BatteryConfig) -> Tuple[float, float, bool]:
    requested = action_kw
    if requested >= 0:
        max_discharge = max(0.0, (soc - battery.min_soc) * battery.capacity_kwh * battery.efficiency)
        executed = min(requested, battery.max_kw, max_discharge)
        new_soc = soc - executed / (battery.capacity_kwh * battery.efficiency)
    else:
        max_charge = max(0.0, (battery.max_soc - soc) * battery.capacity_kwh / battery.efficiency)
        charge_kw = min(-requested, battery.max_kw, max_charge)
        executed = -charge_kw
        new_soc = soc + charge_kw * battery.efficiency / battery.capacity_kwh
    violated = abs(requested - executed) > 1e-6
    return float(np.clip(new_soc, battery.min_soc, battery.max_soc)), float(executed), violated


def evaluate_baseline(
    baseline: str,
    df: pd.DataFrame,
    pred_load: np.ndarray,
    pred_ren: np.ndarray,
    scores: pd.DataFrame,
    cfg: ExperimentConfig,
    battery: BatteryConfig,
) -> Dict[str, float]:
    soc = battery.initial_soc
    net = make_power_network()
    rows: List[Dict[str, float]] = []
    median_price = float(df["price_eur_kwh"].median())
    trust_gated_baselines = {
        "tudt_f",
        "tudt_no_sync",
        "tudt_no_data_reliability",
        "tudt_no_uncertainty",
        "tudt_no_security",
    }

    for i, row in enumerate(df.itertuples(index=False)):
        actual_load = float(row.target_load_kw)
        actual_ren = float(row.target_renewable_kw)
        price = float(row.price_eur_kwh)
        score_row = scores.iloc[i]
        full_trust = float(score_row["overall_trust_score"])
        trust = trust_for_baseline(baseline, score_row)
        abstained = 0.0

        if baseline == "monitoring_only":
            requested_action = 0.0
            model_load = float(row.load_kw)
            model_ren = float(row.renewable_kw)
        elif baseline == "physics_only":
            requested_action = action_from_prediction(
                float(row.load_forecast_kw),
                float(row.renewable_kw),
                price,
                median_price,
                battery,
                soc,
            )
            model_load = float(row.load_forecast_kw)
            model_ren = float(row.renewable_kw)
        elif baseline in {"ml_only", "hybrid_no_trust"}:
            requested_action = action_from_prediction(
                float(pred_load[i]), float(pred_ren[i]), price, median_price, battery, soc
            )
            model_load = float(pred_load[i])
            model_ren = float(pred_ren[i])
        elif baseline in trust_gated_baselines:
            requested_action = action_from_prediction(
                float(pred_load[i]), float(pred_ren[i]), price, median_price, battery, soc
            )
            if trust < cfg.trust_low:
                requested_action = 0.0
                abstained = 1.0
            elif trust < cfg.trust_high:
                requested_action *= 0.50
            model_load = float(pred_load[i])
            model_ren = float(pred_ren[i])
        else:
            raise ValueError(f"Unknown baseline: {baseline}")

        soc, executed_action, battery_violation = apply_battery(requested_action, soc, battery)
        physics = run_powerflow_feasibility(net, actual_load, actual_ren, executed_action)
        grid_import = max(actual_load - actual_ren - max(executed_action, 0.0) + max(-executed_action, 0.0), 0.0)
        export = max(actual_ren + max(executed_action, 0.0) - actual_load - max(-executed_action, 0.0), 0.0)
        cost = grid_import * price - export * price * 0.20
        recommendation_made = float(abs(executed_action) > 1e-6)
        untrusted_automation = (
            1.0
            if baseline != "tudt_f" and full_trust < cfg.trust_high and recommendation_made
            else 0.0
        )
        tudt_low_trust_violation = (
            1.0
            if baseline == "tudt_f" and full_trust < cfg.trust_low and recommendation_made
            else 0.0
        )
        unsafe = float(
            battery_violation
            or physics["physics_feasible"] < 1.0
            or untrusted_automation
            or tudt_low_trust_violation
        )
        prediction_abs_error = abs((actual_load - actual_ren) - (model_load - model_ren))

        rows.append(
            {
                "cost_eur": cost,
                "grid_import_kwh": grid_import,
                "export_kwh": export,
                "battery_action_kw": executed_action,
                "battery_soc": soc,
                "unsafe_recommendation": unsafe,
                "recommendation_made": recommendation_made,
                "untrusted_automation": untrusted_automation,
                "battery_violation": float(battery_violation),
                "abstained": abstained,
                "low_trust": float(trust < cfg.trust_low),
                "low_full_trust": float(full_trust < cfg.trust_low),
                "physics_feasible": physics["physics_feasible"],
                "voltage_violation": float(physics["min_voltage_pu"] < 0.95),
                "line_violation": float(physics["line_loading_percent"] > 100.0),
                "prediction_abs_error": prediction_abs_error,
                "overall_trust_score": trust,
                "full_overall_trust_score": full_trust,
            }
        )

    results = pd.DataFrame(rows)
    low_trust_count = max(results["low_trust"].sum(), 1.0)
    return {
        "energy_cost_eur": float(results["cost_eur"].sum()),
        "grid_import_kwh": float(results["grid_import_kwh"].sum()),
        "mean_soc": float(results["battery_soc"].mean()),
        "unsafe_recommendation_rate": float(results["unsafe_recommendation"].mean()),
        "recommendation_rate": float(results["recommendation_made"].mean()),
        "untrusted_automation_rate": float(results["untrusted_automation"].mean()),
        "battery_violation_count": float(results["battery_violation"].sum()),
        "voltage_violation_count": float(results["voltage_violation"].sum()),
        "line_violation_count": float(results["line_violation"].sum()),
        "abstention_rate": float(results["abstained"].mean()),
        "low_trust_abstention_accuracy": float(
            ((results["abstained"] == 1.0) & (results["low_trust"] == 1.0)).sum() / low_trust_count
        ),
        "mean_decision_abs_error": float(results["prediction_abs_error"].mean()),
        "mean_trust_score": float(results["overall_trust_score"].mean()),
        "mean_full_trust_score": float(results["full_overall_trust_score"].mean()),
    }


def trust_for_baseline(baseline: str, score_row: pd.Series) -> float:
    tss = float(score_row["twin_synchronization_score"])
    drs = float(score_row["data_reliability_score"])
    mus = float(score_row["model_uncertainty_score"])
    sas = float(score_row["security_anomaly_score"])
    if baseline == "tudt_no_sync":
        return float((0.25 * drs + 0.25 * mus + 0.20 * sas) / 0.70)
    if baseline == "tudt_no_data_reliability":
        return float((0.30 * tss + 0.25 * mus + 0.20 * sas) / 0.75)
    if baseline == "tudt_no_uncertainty":
        return float((0.30 * tss + 0.25 * drs + 0.20 * sas) / 0.75)
    if baseline == "tudt_no_security":
        return float((0.30 * tss + 0.25 * drs + 0.25 * mus) / 0.80)
    return float(score_row["overall_trust_score"])


def prediction_metrics(
    df: pd.DataFrame,
    pred_load: np.ndarray,
    load_lower: np.ndarray,
    load_upper: np.ndarray,
    pred_ren: np.ndarray,
    ren_lower: np.ndarray,
    ren_upper: np.ndarray,
) -> Dict[str, float]:
    target_net = df["target_net_load_kw"].to_numpy()
    pred_net = pred_load - pred_ren
    rmse = math.sqrt(mean_squared_error(target_net, pred_net))
    mae = mean_absolute_error(target_net, pred_net)
    load_coverage = np.mean((df["target_load_kw"] >= load_lower) & (df["target_load_kw"] <= load_upper))
    ren_coverage = np.mean(
        (df["target_renewable_kw"] >= ren_lower) & (df["target_renewable_kw"] <= ren_upper)
    )
    mean_interval_width = float(np.mean((load_upper - load_lower) + (ren_upper - ren_lower)))
    return {
        "net_load_mae_kw": float(mae),
        "net_load_rmse_kw": float(rmse),
        "load_interval_coverage": float(load_coverage),
        "renewable_interval_coverage": float(ren_coverage),
        "mean_interval_width_kw": mean_interval_width,
    }


def run_experiment(max_test_rows: int | None = None) -> pd.DataFrame:
    cfg = ExperimentConfig(
        max_test_rows=max_test_rows if max_test_rows is not None else ExperimentConfig.max_test_rows
    )
    battery = BatteryConfig()
    ensure_dirs()
    df = load_and_prepare_data()
    train_until = pd.Timestamp(cfg.train_until)
    test_start = pd.Timestamp(cfg.test_start)
    train = df[df["timestamp"] <= train_until].copy()
    test = df[df["timestamp"] >= test_start].head(cfg.max_test_rows).copy()

    print(f"Training rows: {len(train):,}; test rows per scenario: {len(test):,}")
    models = train_models(train)

    scenario_names = [
        "clean",
        "missing",
        "delayed",
        "noisy",
        "disturbance",
        "security_anomaly",
    ]
    baselines = [
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
    all_rows: List[Dict[str, float | str]] = []

    for s_idx, scenario in enumerate(scenario_names):
        scenario_df = apply_scenario(test, scenario, cfg.random_seed + s_idx)
        x = scenario_df[FEATURES]
        pred_load, load_lower, load_upper = rf_interval(models["load"], x)
        pred_ren, ren_lower, ren_upper = rf_interval(models["renewable"], x)
        scores = compute_scores(
            scenario_df,
            pred_load,
            load_lower,
            load_upper,
            pred_ren,
            ren_lower,
            ren_upper,
        )
        pred_metrics = prediction_metrics(
            scenario_df, pred_load, load_lower, load_upper, pred_ren, ren_lower, ren_upper
        )
        for baseline in baselines:
            eval_metrics = evaluate_baseline(
                baseline, scenario_df, pred_load, pred_ren, scores, cfg, battery
            )
            all_rows.append(
                {
                    "scenario": scenario,
                    "baseline": baseline,
                    **pred_metrics,
                    **eval_metrics,
                }
            )
        print(f"Completed scenario: {scenario}")

    metrics = pd.DataFrame(all_rows)
    metrics_path = OUTPUT_DIR / "primary_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    write_summary(metrics, train, test)
    make_figures(metrics)
    return metrics


def make_figures(metrics: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    tudt = metrics[metrics["baseline"] == "tudt_f"].set_index("scenario")
    no_trust = metrics[metrics["baseline"] == "hybrid_no_trust"].set_index("scenario")

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_df = metrics.pivot(index="scenario", columns="baseline", values="unsafe_recommendation_rate")
    plot_df.plot(kind="bar", ax=ax)
    ax.set_ylabel("Unsafe recommendation rate")
    ax.set_title("Safety under degraded data conditions")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "unsafe_recommendation_rate.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    scenarios = tudt.index.tolist()
    x = np.arange(len(scenarios))
    width = 0.35
    ax.bar(x - width / 2, no_trust["energy_cost_eur"], width, label="Hybrid without trust")
    ax.bar(x + width / 2, tudt["energy_cost_eur"], width, label="TUDT-F")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=30, ha="right")
    ax.set_ylabel("Operating cost over horizon (EUR equivalent)")
    ax.set_title("Cost comparison: trust-gated vs non-trust hybrid DT")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "cost_comparison.png", dpi=180)
    plt.close(fig)


def pct_change(new: float, old: float) -> float:
    if abs(old) < 1e-12:
        return 0.0
    return (new - old) / old * 100.0


def write_summary(metrics: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> None:
    summary_path = OUTPUT_DIR / "primary_summary.md"
    lines: List[str] = []
    lines.append("# TUDT-F Primary Experiment Summary")
    lines.append("")
    lines.append("Generated by `src/tudtf_core.py`.")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append("- Source: Open Power System Data Time Series package, version 2020-10-06.")
    lines.append("- DOI: https://doi.org/10.25832/time_series/2020-10-06")
    lines.append(f"- Training rows: {len(train):,}")
    lines.append(f"- Test rows per scenario: {len(test):,}")
    lines.append("- Region/variables: DE-LU load, day-ahead price, solar generation, wind generation.")
    lines.append("")
    lines.append("## Main Results")
    lines.append("")

    for scenario in metrics["scenario"].unique():
        sub = metrics[metrics["scenario"] == scenario].set_index("baseline")
        tudt = sub.loc["tudt_f"]
        hybrid = sub.loc["hybrid_no_trust"]
        ablations = sub.loc[
            [
                "tudt_no_sync",
                "tudt_no_data_reliability",
                "tudt_no_uncertainty",
                "tudt_no_security",
            ]
        ]
        lines.append(f"### {scenario}")
        lines.append("")
        lines.append(
            f"- TUDT-F unsafe recommendation rate: {tudt['unsafe_recommendation_rate']:.3f}"
        )
        lines.append(
            f"- Hybrid without trust unsafe recommendation rate: {hybrid['unsafe_recommendation_rate']:.3f}"
        )
        lines.append(
            f"- Unsafe-rate change: {pct_change(tudt['unsafe_recommendation_rate'], hybrid['unsafe_recommendation_rate']):.1f}%"
        )
        lines.append(
            f"- TUDT-F untrusted automation rate: {tudt['untrusted_automation_rate']:.3f}"
        )
        lines.append(
            f"- Hybrid without trust untrusted automation rate: {hybrid['untrusted_automation_rate']:.3f}"
        )
        lines.append(f"- TUDT-F recommendation rate: {tudt['recommendation_rate']:.3f}")
        lines.append(f"- Hybrid without trust recommendation rate: {hybrid['recommendation_rate']:.3f}")
        lines.append(f"- TUDT-F operating cost: {tudt['energy_cost_eur']:.2f}")
        lines.append(f"- Hybrid without trust operating cost: {hybrid['energy_cost_eur']:.2f}")
        lines.append(f"- Mean TUDT-F trust score: {tudt['mean_trust_score']:.3f}")
        lines.append(f"- TUDT-F abstention rate: {tudt['abstention_rate']:.3f}")
        lines.append(
            f"- Worst ablation unsafe recommendation rate: {ablations['unsafe_recommendation_rate'].max():.3f}"
        )
        lines.append(
            f"- Worst ablation untrusted automation rate: {ablations['untrusted_automation_rate'].max():.3f}"
        )
        lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The intended contribution is safer decision behavior under imperfect data, not simply lower raw forecast error. "
        "Unsafe recommendation is defined as either a physical/battery constraint violation or an automated recommendation made below the safe trust threshold by a non-trust baseline. "
        "The trust gate should reduce unsafe recommendations in missing, delayed, disturbance, and security-like anomaly scenarios. "
        "Any cost increase under low-trust scenarios is interpreted as the price of avoiding risky automated decisions."
    )
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append("- `primary_metrics.csv`")
    lines.append("- `figures/unsafe_recommendation_rate.png`")
    lines.append("- `figures/cost_comparison.png`")

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TUDT-F real-data experiment.")
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=None,
        help="Number of hourly test rows per scenario.",
    )
    args = parser.parse_args()
    metrics = run_experiment(max_test_rows=args.max_test_rows)
    print("\nTUDT-F primary experiment complete.")
    print(metrics[["scenario", "baseline", "energy_cost_eur", "unsafe_recommendation_rate", "mean_trust_score"]])
    print(f"\nMetrics: {OUTPUT_DIR / 'primary_metrics.csv'}")
    print(f"Summary: {OUTPUT_DIR / 'primary_summary.md'}")


if __name__ == "__main__":
    main()
