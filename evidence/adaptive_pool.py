"""Compare the original adaptive calibration pool with distinct source timestamps."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("study", type=Path, help="Extracted numerical release containing data_pipeline.py")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--workers", type=int, default=4)
args = parser.parse_args()
sys.path.insert(0, str(args.study.resolve()))

from data_pipeline import CONFIG, digest, evidence, observe, prepare
from physics import network, physics_tables
from run_study import evaluate, model_for

REGION = "DE_LU"
METRICS = ["load_coverage", "renewable_coverage", "width_kw", "load_daily_simultaneous_coverage",
           "renewable_daily_simultaneous_coverage", "full_rate", "cautious_rate", "abstention",
           "activity", "high_error_automation"]


def initial_overlap(obs, calibration_targets):
    # Read the history immediately before the first reported interval is formed.
    last_source = -1
    completed = []
    for pos in range(int(obs["indices"][0]) + 1):
        source = int(obs["source"][pos])
        if obs["available"][pos] and source > last_source and source > 0:
            completed.append(obs["times"][source])
            last_source = source
    return len(set(completed[-168:]).intersection(calibration_targets))


def run_window(origin):
    df = prepare(REGION)
    model = model_for(df, REGION, origin)
    params = json.loads((args.study / "results/frozen_parameters.json").read_text())
    gate = ("TUDT_F", params["weights"], params["low"], params["high"], 1)
    net = network()
    start = pd.Timestamp(origin, tz="UTC")
    lag = pd.Timedelta(hours=1)
    cal = df.index[(df.index >= start - pd.Timedelta(hours=CONFIG["calibration_hours"]))
                   & (df.index + lag < start)] + lag
    assert len(cal) == len(model.residuals) and cal.max() == pd.Timestamp(model.calibration_end)
    targets = set(cal)
    records = []
    for condition in CONFIG["scenarios"]:
        stochastic = condition in ("missing", "noisy", "fdi_like")
        for seed in CONFIG["seeds"] if stochastic else CONFIG["seeds"][:1]:
            obs = observe(df, origin, CONFIG["primary_hours"], condition, seed)
            overlap = initial_overlap(obs, targets)
            pf = None
            for method in ("adaptive", "adaptive_distinct"):
                ev = evidence(obs, model, method=method)
                if pf is None:
                    pf = (physics_tables(ev["pred"], net), physics_tables(ev["truth_next"], net))
                _, stats, _ = evaluate(ev, net, gate, None, params["error_cutoff_kw"], params["price_ref"], pf=pf)
                hit = (ev["truth_next"] >= ev["lo"]) & (ev["truth_next"] <= ev["hi"])
                daily = hit.reshape(-1,24,2).all(axis=1).mean(axis=0)
                values = [*hit.mean(axis=0), ev["width"].mean(), *daily, *(stats[k] for k in METRICS[5:])]
                records.append(dict(window=origin, scenario=condition, seed=seed,
                                    seed_multiplicity=1 if stochastic else len(CONFIG["seeds"]),
                                    interval=method, calibration_targets=len(cal), initial_overlap=overlap,
                                    **dict(zip(METRICS,values))))
    return records


def main():
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        runs = pd.DataFrame(row for block in pool.map(run_window, CONFIG["windows"]) for row in block)
    # Average seeds within each month first. Each of the nine months then has equal weight.
    per_month = runs.groupby(["scenario","interval","window"])[METRICS].mean().reset_index()
    summary = per_month.groupby(["scenario","interval"])[METRICS].mean().reset_index()
    old = pd.read_csv(args.study / "results/audit_v2/interval_authorization.csv")
    control = summary.query("interval=='adaptive'").merge(old.query("interval=='adaptive'"),
                                                           on=["scenario","interval"], validate="one_to_one")
    assert len(control) == len(CONFIG["scenarios"])
    for metric in METRICS:
        np.testing.assert_allclose(control[metric+"_x"], control[metric+"_y"], rtol=1e-12, atol=1e-12)
    assert (runs.query("scenario=='delayed'").initial_overlap == 168).all()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, frame in [("adaptive_pool_runs.csv",runs),("adaptive_pool_summary.csv",summary)]:
        frame.to_csv(args.output / name,index=False)
    files = ["data_pipeline.py","run_study.py","physics.py","control.py","protocol.json",
             "data/DE_LU_causal.csv","results/frozen_parameters.json","results/audit_v2/interval_authorization.csv"]
    manifest = dict(region=REGION, windows=CONFIG["windows"], seeds=CONFIG["seeds"],
                    scope="Exploratory initialization sensitivity with identical forecasts, feedback rule and frozen gate.",
                    matched_reference_values=len(control)*len(METRICS),
                    input_sha256={name:digest(args.study / name) for name in files},
                    script_sha256=digest(__file__),
                    result_sha256={name:digest(args.output / name) for name in ("adaptive_pool_runs.csv","adaptive_pool_summary.csv")})
    (args.output / "adaptive_pool_manifest.json").write_text(json.dumps(manifest,indent=2))


if __name__ == "__main__":
    main()
