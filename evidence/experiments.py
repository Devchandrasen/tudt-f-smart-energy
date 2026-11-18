"""Run with python experiments.py PATH_TO_ISJ_STUDY [--output DIR]."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
DESIGN = json.loads((HERE / "design.json").read_text())
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("study", type=Path)
parser.add_argument("--output", type=Path, default=HERE / "results")
parser.add_argument("--workers", type=int, default=3)
args = parser.parse_args()
sys.path.insert(0, str(args.study.resolve()))
from data_pipeline import CONFIG, digest, evidence, observe, prepare
from physics import network, physics_tables
from run_study import RESULT, evaluate, model_for

P = json.loads((RESULT / "frozen_parameters.json").read_text())
GATE = ("TUDT_F", P["weights"], P["low"], P["high"], 1)
SCENARIOS = CONFIG["scenarios"]
STOCHASTIC = {"missing", "noisy", "fdi_like"}
GRID = DESIGN["stale_grid"]
METRICS = ["load_coverage", "renewable_coverage", "width_kw", "load_daily_simultaneous_coverage",
           "renewable_daily_simultaneous_coverage", "full_rate", "cautious_rate", "abstention", "activity",
           "high_error_automation"]


def weighted_correction(scores, ages):
    weights = 2.0 ** (-np.asarray(ages) / DESIGN["weight_half_life_hours"])
    target = (1 - DESIGN["miscoverage"]) * (1 + weights.sum())
    corrections = []
    for column in scores.T:
        order = np.argsort(column, kind="stable")
        index = np.searchsorted(np.cumsum(weights[order]), target, side="left")
        corrections.append(column[order[index]] if index < len(column) else np.inf)
    return np.array(corrections)


def observations(df, origin, condition, seed):
    return observe(df, origin, CONFIG["primary_hours"], condition, seed)


def interval_window(origin):
    df = prepare(DESIGN["region"])
    model = model_for(df, DESIGN["region"], origin)
    origin_utc = pd.Timestamp(origin, tz="UTC")
    calibration = df.index[(df.index >= origin_utc - pd.Timedelta(hours=CONFIG["calibration_hours"]))
                           & (df.index + pd.Timedelta(hours=1) < origin_utc)]
    ages = (origin_utc - (calibration + pd.Timedelta(hours=1))) / pd.Timedelta(hours=1)
    assert len(ages) == len(model.residuals) and min(ages) >= 1
    q = weighted_correction(model.residuals, ages)
    assert np.isfinite(q).all() and (q >= 0).all()
    net = network()
    records, subtle = [], []
    for scenario in SCENARIOS:
        deterministic = scenario not in STOCHASTIC
        for seed in CONFIG["seeds"][:1] if deterministic else CONFIG["seeds"]:
            obs = observations(df, origin, scenario, seed)
            pf = None
            for method, forecaster in (("split", model), ("weighted", replace(model, q=q))):
                e = evidence(obs, forecaster)
                if pf is None:
                    pf = (physics_tables(e["pred"], net), physics_tables(e["truth_next"], net))
                commands, stats, _ = evaluate(e, net, GATE, None, P["error_cutoff_kw"], P["price_ref"], pf=pf)
                hits = (e["truth_next"] >= e["lo"]) & (e["truth_next"] <= e["hi"])
                daily = hits.reshape(-1, 24, 2).all(axis=1).mean(axis=0)
                vals = [*hits.mean(axis=0), e["width"].mean(), *daily,
                        *(stats[m] for m in METRICS[5:])]
                assert np.isclose(sum(vals[5:8]), 1)
                for paired_seed in CONFIG["seeds"] if deterministic else [seed]:
                    records.append(dict(window=origin, scenario=scenario, seed=paired_seed, interval=method,
                                        **dict(zip(METRICS, vals))))
                if method == "split" and scenario in ("bias", "slow_ramp"):
                    mask = e["fault"] & ~e["detectors"][:, 1] & ~e["detectors"][:, 2]
                    subtle.append(dict(window=origin, scenario=scenario, corrupt=int(e["fault"].sum()),
                                       missed=int(mask.sum()), full=int((mask & (commands[:, 2] == 2)).sum()),
                                       active=int(commands[mask, 4].sum())))
    return records, subtle, dict(window=origin, n=len(ages), load_q=q[0], renewable_q=q[1])


def stale_window(origin):
    df = prepare(DESIGN["region"])
    model = model_for(df, DESIGN["region"], origin)
    streams = [observations(df, origin, s, CONFIG["seeds"][0]) for s in ("clean", "delayed")]
    ci, di = [o["indices"] for o in streams]
    c, d = streams
    paired = c["available"][ci] & d["available"][di] & (c["source"][ci] == ci) & (d["source"][di] == di - 2)
    rows = []
    for decay in GRID["decay_hours"]:
        for pairs in GRID["completed_pairs"]:
            cs, ds = [evidence(o, model, decay=decay, lookback=pairs)["scores"][:, 0] for o in streams]
            for hour in np.flatnonzero(paired):
                rows.append((origin, hour, decay, pairs, cs[hour], ds[hour]))
    return rows


def stale_summary(values):
    dev = values[values.window.isin(CONFIG["development_windows"])]
    test = values[values.window.isin(CONFIG["windows"])]
    records, monthly = [], []
    for (decay, pairs), g in test.groupby(["decay", "pairs"]):
        clean = dev.loc[(dev.decay == decay) & (dev.pairs == pairs), "current"]
        cutoff = np.quantile(clean, GRID["development_fpr_target"], method="linear")
        true_alarm, false_alarm = g.replay < cutoff, g.current < cutoff
        for origin, m in g.groupby("window"):
            detected = m[m.replay < cutoff]
            monthly.append(dict(decay=decay, pairs=pairs, window=origin, rows=len(m),
                                tp=len(detected), fp=int((m.current < cutoff).sum()),
                                first_alarm_h=int(detected.hour.min()) if len(detected) else None))
        delays = [r["first_alarm_h"] for r in monthly if r["decay"] == decay and r["pairs"] == pairs
                  and r["first_alarm_h"] is not None]
        labels = np.r_[np.zeros(len(g)), np.ones(len(g))]
        records.append(dict(decay=decay, pairs=pairs, cutoff=cutoff, ndev=len(clean),
                            dev_fpr=float((clean < cutoff).mean()), positives=len(g), negatives=len(g),
                            tp=int(true_alarm.sum()), fn=int((~true_alarm).sum()), fp=int(false_alarm.sum()),
                            tn=int((~false_alarm).sum()), recall=true_alarm.mean(), fpr=false_alarm.mean(),
                            auc=roc_auc_score(labels, -np.r_[g.current, g.replay]), detected=len(delays),
                            episodes=g.window.nunique(), median_delay_h=np.median(delays) if delays else np.nan))
    return pd.DataFrame(records), pd.DataFrame(monthly)


def main():
    dest = args.output
    dest.mkdir(parents=True, exist_ok=True)
    assert len(CONFIG["windows"]) == 9
    assert set(CONFIG["windows"]).isdisjoint(CONFIG["development_windows"])
    assert digest(args.study / "protocol.json") == P["protocol_sha256"]
    assert np.isinf(weighted_correction(np.array([[1., 2.]]), [1])).all()
    assert (weighted_correction(np.zeros((100, 2)), np.arange(1, 101)) == 0).all()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        intervals = list(pool.map(interval_window, CONFIG["windows"]))
        stale = list(pool.map(stale_window, CONFIG["development_windows"] + CONFIG["windows"]))
    runs = pd.DataFrame([r for block, _, _ in intervals for r in block])
    assert len(runs) == 9 * 8 * 5 * 2
    summary = runs.groupby(["scenario", "interval"])[METRICS].mean().reset_index()
    archived = pd.read_csv(RESULT / "audit_v2/interval_authorization.csv")
    split = summary.query("interval == 'split'").merge(archived.query("interval == 'split'"),
                                                     on=["scenario", "interval"], validate="one_to_one")
    errors = {m: float((split[m + "_x"] - split[m + "_y"]).abs().max()) for m in METRICS}
    assert len(split) == 8 and max(errors.values()) < 1e-8, errors
    values = pd.DataFrame([r for block in stale for r in block],
                          columns=["window", "hour", "decay", "pairs", "current", "replay"])
    st, months = stale_summary(values)
    current = values[
        (values.decay == 2) & (values.pairs == 24) & values.window.isin(CONFIG["windows"])]
    # Paired stream eligibility guarantees source ages zero and two, respectively.
    age = {"tp": len(current), "fp": 0, "fn": 0, "tn": len(current),
           "episodes": current.window.nunique(), "median_delay_h": float(current.groupby("window").hour.min().median())}
    out = dict(interval_runs=runs, interval_summary=summary, calibration=pd.DataFrame([r for _, _, r in intervals]),
               stale_summary=st, stale_months=months, stale_scores=values,
               subtle_months=pd.DataFrame([r for _, block, _ in intervals for r in block]))
    for name, frame in out.items():
        frame.to_csv(dest / (name + ".csv"), index=False)
    sources = ["data/DE_LU_causal.csv", "protocol.json", "results/frozen_parameters.json",
               "data_pipeline.py", "run_study.py", "control.py", "physics.py", "results/audit_v2/interval_authorization.csv"]
    manifest = dict(reproduced_split_max_abs_difference=errors, age_reference=age,
                    input_hashes={p: digest(args.study / p) for p in sources},
                    design_sha256=digest(HERE / "design.json"), code_sha256=digest(__file__),
                    output_sha256={p.name: digest(p) for p in sorted(dest.glob("*.csv"))})
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(st.to_string(index=False), flush=True)
    print(summary[summary.scenario == "delayed"].to_string(index=False))
    print(out["subtle_months"].groupby("scenario").sum(numeric_only=True))


if __name__ == "__main__":
    main()
