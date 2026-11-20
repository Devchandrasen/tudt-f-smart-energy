"""Aggregate fixed recording counts over the archived replay cases."""
import argparse
import json
from pathlib import Path

import pandas as pd

from metrics import aggregate


KEY = ["window", "scenario", "horizon"]
SCENARIOS = ("bias", "slow_ramp")


def first_alarm_summary(counts):
    obs = counts.xs("first_alarm", level="horizon")
    full = counts.xs("full", level="horizon").reindex(obs.index)
    alarms = obs.joint_tp + obs.joint_fp
    if not alarms.isin([0, 1]).all():
        raise ValueError("A first-alarm prefix can contain at most one positive hour")
    frame = pd.DataFrame({
        "records": 1, "hours": full.hours, "observed_hours": obs.hours,
        "unresolved_hours": full.hours - obs.hours, "no_alarm": 1 - alarms,
        "first_alarm_corrupt": obs.joint_tp, "first_alarm_uncorrupted": obs.joint_fp,
        "corrupt": obs.corrupt, "detected": obs.joint_tp, "granted_full": obs.corrupt_full,
        "issued_nonzero": obs.corrupt_active, "gated_high": obs.gate_high, "ungated_high": obs.ungated_high,
    })
    totals = frame.groupby("scenario").sum().astype(int).to_dict(orient="index")
    totals["Combined"] = frame.sum().astype(int).to_dict()
    return {"complete_live_test": False, "counts": totals}


def summarize(counts_path, output_dir):
    counts = pd.read_csv(counts_path).set_index(KEY, verify_integrity=True).sort_index()
    windows = sorted(set(counts.index.get_level_values("window")))
    expected = pd.MultiIndex.from_product([windows, SCENARIOS, ["full", "half", "first_alarm"]], names=KEY)
    if len(windows) != 9 or set(counts.index) != set(expected):
        raise ValueError("Expected three horizons for each of 18 recordings")
    members = pd.read_csv(Path(__file__).with_name("case_windows.csv"))
    full = counts.xs("full", level="horizon")
    half = counts.xs("half", level="horizon")
    weekly = [aggregate("fixed " + s, full.xs(s, level="scenario").to_dict("records")) for s in SCENARIOS]
    weekly.append(aggregate("equal fixed mixture", full.to_dict("records")))
    midpoint = [aggregate("fixed 84 hours", half.to_dict("records"), full_weeks=0),
                aggregate("fixed 168 hours", full.to_dict("records"), full_weeks=len(full)),
                aggregate("equal fixed horizons", pd.concat([half, full]).to_dict("records"), full_weeks=len(full))]
    for (analysis, case), group in members.groupby(["analysis", "case"], sort=False):
        group = group.sort_values(["window", "scenario"])
        keys = pd.MultiIndex.from_frame(group[KEY])
        rows = counts.loc[keys]
        if analysis == "weekly":
            if group.window.tolist() != windows or set(group.horizon) != {"full"}:
                raise ValueError(f"Incomplete weekly case: {case}")
            prior = rows.iloc[:-1]
            states = (10 * prior.active_hours >= prior.hours).nunique()
            switches = group.scenario.to_numpy()[1:] != group.scenario.to_numpy()[:-1]
            weekly.append(aggregate(case, rows.to_dict("records"), distinct_feedback=int(states),
                                    family_changes=int(switches.sum())))
        elif analysis == "midpoint":
            if not group.horizon.isin(["half", "full"]).all() or list(keys.droplevel("horizon")) != list(full.index):
                raise ValueError(f"Incomplete midpoint case: {case}")
            midpoint.append(aggregate(case, rows.to_dict("records"), full_weeks=int((group.horizon == "full").sum())))
        else:
            raise ValueError(f"Unknown case type: {analysis}")
    if len(weekly) != 7 or len(midpoint) != 5:
        raise ValueError("Expected four weekly and two midpoint cases in addition to the fixed controls")

    weekly_advantage = all(r["gate_high"] < r["ungated_high"] and r["recall"] >= weekly[2]["recall"]
                           for r in weekly[3:])
    midpoint_check = all(r["gate_high"] <= r["ungated_high"] for r in midpoint[3:])
    decision = {"advantage_rule_met": weekly_advantage and midpoint_check, "fixed_runs": len(full),
                "paired_controllers": 2, "windows": len(windows), "pooled_fixed_recall_percent": weekly[2]["recall"]}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in [("weekly", weekly), ("midpoint", midpoint)]:
        pd.DataFrame(rows).to_csv(output_dir / (name + ".csv"), index=False, lineterminator="\n")
    for name, values in [("first_alarm", first_alarm_summary(counts)), ("result", decision)]:
        (output_dir / (name + ".json")).write_text(json.dumps(values, indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--counts", type=Path, default=Path(__file__).with_name("window_counts.csv"))
    args = parser.parse_args()
    summarize(args.counts, args.output_dir)
