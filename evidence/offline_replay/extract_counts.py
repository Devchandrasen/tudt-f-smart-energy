"""Extract per-window detector and command counts from the pinned numerical study."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from metrics import record_counts


def controller_trace(outputs, columns, metadata, scenario, method, seeds):
    runs = [(i, r["seed"]) for i, r in enumerate(metadata)
            if r["scenario"] == scenario and r["method"] == method]
    if sorted(seed for _, seed in runs) != sorted(seeds):
        raise ValueError(f"Missing or duplicate runs for {scenario}/{method}")
    first = outputs[runs[0][0]]
    for i, _ in runs[1:]:
        np.testing.assert_array_equal(first, outputs[i])
    return pd.DataFrame(first, columns=columns)


def extract(study):
    sys.path.insert(0, str(study.resolve()))
    from audit_experiments import CONFIG, pack

    manifest = json.loads((study / "FILES_SHA256.json").read_text())
    fixed = pd.read_csv(study / "evidence/results/subtle_months.csv").set_index(["window", "scenario"])
    rows = []
    for window in CONFIG["windows"]:
        name = f"results/audit_v2/primary_DE_LU_{window}_traces.npz"
        path = study / name
        with path.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != manifest[name]:
                raise ValueError(f"Archived trace checksum mismatch: {name}")
        with np.load(path, allow_pickle=False) as source:
            data = source["outputs"]
            columns = source["columns"].tolist()
            metadata = json.loads(source["metadata"].item())
        for scenario in ("bias", "slow_ramp"):
            hours = CONFIG["primary_hours"]
            gated, ungated = [controller_trace(data, columns, metadata, scenario, method, CONFIG["seeds"])
                              for method in ("original_composite", "ungated")]
            obs = pack("DE_LU", window, hours, scenario, seed=CONFIG["seeds"][0])
            fault = obs["fault"].astype(bool)
            alarms = obs["detectors"].astype(bool)
            hits = np.flatnonzero(alarms[:, 1:3].any(axis=1))
            end = int(hits[0]) + 1 if hits.size else hours
            for horizon, length in (("half", hours // 2), ("full", hours), ("first_alarm", end)):
                counts = record_counts(alarms, fault, gated, ungated, length)
                if horizon == "full":
                    observed = [counts[k] for k in ("corrupt", "missed", "missed_full", "missed_active")]
                    reference = fixed.loc[(window, scenario), ["corrupt", "missed", "full", "active"]]
                    np.testing.assert_array_equal(observed, reference)
                rows.append({"window": window, "scenario": scenario, "horizon": horizon, **counts})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()
    counts = extract(args.study)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(args.output_csv, index=False, lineterminator="\n")
