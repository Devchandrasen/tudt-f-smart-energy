"""Summarize archived DE-LU net-load errors. No model refitting is required."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def summarize(study):
    cfg = json.loads((study / "protocol.json").read_text())
    runs = []
    hashes = {}
    for month in cfg["windows"]:
        file = study / "results" / f"primary_DE_LU_{month}_traces.npz"
        with file.open("rb") as f:
            hashes[file.name] = hashlib.file_digest(f, "sha256").hexdigest()
        with np.load(file, allow_pickle=False) as arc:
            field = list(arc["columns"]).index("forecast_abs_error")
            trace = arc["outputs"]
            seen = set()
            for idx, row in enumerate(json.loads(arc["metadata"].item())):
                if (row["task"],row["method"],row["interval"]) != ("battery","ungated","split"):
                    continue
                pair = row["scenario"], row["seed"]
                assert pair not in seen
                seen.add(pair)
                error = trace[idx,:,field]
                assert len(error) == cfg["primary_hours"] and np.all(np.isfinite(error)) and np.all(error>=0)
                runs.append((month,*pair,float(error.mean()),float(np.sqrt(np.mean(error**2)))))
            assert seen == {(s,k) for s in cfg["scenarios"] for k in cfg["seeds"]}
    result = []
    for scenario in cfg["scenarios"]:
        vals=np.array([r[-2:] for r in runs if r[1]==scenario])
        result.append((scenario,*vals.mean(axis=0)))
    return result,hashes


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    values,hashes=summarize(args.study)
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/"forecast_errors.csv").open("w",newline="",encoding="utf8") as f:
        writer=csv.writer(f)
        writer.writerow(["scenario","mean_window_mae_kw","mean_window_rmse_kw"])
        writer.writerows(values)
    (args.output/"forecast_trace_hashes.json").write_text(json.dumps(hashes,indent=2),encoding="utf8")
