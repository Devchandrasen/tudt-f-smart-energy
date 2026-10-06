"""Verify archived hashes and numerical invariants without a power-flow rerun."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math

PROJECT = Path(__file__).resolve().parents[2]

def protect_reference(path):
    reference = (PROJECT / "results/reference").resolve()
    target = Path(path).resolve()
    if target == reference or reference in target.parents:
        raise ValueError("The archived reference directory is read-only. Choose another output directory.")

def verify_hashes(root, manifest):
    root = Path(root).resolve()
    entries = json.loads(Path(manifest).read_text(encoding="utf-8"))
    for relative, expected in entries.items():
        file = (root / relative).resolve()
        if root not in file.parents:
            raise ValueError(f"Manifest path escapes its root: {relative}")
        actual = hashlib.sha256(file.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch: {relative}")
    return len(entries)

def verify_decisions(directory):
    directory = Path(directory)
    meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    capacities = {n["network"]: n["capacity_mwh"] for n in meta["networks"]}
    totals = {}
    count = 0
    max_soc_error = max_score_error = 0.0
    with (directory / "decisions.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            count += 1
            def number(name):
                value = float(row[name])
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite {name} at record {count}")
                return value
            u, soc, new_soc = number("command_mw"), number("soc"), number("next_soc")
            expected_soc = soc + (-u / .94 if u >= 0 else -.94 * u) / capacities[row["network"]]
            max_soc_error = max(max_soc_error, abs(expected_soc - new_soc))
            if not .15 - 1e-9 <= new_soc <= .9 + 1e-9:
                raise ValueError(f"SOC bound failure at record {count}")
            score = sum(number(name) * w for name, w in zip(
                ["freshness", "completeness", "maturity", "agreement"], [.30, .25, .25, .20]))
            max_score_error = max(max_score_error, abs(score - number("trust")))
            if row["controller"] == "Trust gate":
                expected = "full" if score >= .85 else ("cautious" if score >= .65 else "hold")
                if expected != row["regime"]:
                    raise ValueError(f"Gate regime failure at record {count}")
            physical = (number("vmin") >= .95 - 1e-7 and number("vmax") <= 1.05 + 1e-7
                        and number("max_thermal_utilisation") <= 1 + 1e-7)
            if int(physical) != int(row["plant_feasible"]):
                raise ValueError(f"Physical-label failure at record {count}")
            if abs(u) > 1e-9 and not int(row["model_feasible"]):
                raise ValueError(f"Unchecked nonzero command at record {count}")
            number("grid_mw"); number("loss_mw")
            key = (row["network"], row["block"], row["scenario"], row["seed"], row["controller"])
            stats = totals.setdefault(key, [0, 0, 0., 0.])
            stats[0] += 1
            stats[1] += int(not physical)
            stats[2] += number("cost_eur")
            stats[3] += number("loss_mw")
    expected_count = (len(meta["networks"]) * len(meta["blocks"]) * len(meta["scenarios"])
                      * len(meta["seeds"]) * meta["block_hours"] * 4)
    if count != expected_count or count != meta["decision_count"]:
        raise ValueError(f"Expected {expected_count} records, found {count}")
    if max_soc_error >= 1e-10 or max_score_error >= 1e-10:
        raise ValueError("Numerical equation check failed")
    with (directory / "block_results.csv").open(newline="", encoding="utf-8") as stream:
        blocks = list(csv.DictReader(stream))
    if len(blocks) != len(totals):
        raise ValueError("Block summary count differs from decisions")
    for row in blocks:
        key = tuple(row[n] for n in ["network", "block", "scenario", "seed", "controller"])
        n, failures, cost, loss = totals[key]
        for name, value, tolerance in [("violation_rate", failures/n, 1e-12),
                                       ("cost_eur", cost, 1e-7), ("loss_mwh", loss, 1e-9)]:
            if abs(float(row[name]) - value) > tolerance:
                raise ValueError(f"Block summary mismatch: {key}, {name}")
    return {"decision_records": count, "blocks_checked": len(totals),
            "maximum_soc_equation_error": max_soc_error, "maximum_score_error": max_score_error}

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/reference"))
    parser.add_argument("--skip-hashes", action="store_true", help="For newly generated results without an archive manifest.")
    args = parser.parse_args(argv)
    manifest = args.results / "SHA256.json"
    if not args.skip_hashes:
        files = verify_hashes(args.results, manifest)
        print(f"Verified SHA-256 hashes for {files} archived files.")
    print(json.dumps(verify_decisions(args.results), indent=2))
