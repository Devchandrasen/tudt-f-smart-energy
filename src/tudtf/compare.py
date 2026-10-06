"""Compare a complete rerun with the published reference study."""

from pathlib import Path
import argparse
import json
import platform
from importlib.metadata import version
import numpy as np
import pandas as pd


def compare(reference, results):
    files = [
        "hourly_profiles.csv",
        "decisions.csv",
        "block_results.csv",
        "summary.csv",
        "proposal_replay.csv",
        "risk_coverage.csv",
    ]
    report = {
        "python": platform.python_version(),
        "dependencies": {
            name: version(name)
            for name in [
                "numpy",
                "pandas",
                "scipy",
                "matplotlib",
                "scikit-learn",
                "pandapower",
                "simbench",
            ]
        },
        "relative_tolerance": 1e-8,
        "absolute_tolerance": 1e-8,
        "files": {},
    }
    for name in files:
        left = pd.read_csv(Path(reference) / name)
        right = pd.read_csv(Path(results) / name)
        if left.shape != right.shape or list(left.columns) != list(right.columns):
            raise ValueError(f"Shape or column mismatch in {name}")
        maxima = {}
        for column in left:
            if pd.api.types.is_numeric_dtype(left[column]):
                a, b = left[column].to_numpy(dtype=float), right[column].to_numpy(
                    dtype=float
                )
                if not np.allclose(a, b, rtol=1e-8, atol=1e-8, equal_nan=True):
                    raise ValueError(f"Numerical mismatch in {name}: {column}")
                finite = np.isfinite(a) & np.isfinite(b)
                maxima[column] = (
                    float(np.max(np.abs(a[finite] - b[finite])))
                    if finite.any()
                    else 0.0
                )
            elif not left[column].equals(right[column]):
                raise ValueError(f"Category mismatch in {name}: {column}")
        report["files"][name] = {
            "rows": len(left),
            "maximum_absolute_difference_by_column": maxima,
        }
    report["passed"] = True
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("results/reference"))
    parser.add_argument("--results", type=Path, default=Path("results/full"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/reproduction_check.json")
    )
    args = parser.parse_args(argv)
    from .verify import protect_reference

    protect_reference(args.output)
    report = compare(args.reference, args.results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Reproduction passed for {len(report['files'])} files; report: {args.output}"
    )


if __name__ == "__main__":
    main()
