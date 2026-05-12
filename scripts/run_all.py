from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("\n$ " + " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    python = sys.executable
    run([python, "src/tudtf_core.py"])
    run([python, "scripts/run_rolling_validation.py"])
    run([python, "scripts/run_extended_analysis.py"])
    run([python, "scripts/run_control_and_runtime.py"])


if __name__ == "__main__":
    main()

