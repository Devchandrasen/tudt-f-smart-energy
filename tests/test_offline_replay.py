"""Reproduce the archived fixed-recording summaries from the public branch inputs."""
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


def test_replay_outputs(tmp_path):
    replay = Path(__file__).resolve().parents[1] / "evidence/offline_replay"
    subprocess.run([sys.executable, str(replay / "summarize.py"), str(tmp_path)], check=True)
    for filename in ("weekly.csv", "midpoint.csv"):
        expected = pd.read_csv(replay / "reference" / filename)
        actual = pd.read_csv(tmp_path / filename)
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    for filename in ("first_alarm.json", "result.json"):
        assert json.loads((tmp_path / filename).read_bytes()) == json.loads((replay / "reference" / filename).read_bytes())
