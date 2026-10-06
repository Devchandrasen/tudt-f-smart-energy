"""Checks for information timing, battery physics and command acceptance."""

from pathlib import Path
import unittest
import subprocess
import sys
import numpy as np
import pandas as pd

from tudtf.study import (
    Grid,
    chronological_partitions,
    corrupt,
    evidence_channels,
    features,
    gate_regime,
    next_soc,
    energy_cost,
    run_block,
)
from tudtf.verify import protect_reference


class TimingTests(unittest.TestCase):
    def test_run_help_describes_run_options(self):
        result = subprocess.run(
            [sys.executable, "-m", "tudtf", "run", "--help"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("--output", result.stdout)
        self.assertIn("--quick", result.stdout)

    def test_targets_stay_inside_chronological_partitions(self):
        dates = pd.date_range("2016-01-01", periods=8784, freq="h")
        train, cal = chronological_partitions(dates)
        self.assertEqual(train[0], 24)
        self.assertTrue((dates[train + 1] < pd.Timestamp("2016-09-01")).all())
        self.assertTrue((dates[cal] >= pd.Timestamp("2016-09-01")).all())
        self.assertTrue((dates[cal + 1] < pd.Timestamp("2016-11-01")).all())
        self.assertFalse(set(train).intersection(cal))
        self.assertEqual(len(cal), 1463)

    def test_features_ignore_future_observations(self):
        dates = pd.date_range("2016-11-01", periods=60, freq="h")
        x = np.arange(120, dtype=float).reshape(60, 2)
        before = features(x, dates, [25])
        altered = x.copy()
        altered[26:] = 1e6
        np.testing.assert_array_equal(before, features(altered, dates, [25]))
        np.testing.assert_array_equal(before[0, :6], np.r_[x[25], x[24], x[1]])

    def test_evidence_ignores_future_meter_and_metadata_values(self):
        obs = np.full((40, 2), 0.5)
        ref = obs + 0.01
        present = np.ones(40, dtype=bool)
        age = np.zeros(40)
        args = [obs, ref, present, age, 25, [], np.array([0.04, 0.05]), 0.016]
        before = evidence_channels(*args)
        obs[26:] = 1e6
        ref[26:] = -1e6
        present[26:] = False
        age[26:] = 1000
        np.testing.assert_array_equal(before, evidence_channels(*args))

    def test_completed_bad_residuals_reduce_maturity(self):
        obs = np.ones((30, 2))
        other = [obs, obs.copy(), np.ones(30, bool), np.zeros(30), 25]
        good = evidence_channels(
            *other, [(np.ones(2), np.ones(2))], np.array([0.04, 0.05]), 0.016
        )
        bad = evidence_channels(
            *other, [(np.zeros(2), np.ones(2))], np.array([0.04, 0.05]), 0.016
        )
        self.assertGreater(good[2], bad[2])

    def test_missing_data_is_forward_carried(self):
        true = np.arange(240, dtype=float).reshape(120, 2) / 240
        obs, ref, present, age = corrupt(true, "Missing", 11, 400)
        self.assertTrue((~present[1:]).any())
        for i in range(1, len(true)):
            if not present[i]:
                np.testing.assert_array_equal(obs[i], obs[i - 1])
                self.assertEqual(age[i], age[i - 1] + 1)
            else:
                np.testing.assert_array_equal(obs[i], true[i])

    def test_measurement_seed_is_reproducible(self):
        true = np.full((100, 2), 0.5)
        for left, right in zip(
            corrupt(true, "Noisy", 29, 10), corrupt(true, "Noisy", 29, 10)
        ):
            np.testing.assert_array_equal(left, right)


class PhysicsTests(unittest.TestCase):
    def test_charge_discharge_and_hold_units(self):
        self.assertAlmostEqual(next_soc(0.55, 0.8, 3.2), 0.55 - 0.8 / (0.94 * 3.2))
        self.assertAlmostEqual(next_soc(0.55, -0.8, 3.2), 0.55 + 0.94 * 0.8 / 3.2)
        self.assertEqual(next_soc(0.55, 0, 3.2), 0.55)

    def test_round_trip_loses_energy(self):
        charged = next_soc(0.55, -0.4, 3.2)
        final = next_soc(charged, 0.4, 3.2)
        self.assertLess(final, 0.55)

    def test_energy_cost_import_and_export_units(self):
        self.assertAlmostEqual(energy_cost(1, 0, 0.15), 150)
        self.assertAlmostEqual(energy_cost(-1, 0, 0.15), -40)
        expected = 150 + 10 * 0.4 + 130 * 0.4 / 0.94
        self.assertAlmostEqual(energy_cost(1, 0.4, 0.15), expected)

    def test_physical_label_checks_voltage_and_thermal_limits(self):
        grid = object.__new__(Grid)
        grid.nb = 2
        cases = [
            ([0.96, 1.04, 0.99, 1.0, 0.01], 1),
            ([0.949, 1.04, 0.99, 1.0, 0.01], 0),
            ([0.96, 1.051, 0.99, 1.0, 0.01], 0),
            ([0.96, 1.04, 1.01, 1.0, 0.01], 0),
        ]
        for values, expected in cases:
            self.assertEqual(grid.metrics(np.array(values))[0], expected)

    def test_nonfinite_power_flow_is_infeasible(self):
        grid = object.__new__(Grid)
        grid.nb = 2
        self.assertEqual(grid.metrics(np.array([np.nan, 1, 0.5, 1, 0.01]))[0], 0)


class GateTests(unittest.TestCase):
    def test_exact_regime_thresholds(self):
        self.assertEqual(gate_regime(0.649999), "hold")
        self.assertEqual(gate_regime(0.65), "cautious")
        self.assertEqual(gate_regime(0.849999), "cautious")
        self.assertEqual(gate_regime(0.85), "full")

    def test_scaled_command_is_checked_and_hold_can_fail(self):
        class Forecast:
            def predict(self, x):
                return np.ones((len(x), 2))

        class FakeGrid:
            name = "IEEE 33"
            pmax = 0.8
            capacity = 3.2

            def approximate(self, load, generation, power):
                return (1, 0.0, 1 - power, 0.01, 0.96, 1.0, 0.5)

            def evaluate(self, load, generation, power):
                # The selected full command passes, half power does not, and hold
                # fails as well. This makes the actual-command recheck observable.
                ok = abs(power - 0.8) < 1e-9
                return (
                    int(ok),
                    0.0 if ok else 0.02,
                    1 - power,
                    0.01,
                    0.96 if ok else 0.93,
                    1.0,
                    0.5,
                )

        dates = pd.date_range("2016-11-01 16:00", periods=30, freq="h")
        rows, _ = run_block(
            FakeGrid(),
            dates,
            np.ones((30, 2)),
            Forecast(),
            np.array([0.04, 0.05]),
            0.016,
            24,
            1,
            "Delayed",
            11,
            np.array([0.30, 0.25, 0.25, 0.20]),
            0.65,
            0.85,
        )
        trust = next(r for r in rows if r["controller"] == "Trust gate")
        self.assertEqual(trust["regime"], "cautious")
        self.assertAlmostEqual(trust["proposal_mw"], 0.8)
        self.assertEqual(trust["command_mw"], 0.0)
        self.assertEqual(trust["model_feasible"], 0)
        self.assertEqual(trust["plant_feasible"], 0)

    def test_reference_is_protected_from_overwrite(self):
        root = Path(__file__).resolve().parents[1]
        with self.assertRaises(ValueError):
            protect_reference(root / "results/reference")
        with self.assertRaises(ValueError):
            protect_reference(root / "results/reference/new-run")


if __name__ == "__main__":
    unittest.main()
