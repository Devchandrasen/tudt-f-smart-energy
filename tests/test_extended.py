"""Additional study timing, cutoff, reference fallback and storage checks."""
import unittest
import numpy as np
import pandas as pd
from tudtf.extended import (measurement_streams, fitted_cutoffs, regime, WEIGHTS,
                           SCORE_WEIGHTS, choose, duration, MODES, evidence)


class ExtendedTests(unittest.TestCase):
    def test_missing_reference_is_carried_and_flags_are_retained(self):
        true = np.arange(194).reshape(97, 2) / 194
        _, r, _, ok, _ = measurement_streams(true, "Reference outage", 11, 10, [0], [1])
        self.assertFalse(ok[36:42].any())
        np.testing.assert_array_equal(r[36:42], np.tile(r[35], (6, 1)))
        self.assertTrue(ok[42])

    def test_common_bias_affects_both_meters(self):
        true = np.full((97, 2), .8)
        z, r, *_ = measurement_streams(true, "Common bias", 11, 10, [0], [1])
        self.assertAlmostEqual(z[36, 0], 1.2)
        self.assertAlmostEqual(z[36, 1], .4)
        self.assertGreater(r[36, 0], 1.1)
        self.assertLess(r[36, 1], .5)

    def test_reference_unavailability_forces_all_reference_gates_to_hold(self):
        cutoffs = {"Composite": [.5, .8], "Agreement": [.5, .8],
                   **{k: [.5, .8] for k in SCORE_WEIGHTS}}
        for mode in MODES:
            if mode not in ["Ungated", "Always half", "Metadata", "Reference forecast", "Stress screen"]:
                self.assertEqual(regime(np.ones(4), mode, cutoffs, False, True, 0), "hold")

    def test_coverage_targets_and_ablations(self):
        low, high = fitted_cutoffs(np.arange(100) / 100)
        self.assertAlmostEqual(low, .1485)
        self.assertAlmostEqual(high, .495)
        for j, key in enumerate(list(SCORE_WEIGHTS)[1:]):
            self.assertEqual(SCORE_WEIGHTS[key][j], 0)
            self.assertAlmostEqual(SCORE_WEIGHTS[key].sum(), 1)

    def test_unavailable_reference_is_not_treated_as_agreement(self):
        z = np.ones((30, 2))
        r = z.copy()
        result = evidence(z, r, np.ones(30, bool), np.zeros(30, bool),
                          np.zeros(30), 25, [], [.04, .05], .02)
        self.assertEqual(result[3], 0)

    def test_duration_counts_consecutive_hours(self):
        self.assertEqual(duration([1, 1, 0, 1, 1, 1, 0]), 3)
        self.assertEqual(duration([0, 0]), 0)

    def test_uncertainty_screen_checks_stress_states_before_acceptance(self):
        class FakeGrid:
            pmax, capacity = .8, 3.2
            def approximate(self, load, generation, power):
                return (1, 0., 1. - power, .01, .98, 1., .5)
            def evaluate(self, load, generation, power):
                ok = int(load + power <= 1.1 + 1e-10)
                return (ok, 1. - ok, 1. - power, .01, .98, 1., .5)
        grid = FakeGrid()
        state, q = np.array([.6, .4]), np.array([.2, .1])
        point = choose(grid, state, .8, .24, {})
        stressed = choose(grid, state, .8, .24, {}, q, np.array([0]), np.array([1]))
        self.assertAlmostEqual(point, .4)
        self.assertAlmostEqual(stressed, .2)
        for load in [.4, .8]:
            for gen in [.3, .5]:
                self.assertTrue(grid.evaluate(load, gen, stressed)[0])

    def test_spatial_features_ignore_future_values(self):
        from tudtf.study import features
        x = np.arange(60 * 32, dtype=float).reshape(60, 32)
        dates = pd.date_range('2016-07-01', periods=60, freq='h')
        before = features(x, dates, [25])
        x[26:] = -1e9
        np.testing.assert_array_equal(before, features(x, dates, [25]))

    def test_asset_profiles_match_official_simbench_absolute_values(self):
        import simbench as sb
        from tudtf.extended import asset_profiles
        net = sb.get_simbench_net('1-MV-rural--0-sw')
        values, _, lp, lq, gp = asset_profiles(net)
        official = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
        for element, column, mapping in [('load','p_mw',lp), ('load','q_mvar',lq), ('sgen','p_mw',gp)]:
            expected = official[element, column].to_numpy().reshape(-1, 4, len(mapping)).mean(axis=1)
            actual = values[:, mapping] * net[element][column].to_numpy()
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
