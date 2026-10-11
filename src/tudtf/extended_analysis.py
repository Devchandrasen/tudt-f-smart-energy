"""Audit and derive tables from the additional rolling-origin dispatch experiment."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import pandapower as pp
from .study import profiles, next_soc
from .extended import DiagnosticGrid, AssetGrid, asset_profiles, aggregate, duration, MODES, regime, SCORE_WEIGHTS


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=Path("results/epsr-2026"))
    args = ap.parse_args(argv)
    out = args.results.resolve()
    meta = json.loads((out / "metadata.json").read_text())
    columns = pd.read_csv(out / "decisions.csv", nrows=0).columns
    categories = ["network", "origin", "block", "scenario", "controller", "regime"]
    integers = ["seed", "hour", "primary_available", "reference_available", "reference_fallback",
                "model_feasible", "rejected_scaled", "plant_feasible", "undervoltage_buses",
                "overvoltage_buses", "overloaded_elements", "lowest_voltage_bus"]
    dtype = {**{c: "category" for c in categories}, **{c: "int16" for c in integers}}
    d = pd.read_csv(out / "decisions.csv", usecols=[c for c in columns if c not in ["state_json", "plant_json"]], dtype=dtype)
    assert len(d) == meta["decision_count"]
    d["violation"] = 1 - d.plant_feasible
    d["authorised"] = d.regime != "hold"
    d["nonzero"] = abs(d.command_mw) > 1e-9
    cols = ["network", "origin", "block", "scenario", "seed", "controller"]
    b = d.groupby(cols, sort=False, observed=True).agg(
        hours=("hour", "count"), violations=("violation", "sum"),
        violation_rate=("violation", "mean"), authorised_rate=("authorised", "mean"),
        nonzero_rate=("nonzero", "mean"), full_rate=("regime", lambda v: (v == "full").mean()),
        severity=("severity", "mean"), maximum_severity=("severity", "max"),
        longest_violation_hours=("violation", duration), cost_eur=("cost_eur", "sum"),
        loss_mwh=("loss_mw", "sum"), minimum_voltage=("vmin", "min"),
        maximum_voltage=("vmax", "max"), maximum_thermal=("max_thermal_utilisation", "max"),
        undervoltage_bus_hours=("undervoltage_buses", "sum"),
        overvoltage_bus_hours=("overvoltage_buses", "sum"),
        overload_element_hours=("overloaded_elements", "sum"), terminal_soc=("next_soc", "last")
    ).reset_index()
    b.to_csv(out / "block_results.csv", index=False)
    overall = b.groupby(["network", "controller"], sort=False, observed=True).agg(
        hours=("hours", "sum"), violations=("violations", "sum"),
        violation_rate=("violation_rate", "mean"), authorised_rate=("authorised_rate", "mean"),
        nonzero_rate=("nonzero_rate", "mean"), full_rate=("full_rate", "mean"),
        cost_eur=("cost_eur", "mean"), loss_mwh=("loss_mwh", "mean"),
        severity=("severity", "mean"), maximum_severity=("maximum_severity", "max"),
        longest_violation_hours=("longest_violation_hours", "max"),
        minimum_voltage=("minimum_voltage", "min"), maximum_voltage=("maximum_voltage", "max"),
        maximum_thermal=("maximum_thermal", "max"),
        undervoltage_bus_hours=("undervoltage_bus_hours", "sum"),
        overvoltage_bus_hours=("overvoltage_bus_hours", "sum"),
        overload_element_hours=("overload_element_hours", "sum"), terminal_soc=("terminal_soc", "mean")
    ).reset_index()
    overall.to_csv(out / "overall.csv", index=False)
    numeric = b.select_dtypes("number").columns.difference(["seed", "hours"])
    scenario = b.groupby(["network", "origin", "scenario", "controller"], observed=True)[numeric].mean().reset_index()
    scenario.to_csv(out / "scenario_results.csv", index=False)
    rng = np.random.default_rng(20261011)
    paired = []
    for network in d.network.unique():
        t = b[b.network == network].groupby(["origin", "block", "controller"], observed=True)[
            ["violation_rate", "cost_eur", "loss_mwh", "nonzero_rate"]].mean()
        for left, right in [("Composite ternary", "Ungated"),
                            ("Composite ternary", "Agreement ternary"),
                            ("Composite binary", "Agreement binary"),
                            ("Composite ternary", "Always half"),
                            ("Always half", "Ungated"),
                            ("Stress screen", "Always half"),
                            ("Reference forecast", "Ungated"),
                            ("Stress screen", "Ungated")]:
            diff = t.xs(left, level="controller") - t.xs(right, level="controller")
            n = len(diff)
            samples = np.zeros((10000, 4))
            for origin in diff.index.get_level_values("origin").unique():
                values = diff.xs(origin, level="origin").to_numpy()
                ix = rng.integers(0, len(values), size=(10000, len(values)))
                samples += values[ix].sum(axis=1) / n
            for j, metric in enumerate(diff.columns):
                lo, hi = np.quantile(samples[:, j], [.025, .975])
                paired.append(dict(network=network, left=left, right=right, metric=metric,
                                   estimate=float(diff.iloc[:, j].mean()), lower=float(lo), upper=float(hi),
                                   temporal_blocks=n, resamples=10000))
    pd.DataFrame(paired).to_csv(out / "paired_uncertainty.csv", index=False)

    # Common realised proposals from the ungated trajectory isolate score ranking.
    replay = d[(d.controller == "Ungated") & (d.scenario != "Clean") & (d.reference_available == 1)].copy()
    ranks = []
    for network in replay.network.unique():
        t = replay[replay.network == network].copy()
        ties = np.random.default_rng(1729).random(len(t))
        parts = t[["freshness", "completeness", "maturity", "agreement"]].to_numpy()
        scores = {"Agreement": t.agreement.to_numpy(), **{key: parts @ w for key, w in SCORE_WEIGHTS.items()}}
        for key, values in scores.items():
            order = np.lexsort((ties, -values))
            for coverage in [.5, .75, .9]:
                count = int(round(coverage * len(t)))
                selected = t.iloc[order[:count]]
                ranks.append(dict(network=network, score=key, coverage=count / len(t),
                                  samples=count, violations=int(selected.violation.sum()),
                                  risk=float(selected.violation.mean()), severity=float(selected.severity.mean())))
    pd.DataFrame(ranks).to_csv(out / "matched_coverage.csv", index=False)
    replay.to_csv(out / "common_proposals.csv", index=False)

    cap = d.network.map({"IEEE 33": 3.2, "SimBench MV": 8., "SimBench assets": 8.}).to_numpy()
    calculated = d.soc.to_numpy() + np.where(d.command_mw >= 0, -d.command_mw / .94, -.94 * d.command_mw) / cap
    audit = dict(decision_count=len(d), maximum_soc_error=float(np.max(np.abs(calculated - d.next_soc))),
                 soc_bounds_errors=int(((d.next_soc < .15 - 1e-9) | (d.next_soc > .9 + 1e-9)).sum()),
                 nonzero_model_infeasible=int(((abs(d.command_mw) > 1e-9) & (d.model_feasible == 0)).sum()),
                 nonfinite_outcomes=int(d[["vmin", "vmax", "loss_mw", "grid_mw"]].isna().any(axis=1).sum()))
    proposal_soc = d.soc.to_numpy() + np.where(d.proposal_mw >= 0, -d.proposal_mw / .94, -.94 * d.proposal_mw) / cap
    audit['proposal_battery_bounds_errors'] = int(((proposal_soc < .15 - 1e-9) | (proposal_soc > .9 + 1e-9)).sum())
    expected_command = d.proposal_mw * np.where(d.regime == 'full', 1., np.where(d.regime == 'cautious', .5, 0.))
    expected_command = np.where(d.rejected_scaled == 1, 0., expected_command)
    audit['maximum_command_rule_error'] = float(np.max(abs(expected_command - d.command_mw)))
    audit['maximum_score_error'] = float(np.max(abs(d[['freshness', 'completeness', 'maturity', 'agreement']].to_numpy() @ np.array([.3, .25, .25, .2]) - d.composite)))
    physical = (d.vmin >= .95 - 1e-7) & (d.vmax <= 1.05 + 1e-7) & (d.max_thermal_utilisation <= 1 + 1e-7)
    audit["physical_label_disagreements"] = int((physical.astype(int) != d.plant_feasible).sum())
    infos = {(x["network"], x["origin"]): x for x in meta["studies"]}
    errors = 0
    for (network, origin), t in d.groupby(["network", "origin"], observed=True):
        cutoffs = infos[network, origin]["cutoffs"]
        for mode, rows in t.groupby("controller", observed=True):
            for row in rows.itertuples():
                parts = np.array([row.freshness, row.completeness, row.maturity, row.agreement])
                expected = regime(parts, mode, cutoffs, row.reference_available, row.primary_available, row.primary_age)
                errors += expected != row.regime
    audit["regime_disagreements"] = int(errors)
    audit["reference_fallback_disagreements"] = int((d.reference_fallback != (
        (d.controller == "Reference forecast") & (d.reference_available == 0)).astype(int)).sum())
    dates, scalar, net = profiles()
    assets, names, lp, lq, gp = asset_profiles(net)
    pd.DataFrame(scalar, index=dates, columns=['load_factor', 'generation_factor']).rename_axis('time').to_csv(out / 'hourly_aggregate_profiles.csv')
    pd.DataFrame(assets, index=dates, columns=names).rename_axis('time').to_csv(out / 'hourly_asset_profiles.csv')
    maxdiff, labeldiff, fresh, model_labeldiff = 0., 0, 0, 0
    example_ids = set()
    for (network, origin), t in d.groupby(["network", "origin"], observed=True):
        examples = pd.concat([t.sample(12, random_state=71), t[t.plant_feasible == 0].head(6),
                              t[(t.regime == "hold") & (t.plant_feasible == 0)].head(2)]).drop_duplicates()
        example_ids.update(examples.index)
    states, forecast_errors = {}, {}
    for chunk in pd.read_csv(out / "decisions.csv", usecols=["network", "origin", "controller", "scenario", "seed", "state_json", "plant_json"], chunksize=20000):
        selected = chunk.loc[chunk.index.isin(example_ids)]
        states.update({index: (row.plant_json, row.state_json) for index, row in selected.iterrows()})
        clean = chunk[(chunk.controller == 'Ungated') & (chunk.scenario == 'Clean') & (chunk.seed == 11)]
        for row in clean.itertuples():
            pred, true = np.array(json.loads(row.state_json)), np.array(json.loads(row.plant_json))
            if row.network == 'SimBench assets':
                pred, true = aggregate(pred[None, :], net, (lp, lq, gp))[0], aggregate(true[None, :], net, (lp, lq, gp))[0]
            forecast_errors.setdefault((row.network, row.origin), []).append(abs(pred - true))
    forecast_table = []
    for (network, origin), values in forecast_errors.items():
        err = np.array(values)
        info = infos[network, origin]['forecast']
        forecast_table.append(dict(network=network, origin=origin, unique_test_hours=len(err),
            load_mae=float(err[:, 0].mean()), generation_mae=float(err[:, 1].mean()),
            joint_coverage=float(np.mean(np.all(err <= np.array(info['joint_halfwidth']), axis=1))),
            calibration_mae=info['calibration_mae'], calibration_coverage=info['calibration_coverage'],
            q_load=info['joint_halfwidth'][0], q_generation=info['joint_halfwidth'][1]))
    pd.DataFrame(forecast_table).to_csv(out / 'forecast_validation.csv', index=False)
    for (network, origin), t in d.loc[sorted(example_ids)].groupby(["network", "origin"], observed=True):
        grid = AssetGrid(net, (lp, lq, gp), infos[network, origin]["centre"]) if network == "SimBench assets" else DiagnosticGrid(network, net)
        for row in t.itertuples():
            state = np.array(json.loads(states[row.Index][0]))
            if network == "SimBench assets":
                grid.set_vector(state, row.command_mw)
            else:
                grid.set_state(*state, row.command_mw)
            pp.runpp(grid.net, numba=False, init="auto", recycle=None, max_iteration=50)
            a = np.r_[grid.net.res_bus.vm_pu, grid.net.res_line.loading_percent / 100,
                      grid.net.res_trafo.loading_percent / 100, grid.net.res_ext_grid.p_mw.sum(),
                      grid.net.res_line.pl_mw.sum() + grid.net.res_trafo.pl_mw.sum()]
            actual = grid.metrics(a)
            recorded = [row.severity, row.grid_mw, row.loss_mw, row.vmin, row.vmax, row.max_thermal_utilisation]
            maxdiff = max(maxdiff, float(np.max(abs(np.array(actual[1:7]) - recorded))))
            labeldiff += actual[0] != row.plant_feasible
            fresh += 1
            forecast_state = np.array(json.loads(states[row.Index][1]))
            if network == "SimBench assets":
                grid.set_vector(forecast_state, row.command_mw)
            else:
                grid.set_state(*forecast_state, row.command_mw)
            pp.runpp(grid.net, numba=False, init="auto", recycle=None, max_iteration=50)
            mv = np.r_[grid.net.res_bus.vm_pu, grid.net.res_line.loading_percent / 100,
                       grid.net.res_trafo.loading_percent / 100, grid.net.res_ext_grid.p_mw.sum(),
                       grid.net.res_line.pl_mw.sum() + grid.net.res_trafo.pl_mw.sum()]
            model_labeldiff += grid.metrics(mv)[0] != row.model_feasible
    audit.update(fresh_ac_samples=fresh, fresh_model_ac_samples=fresh,
                 maximum_fresh_ac_difference=maxdiff, fresh_ac_label_disagreements=int(labeldiff),
                 fresh_model_label_disagreements=int(model_labeldiff))
    assert audit["maximum_soc_error"] < 1e-10
    assert audit['maximum_command_rule_error'] < 1e-10
    assert audit['maximum_score_error'] < 1e-10
    assert maxdiff < 1e-6
    for key in ["soc_bounds_errors", "proposal_battery_bounds_errors", "nonzero_model_infeasible", "nonfinite_outcomes",
                "physical_label_disagreements", "regime_disagreements", "reference_fallback_disagreements",
                "fresh_ac_label_disagreements", "fresh_model_label_disagreements"]:
        assert audit[key] == 0, (key, audit[key])
    (out / "validation.json").write_text(json.dumps(audit, indent=2))
    # Terminal valuation sensitivity for fixed recorded commands, no redispatch claim.
    value_rows = []
    for row in b.itertuples():
        capacity = 3.2 if row.network == "IEEE 33" else 8.
        change = capacity * (row.terminal_soc - .55)
        for value in [90., 130., 170.]:
            value_rows.append(dict(network=row.network, origin=row.origin, block=row.block,
                scenario=row.scenario, seed=row.seed, controller=row.controller, value_eur_mwh=value,
                cost_eur=row.cost_eur - (value - 130.) * change))
    pd.DataFrame(value_rows).groupby(["network", "controller", "value_eur_mwh"]).cost_eur.mean().reset_index().to_csv(out / "terminal_valuation.csv", index=False)
    manifest = {}
    for p in sorted(out.glob("*")):
        if p.is_file() and p.name != "SHA256.json":
            digest = hashlib.sha256()
            with p.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            manifest[p.name] = digest.hexdigest()
    (out / "SHA256.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(audit, indent=2))
    print(overall[["network", "controller", "violations", "violation_rate", "cost_eur"]].to_string(index=False))


if __name__ == "__main__":
    main()
