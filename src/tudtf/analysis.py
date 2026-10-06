from pathlib import Path
import json, re
import numpy as np
import pandas as pd
import pandapower as pp
from .study import Grid, profiles


def main(argv=None):
    """Audit decisions with fresh AC solves and regenerate statistical summaries."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=Path("results/full"))
    ap.add_argument(
        "--output",
        type=Path,
        help="Write derived audit tables here; default: the input directory.",
    )
    args = ap.parse_args(argv)
    P = args.results.resolve()
    out = (args.output or P).resolve()
    from .verify import protect_reference

    protect_reference(out)
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(P / "decisions.csv")
    b = pd.read_csv(P / "block_results.csv")
    r = pd.read_csv(P / "proposal_replay.csv")
    meta = json.loads((P / "metadata.json").read_text())
    assert len(d) == meta["decision_count"]
    audit = {"decision_records": len(d)}
    cap = d.network.map({"IEEE 33": 3.2, "SimBench MV": 8}).to_numpy()
    u = d.command_mw.to_numpy()
    x = d.soc.to_numpy()
    calculated = x + np.where(u >= 0, -u / 0.94, -0.94 * u) / cap
    audit["maximum_soc_equation_error"] = float(abs(calculated - d.next_soc).max())
    audit["soc_out_of_bounds"] = int(
        ((d.next_soc < 0.15 - 1e-9) | (d.next_soc > 0.9 + 1e-9)).sum()
    )
    recalc = d[
        ["freshness", "completeness", "maturity", "agreement"]
    ].to_numpy() @ np.array([0.30, 0.25, 0.25, 0.20])
    audit["maximum_score_error"] = float(abs(recalc - d.trust).max())
    trust_rows = d[d.controller == "Trust gate"]
    expected_regime = np.where(
        trust_rows.trust >= 0.85,
        "full",
        np.where(trust_rows.trust >= 0.65, "cautious", "hold"),
    )
    audit["trust_regime_disagreements"] = int(
        (expected_regime != trust_rows.regime).sum()
    )
    physical = (
        (d.vmin >= 0.95 - 1e-7)
        & (d.vmax <= 1.05 + 1e-7)
        & (d.max_thermal_utilisation <= 1 + 1e-7)
    )
    audit["physical_label_disagreements"] = int(
        (physical.astype(int) != d.plant_feasible).sum()
    )
    audit["nonzero_model_infeasible_commands"] = int(
        ((abs(d.command_mw) > 1e-9) & (d.model_feasible == 0)).sum()
    )
    audit["nonfinite_powerflows"] = int(
        d[["vmin", "vmax", "max_thermal_utilisation", "grid_mw", "loss_mw"]]
        .isna()
        .any(axis=1)
        .sum()
    )
    _, _, net = profiles()
    maxdiff = 0.0
    for name in d.network.unique():
        grid = Grid(name, net)
        for row in d[d.network == name].sample(20, random_state=71).itertuples():
            grid.set_state(
                row.plant_load_factor, row.plant_generation_factor, row.command_mw
            )
            pp.runpp(grid.net, numba=False, init="auto", recycle=None, max_iteration=50)
            values = np.r_[
                grid.net.res_bus.vm_pu,
                grid.net.res_line.loading_percent / 100,
                grid.net.res_trafo.loading_percent / 100,
                grid.net.res_ext_grid.p_mw.sum(),
                grid.net.res_line.pl_mw.sum() + grid.net.res_trafo.pl_mw.sum(),
            ]
            m = grid.metrics(values)
            maxdiff = max(
                maxdiff,
                max(
                    abs(
                        np.array(m[1:])
                        - np.array(
                            [
                                row.severity,
                                row.grid_mw,
                                row.loss_mw,
                                row.vmin,
                                row.vmax,
                                row.max_thermal_utilisation,
                            ]
                        )
                    )
                ),
            )
    audit["fresh_ac_solutions_checked"] = 40
    audit["maximum_fresh_ac_difference"] = float(maxdiff)
    assert (
        audit["soc_out_of_bounds"]
        == audit["physical_label_disagreements"]
        == audit["nonzero_model_infeasible_commands"]
        == audit["nonfinite_powerflows"]
        == audit["trust_regime_disagreements"]
        == 0
    )
    assert (
        audit["maximum_soc_equation_error"] < 1e-10
        and audit["maximum_score_error"] < 1e-10
        and maxdiff < 1e-6
    )
    (out / "validation.json").write_text(json.dumps(audit, indent=2))

    metrics = [
        "violation_rate",
        "cost_eur",
        "authorised_rate",
        "nonzero_rate",
        "loss_mwh",
        "regret_eur",
    ]
    overall = b.groupby(["network", "controller"])[metrics].mean().reset_index()
    overall.to_csv(out / "overall_results.csv", index=False)
    ci = []
    rng = np.random.default_rng(101)
    for name in b.network.unique():
        paired = b[b.network == name].pivot(
            index=["block", "scenario", "seed"], columns="controller", values=metrics
        )
        for metric in metrics:
            delta = paired[metric]["Trust gate"] - paired[metric]["Ungated"]
            blocks = delta.groupby(level="block").mean().to_numpy()
            means = blocks[
                rng.integers(0, len(blocks), size=(10000, len(blocks)))
            ].mean(axis=1)
            ci.append(
                dict(
                    network=name,
                    metric=metric,
                    difference=blocks.mean(),
                    lower=np.quantile(means, 0.025),
                    upper=np.quantile(means, 0.975),
                    temporal_clusters=len(blocks),
                )
            )
    pd.DataFrame(ci).to_csv(out / "paired_uncertainty.csv", index=False)

    channels = r[["freshness", "completeness", "maturity", "agreement"]].to_numpy()
    sensitivity = []
    variants = {
        "Default": np.array([0.30, 0.25, 0.25, 0.20]),
        "Equal weights": np.array([0.25] * 4),
        "Freshness emphasis": np.array([0.40, 0.20, 0.20, 0.20]),
        "No freshness": np.array([0, 0.25, 0.25, 0.20]) / 0.70,
    }
    for name in r.network.unique():
        sel = (r.network == name) & (r.scenario != "Clean")
        for variant, w in variants.items():
            score = channels @ w
            for threshold in [0.65, 0.75, 0.85, 0.95]:
                chosen = sel & (score >= threshold)
                sensitivity.append(
                    dict(
                        network=name,
                        variant=variant,
                        threshold=threshold,
                        coverage=chosen.sum() / sel.sum(),
                        risk=(1 - r.loc[chosen, "plant_feasible"]).mean(),
                    )
                )
    pd.DataFrame(sensitivity).to_csv(out / "score_sensitivity.csv", index=False)
    matched = []
    for name in r.network.unique():
        t = r[(r.network == name) & (r.scenario != "Clean")]
        for fraction in [0.9, 0.75, 0.5]:
            for channel in ["trust", "agreement"]:
                selected = t.sort_values(channel, ascending=False, kind="stable").iloc[
                    : int(round(fraction * len(t)))
                ]
                matched.append(
                    dict(
                        network=name,
                        coverage=fraction,
                        channel=channel,
                        risk=(1 - selected.plant_feasible).mean(),
                        n=len(selected),
                    )
                )
    pd.DataFrame(matched).to_csv(out / "matched_coverage.csv", index=False)
    clean = d[
        (d.network == "IEEE 33")
        & (d.controller == "Ungated")
        & (d.scenario == "Clean")
        & (d.seed == 11)
    ]
    err = abs(
        clean[["forecast_load_factor", "forecast_generation_factor"]].to_numpy()
        - clean[["plant_load_factor", "plant_generation_factor"]].to_numpy()
    )
    meta = json.loads((P / "metadata.json").read_text())
    forecast = {
        "selected_block_load_mae": float(err[:, 0].mean()),
        "selected_block_generation_mae": float(err[:, 1].mean()),
        "selected_block_joint_coverage": float(
            np.mean(
                np.all(err <= meta["forecast"]["interval_halfwidth_factor"], axis=1)
            )
        ),
    }
    (out / "forecast_validation.json").write_text(json.dumps(forecast, indent=2))
    print("AUDIT", json.dumps(audit))
    print("OVERALL", overall.to_string(index=False))
    print("PAIRED", pd.DataFrame(ci).to_string(index=False))
    print("FORECAST", json.dumps(forecast))
    print(
        "SENSITIVITY",
        pd.DataFrame(sensitivity).query("threshold==.85").to_string(index=False),
    )
    print("MATCHED", pd.DataFrame(matched).to_string(index=False))


if __name__ == "__main__":
    main()
