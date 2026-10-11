"""Rolling-origin, score-policy factorial and asset-profile dispatch study.

The development set determines coverage cutoffs, never test outcomes. Reference
measurements are causal inputs. Independent realised outcomes never enter dispatch.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import platform
import time
import numpy as np
import pandas as pd
import pandapower as pp
import simbench as sb
import sklearn
from sklearn.ensemble import RandomForestRegressor
from .study import Grid, features, next_soc, energy_cost, price_at, profiles, STEPS
from .verify import protect_reference

SCENARIOS = ["Clean", "Missing", "Delayed", "Noisy", "Bias", "Disturbance",
             "Reference outage", "Reference noisy", "Common bias"]
SEEDS = [11, 29, 47]
WEIGHTS = np.array([.30, .25, .25, .20])
MODES = ["Ungated", "Always half", "Metadata", "Agreement binary",
         "Composite binary", "Agreement ternary", "Composite ternary",
         "Without freshness", "Without completeness", "Without maturity",
         "Without agreement", "Reference forecast", "Stress screen", "Fixed composite"]
SCORE_WEIGHTS = {"Composite": WEIGHTS}
for j, label in enumerate(["freshness", "completeness", "maturity", "agreement"]):
    w = WEIGHTS.copy()
    w[j] = 0
    SCORE_WEIGHTS["Without " + label] = w / w.sum()


def asset_profiles(net):
    """Retain the supplied profile mapping for each P load, Q load and generator."""
    names, series, lp, lq, gp = [], [], [], [], []
    for kind, table, suffix, mapping in [
        ("P", net.load, "_pload", lp), ("Q", net.load, "_qload", lq),
        ("G", net.sgen, "", gp)]:
        for profile in table.profile:
            name = kind + ":" + str(profile)
            if name not in names:
                names.append(name)
                source = net.profiles["load"] if kind != "G" else net.profiles["renewables"]
                values = source[str(profile) + suffix].to_numpy(dtype=float)
                series.append(values.reshape(-1, 4).mean(axis=1))
            mapping.append(names.index(name))
    data = np.stack(series, axis=1)
    return data, names, np.array(lp), np.array(lq), np.array(gp)


class DiagnosticGrid(Grid):
    def metrics(self, a):
        base = super().metrics(a)
        if not np.isfinite(a).all():
            return (*base, -1, -1, -1, -1)
        v, thermal = a[:self.nb], a[self.nb:-2]
        return (*base, int((v < .95 - 1e-7).sum()), int((v > 1.05 + 1e-7).sum()),
                int((thermal > 1 + 1e-7).sum()), int(self.net.bus.index[np.argmin(v)]))


class AssetGrid(DiagnosticGrid):
    def __init__(self, net, mapping, centre):
        # Initialise the benchmark topology and storage exactly as in the scalar case.
        super().__init__("SimBench MV", net)
        self.name = "SimBench assets"
        self.lp, self.lq, self.gp = mapping
        self.centre = np.asarray(centre)
        self.cache = {}
        self.set_vector(self.centre, 0)
        base = self.solve(raw=True)
        sens = []
        for j in range(len(centre) + 1):
            state = self.centre.copy()
            if j < len(centre):
                state[j] += .001
            self.set_vector(state, .001 if j == len(centre) else 0)
            sens.append((self.solve(raw=True) - base) / .001)
        self.linear_base, self.sens = base, np.array(sens)

    def set_vector(self, state, u):
        self.net.load.loc[:, "p_mw"] = self.load_p.to_numpy() * state[self.lp]
        self.net.load.loc[:, "q_mvar"] = self.load_q.to_numpy() * state[self.lq]
        self.net.sgen.loc[:, "p_mw"] = np.r_[self.gen_p.iloc[:-1].to_numpy() * state[self.gp], u]

    def evaluate_vector(self, state, u):
        key = (np.round(state, 10).tobytes(), round(float(u), 10))
        if key not in self.cache:
            self.set_vector(state, u)
            self.cache[key] = self.solve()
        return self.cache[key]

    def approximate_vector(self, state, u):
        return self.metrics(self.linear_base + np.r_[state - self.centre, u] @ self.sens)


def aggregate(data, net=None, mapping=None):
    if mapping is None:
        return data.copy()
    lp, _, gp = mapping
    return np.c_[data[:, lp] @ net.load.p_mw.to_numpy() / net.load.p_mw.sum(),
                 data[:, gp] @ net.sgen.p_mw.to_numpy() / net.sgen.p_mw.sum()]


def measurement_streams(true, scenario, seed, start, load_dims, gen_dims):
    rng = np.random.default_rng(seed + start)
    z = true.copy()
    r = np.clip(true + rng.normal(0, .01, true.shape), 0, None)
    available = np.ones(len(true), bool)
    ref_available = available.copy()
    age = np.zeros(len(true))
    if scenario == "Missing":
        available = rng.random(len(true)) >= .12
    elif scenario == "Delayed":
        z[2:] = true[:-2]
        age[:] = 2
    elif scenario == "Noisy":
        z = np.clip(true + rng.normal(0, .08, true.shape), 0, None)
    elif scenario == "Reference noisy":
        r = np.clip(true + rng.normal(0, .08, true.shape), 0, None)
    # The prehistory starts with a received observation, providing a causal anchor.
    available[0] = True
    for offset in [12, 48]:
        sl = slice(24 + offset, 24 + offset + 6)
        if scenario in ["Bias", "Common bias"]:
            z[sl, load_dims] *= 1.5
            z[sl, gen_dims] *= .5
        if scenario == "Common bias":
            r[sl, load_dims] *= 1.5
            r[sl, gen_dims] *= .5
        if scenario == "Reference outage":
            ref_available[sl] = False
    for i in range(1, len(true)):
        if not available[i]:
            z[i] = z[i - 1]
            age[i] = age[i - 1] + 1
        if not ref_available[i]:
            r[i] = r[i - 1]
    return z, r, available, ref_available, age


def evidence(z, r, available, ref_available, age, i, history, q, cal_mae):
    recent = history[-24:]
    if recent:
        err = np.array([np.abs(a - b) for a, b in recent])
        coverage, mae = np.mean(np.all(err <= q, axis=1)), err.mean()
    else:
        coverage, mae = .9, cal_mae
    maturity = .5 * min(1., coverage / .9) + .5 * np.exp(-mae / (3 * cal_mae))
    agreement = np.exp(-np.mean(np.abs(z[i] - r[i])) / .10) if ref_available[i] else 0.
    return np.array([np.exp(-age[i] / 2), available[max(0, i - 5):i + 1].mean(),
                     maturity, agreement])


def fitted_cutoffs(values):
    """Fixed coverage targets; ties are retained and actual coverage is recorded."""
    return [float(np.quantile(values, .15, method="linear")),
            float(np.quantile(values, .50, method="linear"))]


def score_value(parts, mode):
    if mode.startswith("Agreement"):
        return float(parts[3])
    key = mode if mode.startswith("Without ") else "Composite"
    return float(parts @ SCORE_WEIGHTS[key])


def regime(parts, mode, cutoffs, ref_available, present, age):
    if mode in ["Ungated", "Reference forecast", "Stress screen"]:
        return "full"
    if mode == "Always half":
        return "cautious"
    if mode == "Metadata":
        return "full" if present and age == 0 else "hold"
    if not ref_available:
        return "hold"
    value = score_value(parts, mode)
    key = "Agreement" if mode.startswith("Agreement") else (
        mode if mode.startswith("Without ") else "Composite")
    low, high = [.65, .85] if mode == "Fixed composite" else cutoffs[key]
    if mode.endswith("binary"):
        return "full" if value >= low else "hold"
    return "full" if value >= high else ("cautious" if value >= low else "hold")


def make_model(data, dates, origin, agg):
    train_end, cal_end, dev_end = map(pd.Timestamp, origin)
    indices = np.arange(24, len(data) - 1)
    train = indices[dates[indices + 1] < train_end]
    cal = indices[(dates[indices] >= train_end) & (dates[indices + 1] < cal_end)]
    dev = indices[(dates[indices] >= cal_end) & (dates[indices + 1] < dev_end)]
    model = RandomForestRegressor(n_estimators=60, min_samples_leaf=8,
                                  max_features=.75, random_state=42, n_jobs=2)
    model.fit(features(data, dates, train), data[train + 1])
    tp = model.predict(features(data, dates, train))
    cp = model.predict(features(data, dates, cal))
    errs = np.abs(agg(cp) - agg(data[cal + 1]))
    scales = np.maximum(np.quantile(np.abs(agg(tp) - agg(data[train + 1])), .9, axis=0), 1e-6)
    rank = min(len(cal), int(np.ceil((len(cal) + 1) * .9)))
    q = scales * np.sort(np.max(errs / scales, axis=1))[rank - 1]
    state_q = np.quantile(np.abs(cp - data[cal + 1]), .9, axis=0)
    return model, q, float(errs.mean()), state_q, dev, dict(
        training_hours=len(train), calibration_hours=len(cal), development_hours=len(dev),
        train_target_last=str(dates[train[-1] + 1]), cal_target_last=str(dates[cal[-1] + 1]),
        dev_target_last=str(dates[dev[-1] + 1]), joint_halfwidth=q.tolist(),
        stress_halfwidth=state_q.tolist(), calibration_mae=float(errs.mean()),
        calibration_coverage=float(np.mean(np.all(errs <= q, axis=1))))


def block_inputs(data, dates, model, q, mae, start, length, scenario, seed, agg, ld, gd):
    sl = slice(start - 24, start + length + 1)
    true = data[sl].copy()
    if scenario == "Disturbance":
        true[48:72, ld] *= 1.28
        true[48:72, gd] *= .55
    z, r, present, ref_present, age = measurement_streams(true, scenario, seed, start, ld, gd)
    localdates = dates[sl]
    indices = np.arange(24, 24 + length)
    pred = np.clip(model.predict(features(z, localdates, indices)), 0, None)
    rp = np.clip(model.predict(features(r, localdates, indices)), 0, None)
    az, ar, ap = agg(z), agg(r), agg(pred)
    history, parts = [], []
    for k, i in enumerate(indices):
        if k and ref_present[i]:
            history.append((ap[k - 1], ar[i]))
        parts.append(evidence(az, ar, present, ref_present, age, i, history, q, mae))
    return true, pred, rp, np.array(parts), present[indices], ref_present[indices], age[indices], localdates[indices + 1]


def develop_cutoffs(data, dates, model, q, mae, dev, agg, ld, gd):
    # Non-overlapping development blocks spanning the whole development month.
    starts = dev[::72]
    pools = {"Agreement": [], **{key: [] for key in SCORE_WEIGHTS}}
    for start in starts:
        length = min(72, int(dev[-1] - start + 1))
        for scenario in SCENARIOS:
            for seed in SEEDS:
                _, _, _, parts, _, ref_ok, _, _ = block_inputs(
                    data, dates, model, q, mae, start, length, scenario, seed, agg, ld, gd)
                parts = parts[ref_ok]
                pools["Agreement"].extend(parts[:, 3])
                for key, w in SCORE_WEIGHTS.items():
                    pools[key].extend(parts @ w)
    cutoffs = {key: fitted_cutoffs(value) for key, value in pools.items()}
    achieved = {key: {"authorised": float(np.mean(np.array(value) >= cutoffs[key][0])),
                      "full": float(np.mean(np.array(value) >= cutoffs[key][1])),
                      "available_reference_samples": len(value)} for key, value in pools.items()}
    return cutoffs, achieved


def evaluate(grid, state, u):
    return grid.evaluate_vector(state, u) if isinstance(grid, AssetGrid) else grid.evaluate(*state, u)


def approximation(grid, state, u):
    return grid.approximate_vector(state, u) if isinstance(grid, AssetGrid) else grid.approximate(*state, u)


def choose(grid, state, soc, price, cache, stress_q=None, ld=None, gd=None):
    allowed = [j for j, ratio in enumerate(STEPS)
               if .15 - 1e-10 <= next_soc(soc, ratio * grid.pmax, grid.capacity) <= .9 + 1e-10]
    key = (np.round(state, 10).tobytes(), tuple(allowed), price, stress_q is not None)
    if key not in cache:
        ranked = []
        for j in allowed:
            u = STEPS[j] * grid.pmax
            m = approximation(grid, state, u)
            ranked.append((energy_cost(m[2], u, price) + 1e5 * m[1], u))
        corners = [state]
        if stress_q is not None:
            corners = []
            for lsign, gsign in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                s = state.copy()
                s[ld] += lsign * stress_q[ld]
                s[gd] += gsign * stress_q[gd]
                corners.append(np.maximum(s, 0))
            corners.append(state)
        selected = 0.
        for _, u in sorted(ranked):
            if all(evaluate(grid, s, u)[0] for s in corners):
                selected = float(u)
                break
        cache[key] = selected
    return cache[key]


def duration(flags):
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def run_block(grid, inputs, cutoffs, stress_q, ld, gd, origin, start, scenario, seed):
    true, pred, refpred, parts, present, ref_present, age, dates = inputs
    soc = {mode: .55 for mode in MODES}
    rows, cache = [], {}
    for k in range(len(pred)):
        for mode in MODES:
            state = refpred[k] if mode == "Reference forecast" and ref_present[k] else pred[k]
            prior, price = soc[mode], price_at(dates[k].hour)
            u = choose(grid, state, prior, price, cache,
                       stress_q if mode == "Stress screen" else None, ld, gd)
            rg = regime(parts[k], mode, cutoffs, ref_present[k], present[k], age[k])
            command = u if rg == "full" else (.5 * u if rg == "cautious" else 0.)
            checked = evaluate(grid, state, command)
            rejected = bool(command != 0 and not checked[0])
            if rejected:
                command = 0.
                checked = evaluate(grid, state, 0.)
            actual = evaluate(grid, true[25 + k], command)
            soc[mode] = next_soc(prior, command, grid.capacity)
            rows.append(dict(network=grid.name, origin=origin, block=str(start), scenario=scenario,
                seed=seed, hour=k, controller=mode, regime=rg, proposal_mw=u,
                command_mw=command, soc=prior, next_soc=soc[mode],
                freshness=parts[k, 0], completeness=parts[k, 1], maturity=parts[k, 2],
                agreement=parts[k, 3], composite=float(parts[k] @ WEIGHTS),
                primary_available=int(present[k]), reference_available=int(ref_present[k]),
                primary_age=age[k], reference_fallback=int(mode == "Reference forecast" and not ref_present[k]),
                model_feasible=checked[0], rejected_scaled=int(rejected), plant_feasible=actual[0],
                severity=actual[1], grid_mw=actual[2], loss_mw=actual[3], vmin=actual[4],
                vmax=actual[5], max_thermal_utilisation=actual[6],
                undervoltage_buses=actual[7], overvoltage_buses=actual[8],
                overloaded_elements=actual[9], lowest_voltage_bus=actual[10],
                cost_eur=energy_cost(actual[2], command, price),
                state_json=json.dumps(state.tolist(), separators=(",", ":")),
                plant_json=json.dumps(true[25 + k].tolist(), separators=(",", ":"))))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("results/extended"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--network", choices=["all", "IEEE 33", "SimBench MV", "SimBench assets"], default="all")
    args = ap.parse_args(argv)
    out = args.output.resolve()
    protect_reference(out)
    out.mkdir(parents=True, exist_ok=True)
    dates, scalar, net = profiles()
    assets, names, lp, lq, gp = asset_profiles(net)
    mapping = (lp, lq, gp)
    origins = [("summer", ["2016-05-01", "2016-06-01", "2016-07-01"], [7]),
               ("winter", ["2016-09-01", "2016-10-01", "2016-11-01"], [11, 12])]
    begin, metadata = time.perf_counter(), []
    total = 0
    decision_file = out / "decisions.csv"
    if decision_file.exists():
        raise FileExistsError(f"Refusing to overwrite {decision_file}")
    networks = ["IEEE 33", "SimBench MV", "SimBench assets"] if args.network == "all" else [args.network]
    for name in networks:
        data = assets if name == "SimBench assets" else scalar
        agg = (lambda x: aggregate(x, net, mapping)) if name == "SimBench assets" else aggregate
        ld = np.flatnonzero([not n.startswith("G:") for n in names]) if name == "SimBench assets" else np.array([0])
        gd = np.flatnonzero([n.startswith("G:") for n in names]) if name == "SimBench assets" else np.array([1])
        for label, origin, months in (origins[:1] if args.quick else origins):
            model, q, mae, sq, dev, fm = make_model(data, dates, origin, agg)
            cutoffs, achieved = develop_cutoffs(data, dates, model, q, mae, dev, agg, ld, gd)
            centre = np.median(data[:dev[0]], axis=0)
            grid = AssetGrid(net, mapping, centre) if name == "SimBench assets" else DiagnosticGrid(name, net)
            info = dict(network=name, origin=label, boundaries=origin, forecast=fm,
                        cutoffs=cutoffs, development_coverage=achieved, centre=centre.tolist(),
                        battery_bus=grid.battery_bus, capacity_mwh=grid.capacity, pmax_mw=grid.pmax)
            metadata.append(info)
            for month in months:
                for day in ([1] if args.quick else [1, 5, 9, 13, 17, 21, 25, 28]):
                    date = pd.Timestamp(f"2016-{month:02}-{day:02}")
                    start = int(np.flatnonzero(dates == date)[0])
                    for scenario in (["Clean", "Reference outage"] if args.quick else SCENARIOS):
                        for seed in (SEEDS[:1] if args.quick else SEEDS):
                            length = 24 if args.quick else 72
                            inputs = block_inputs(data, dates, model, q, mae, start, length,
                                                  scenario, seed, agg, ld, gd)
                            rows = run_block(grid, inputs, cutoffs, sq, ld, gd, label, date, scenario, seed)
                            frame = pd.DataFrame(rows)
                            frame.to_csv(decision_file, mode="a", header=total == 0, index=False)
                            total += len(frame)
                    print(f"{name} {label} {date.date()}: {total} records; {len(grid.cache)} AC states; {time.perf_counter()-begin:.1f}s", flush=True)
            # Cache is numerical reuse only. Drop it between independent origins.
    meta = dict(created_utc=pd.Timestamp.now(tz="UTC").isoformat(),
                runtime_seconds=time.perf_counter() - begin, python=platform.python_version(),
                pandapower=pp.__version__, simbench=sb.__version__, sklearn=sklearn.__version__,
                decision_count=total, modes=MODES, scenarios=SCENARIOS, seeds=SEEDS,
                profiles=names, asset_mapping={"load_p": lp.tolist(), "load_q": lq.tolist(), "sgen_p": gp.tolist()},
                studies=metadata, quick=args.quick, development_targets={"authorised": .85, "full": .50},
                reference_outage_action="reference-dependent gates hold; reference predictor falls back to primary",
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Finished {total} controller records.", flush=True)


if __name__ == "__main__":
    main()
