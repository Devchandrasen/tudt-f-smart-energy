"""Retrospective feasibility on all possible issued-command levels.

This diagnostic uses realised states only after dispatch. It is not a controller
and its results never enter score fitting, forecasts, proposals or issued actions.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
from .study import profiles, STEPS
from .extended import DiagnosticGrid, AssetGrid, asset_profiles, evaluate


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', type=Path, default=Path('results/epsr-2026'))
    args = ap.parse_args(argv)
    p = args.results.resolve()
    meta = json.loads((p / 'metadata.json').read_text())
    _, _, net = profiles()
    _, _, lp, lq, gp = asset_profiles(net)
    fractions = np.unique(np.r_[STEPS, .5*STEPS])
    assert len(fractions) == 13
    records, details = [], []
    cache = {}
    grids = {}
    for network in ['IEEE 33', 'SimBench MV', 'SimBench assets']:
        info = next(x for x in meta['studies'] if x['network'] == network)
        grids[network] = AssetGrid(net, (lp, lq, gp), info['centre']) if network == 'SimBench assets' else DiagnosticGrid(network, net)
    begin, count, missed = time.perf_counter(), 0, 0
    result_file = p / 'feasibility_by_block.csv'
    if result_file.exists():
        raise FileExistsError(f'Refusing to overwrite {result_file}')
    cols = ['network', 'origin', 'block', 'scenario', 'seed', 'controller', 'hour',
            'soc', 'plant_feasible', 'command_mw', 'model_feasible', 'plant_json']
    totals = {}
    for chunk in pd.read_csv(p / 'decisions.csv', usecols=cols, chunksize=20000):
        counts = {}
        for row in chunk.itertuples():
            grid = grids[row.network]
            key = (row.network, row.plant_json)
            powers = fractions * grid.pmax
            if key not in cache:
                state = np.array(json.loads(row.plant_json))
                flags = np.array([evaluate(grid, state, power)[0] for power in powers], bool)
                cache[key] = flags
                details.append(dict(network=row.network, plant_json=row.plant_json,
                    feasible_mask=''.join('1' if v else '0' for v in flags),
                    feasible_levels=int(flags.sum()), any_feasible=int(flags.any())))
            flags = cache[key]
            next_x = row.soc + np.where(powers >= 0, -powers/.94, -.94*powers) / grid.capacity
            eligible = (next_x >= .15-1e-10) & (next_x <= .9+1e-10)
            any_full = bool(flags.any())
            any_energy = bool((flags & eligible).any())
            failure = 1 - row.plant_feasible
            floor = int(not any_full)
            energy = int(any_full and not any_energy)
            avoidable = int(failure and any_energy)
            missed += int(row.plant_feasible and not any_energy)
            record_key = (row.network, row.origin, row.block, row.scenario, row.seed, row.controller)
            v = counts.setdefault(record_key, np.zeros(7, dtype=np.int64))
            v += [1, failure, floor, energy, avoidable, int(row.model_feasible == 0),
                  int(row.model_feasible == 0 and row.plant_feasible == 0)]
            count += 1
        for key, value in counts.items():
            totals.setdefault(key, np.zeros(7, dtype=np.int64))[:] += value
        if count % 200000 == 0:
            print(f'{count} records; {len(cache)} unique realised states; {time.perf_counter()-begin:.1f}s', flush=True)
    columns = ['network', 'origin', 'block', 'scenario', 'seed', 'controller', 'hours',
               'violations', 'no_feasible_power_level', 'energy_limited', 'avoidable_violations',
               'forecast_hold_infeasible', 'forecast_hold_and_plant_infeasible']
    b = pd.DataFrame([(*key, *value) for key, value in totals.items()], columns=columns)
    assert count == meta['decision_count']
    assert missed == 0
    assert (b.violations == b.no_feasible_power_level + b.energy_limited + b.avoidable_violations).all()
    b.to_csv(result_file, index=False)
    b.groupby(['network', 'controller']).sum(numeric_only=True).drop(columns='seed').reset_index().to_csv(p / 'feasibility_overall.csv', index=False)
    b.groupby(['network', 'origin', 'scenario', 'controller']).sum(numeric_only=True).drop(columns='seed').reset_index().to_csv(p / 'feasibility_scenarios.csv', index=False)
    pd.DataFrame(details).to_csv(p / 'realised_action_feasibility.csv', index=False)
    audit = dict(records=count, unique_realised_states=len(cache), action_fractions=fractions.tolist(),
                 feasible_issued_commands_missed=missed, partition_disagreements=0,
                 diagnostic_only=True, runtime_seconds=time.perf_counter()-begin,
                 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (p / 'feasibility_validation.json').write_text(json.dumps(audit, indent=2))
    manifest = {}
    for file in sorted(p.iterdir()):
        if file.is_file() and file.name != 'SHA256.json':
            digest = hashlib.sha256()
            with file.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024*1024), b''): digest.update(chunk)
            manifest[file.name] = digest.hexdigest()
    (p / 'SHA256.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(audit, indent=2), flush=True)
    print(pd.read_csv(p / 'feasibility_overall.csv').to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
