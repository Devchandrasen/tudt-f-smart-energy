"""Verify the EPSR numerical archive or its repository summary without large reads."""
from pathlib import Path
import argparse
import hashlib
import json


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results/epsr-summary'))
    args = parser.parse_args(argv)
    root = args.results.resolve()
    manifest = json.loads((root / 'SHA256.json').read_text())
    errors = []
    for name, expected in manifest.items():
        file = root / name
        if not file.is_file():
            errors.append(f'Missing archive file: {name}')
        elif digest(file) != expected:
            errors.append(f'Checksum mismatch: {name}')
    meta = json.loads((root / 'metadata.json').read_text())
    audit = json.loads((root / 'validation.json').read_text())
    diag = json.loads((root / 'feasibility_validation.json').read_text())
    if meta['decision_count'] != audit['decision_count'] or meta['decision_count'] != diag['records']:
        errors.append('Controller-record counts disagree across audit records')
    if digest(Path(__file__).with_name('extended.py')) != meta['source_sha256']:
        errors.append('Study source differs from the executed version')
    if digest(Path(__file__).with_name('feasibility_diagnostic.py')) != diag['source_sha256']:
        errors.append('Feasibility diagnostic differs from the executed version')
    for key in ['soc_bounds_errors', 'nonzero_model_infeasible', 'nonfinite_outcomes',
                'proposal_battery_bounds_errors', 'physical_label_disagreements',
                'regime_disagreements', 'reference_fallback_disagreements',
                'fresh_ac_label_disagreements', 'fresh_model_label_disagreements']:
        if audit[key] != 0:
            errors.append(f'Failed recorded audit: {key}')
    if diag['partition_disagreements'] or diag['feasible_issued_commands_missed']:
        errors.append('Failed recorded action-feasibility audit')
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'Verified {len(manifest)} file digests and executed source hashes. '
          f'Audit records cover {meta["decision_count"]:,} controller records.')
    if not (root / 'decisions.csv').exists():
        print('This is the repository summary. Complete per-hour data are in the v3.0.0 release asset.')


if __name__ == '__main__':
    main()
