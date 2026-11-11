# TUDT-F: reconstructed ISJ experiments

This release contains the causal reconstruction and extended experiments for TUDT-F. It is separate from the EPSR study on the repository's main branch. The exact originally submitted implementation was unavailable; this release does not claim exact reproduction of that implementation.

Download and extract the numerical archive attached to this release. It contains the runnable source, causal energy extracts, selected waveform arrays, frozen parameters, protocols, numerical summaries and command traces. The small source tree in this branch is also included in the archive. Run commands from the extracted archive root.

## Environment and verification

Use Python 3.12 and the versions in `requirements-revision.txt`. The recorded environment is in `results/environment.json`.

```powershell
python -m pip install -r requirements-revision.txt
python verify_release.py
python -m pytest tests -q
```

## Recompute the extended evaluation

Use a separate extracted copy to preserve the supplied reference results.

```powershell
python audit_experiments.py develop
python audit_experiments.py primary --workers 4 --rerun
python audit_experiments.py physical --workers 3 --rerun
python pv_followup.py --rerun
python audit_experiments.py sensitivity
python audit_analysis.py
```

Without `--rerun`, completed primary and physical jobs are skipped. Development and sensitivity stages recompute their outputs. Model caches regenerate from the supplied causal extracts; `fetch_inputs.py` can download the original OPSD release and verify its hash. No GESL measurements are included.

## Design and interpretation

The initial reconstructed protocol and frozen gate are preserved. Extended comparisons and fallback follow-ups are explicitly exploratory amendments after inspection of the same test periods. The protocol and amendment files record this chronology. There are 18,000 primary controller-window runs and 11,016 physical runs; these counts are not independent sample sizes. Inference retains the nine monthly windows as clusters. Paired corruption seeds are robustness checks.

The original selected gate reduces independent forecast-error exposure while increasing review and cost. Comparisons under common action policies do not establish general four-channel superiority. Generic half/zero PV gating can worsen violations. Feasible-only recovery reproduces ungated commands in all 435,456 tested hourly states and therefore supplies no added physical benefit. Weak real-disturbance detector transfer and null or unfavorable stress results remain included. No calibrated safety probability, field dispatch validation or adversarial guarantee is claimed.

## Data and licensing

Code follows the included MIT license. Data retain their source-specific terms: OPSD/ENTSO-E attribution is in `provenance/OPSD_datapackage.json`; the RTE recordings use ODbL, and the third-party annotation notice is BSD. Both notices are included. `data/rte/selection.json` identifies the selected records. Keep these attributions with redistributed data. The release manifest and SHA-256 inventory identify the exact payload.

Historical source references are retained for reconstruction provenance. The scientific code and numerical outputs were preserved from the local scientific revision f1464d8a2724a2b799a90a66bb36866da54077bc; that identifier is a provenance record, not a promise that the private development history is publicly accessible. The legacy public implementation was 94a9e8360a9c3c86eb99a2a5494e7fc0ba7f1a8e. This release has its own public commit and tag.
