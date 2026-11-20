# TUDT-F: reconstructed ISJ experiments

This release contains the causal reconstruction and extended experiments for TUDT-F. It is separate from the EPSR study on the repository's main branch. The exact originally submitted implementation was unavailable; this release does not claim exact reproduction of that implementation.

The [18 November numerical archive](https://github.com/Devchandrasen/tudt-f-smart-energy/releases/tag/isj-revision-2026-11-18) contains the source at that tag, causal energy extracts, selected waveform arrays, frozen parameters, protocols and command traces. Run the original evaluation from its extracted root. The later offline analyses below run from a checkout of this branch.

## Offline replay and diagnostics

Run these commands from the branch root. The first command needs only the checked-in window counts, case membership, NumPy and pandas. For the next two commands, replace EXTRACTED_STUDY with the directory containing the numerical archive's protocol.json.

~~~powershell
python evidence/offline_replay/summarize.py rerun/replay
python evidence/offline_replay/extract_counts.py EXTRACTED_STUDY rerun/inputs/window_counts.csv
python evidence/forecast_diagnostics/forecast_errors.py EXTRACTED_STUDY rerun/forecast
~~~

To aggregate regenerated counts, add --counts rerun/inputs/window_counts.csv to the first command. Each row of case_windows.csv identifies an existing recording and observation horizon. Weekly cases use whole windows. Midpoint cases use the first 84 or all 168 hours. The first-alarm summary includes the hour of the first jump or roughness alarm, including a false alarm. Each deterministic recording contributes once per window and condition. The paired counts measure historical dispatch exposure on fixed inputs.

The separate timing check takes the original Energy_Community.zip from [COCOON's Zenodo record](https://zenodo.org/records/21476677). It aligns recorded flags, factors and measurements in six profiles. Its output describes timestamp order and channel co-occurrence. A causal response to an alarm would require additional evidence.

~~~powershell
python evidence/measurement_timing/check_timing.py Energy_Community.zip rerun/timing.json
python -m pytest tests/test_offline_replay.py -q
~~~

## Environment and verification

Use Python 3.12 and the versions in `requirements-revision.txt`. The recorded environment is in `results/environment.json`.

```powershell
python -m pip install -r requirements-revision.txt
python verify_release.py
python -m pytest tests -q
```

## Interval calibration and stale-state detection

The [18 November numerical release](https://github.com/Devchandrasen/tudt-f-smart-energy/releases/tag/isj-revision-2026-11-18) includes the original numerical archive and three follow-up analyses. These compare weighted intervals with the frozen split gate, measure synchronization-score alarms against timestamped replay, and count joint detector misses in the existing corruption traces. A further sensitivity analysis changes how calibration timestamps are counted during adaptive initialization.

```powershell
python evidence/experiments.py . --output rerun/followup --workers 4
python evidence/adaptive_pool.py . --output rerun/adaptive --workers 4
```

The reference outputs are in `evidence/results/` and `evidence/adaptive_results/`. The latter contains scenario means and window-level runs for both adaptive initializations, including the number of overlapping timestamps. The delayed-input comparison uses nine 168-hour windows with the original frozen controller.

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
