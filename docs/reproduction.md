# Reproducing the ISJ experiments

[Project overview](../README.md) · [Scientific release and checksums](https://github.com/Devchandrasen/tudt-f-smart-energy/releases/tag/isj-revision-2026-11-18)

| What you want to run | Working directory | Inputs |
| --- | --- | --- |
| Rebuild the replay summaries or the overview chart | ISJ branch checkout | Checked-in CSV files |
| Extract replay counts or forecast diagnostics from traces | ISJ branch checkout | Path to the extracted scientific release |
| Check external event timing | ISJ branch checkout | Original COCOON `Energy_Community.zip` |
| Verify or recompute the full dispatch, feeder and PV evaluations | Extracted scientific release | Data, frozen parameters and traces included in the ZIP |

## From the branch checkout

Use Python 3.12. An isolated environment keeps the study dependencies together. To install the full recorded versions, run the following in the checkout.

```sh
python -m pip install -r requirements-revision.txt
python evidence/offline_replay/summarize.py rerun/replay
python -m pytest tests/test_offline_replay.py -q
python docs/plot_dispatch_conditions.py
```

The replay test checks two CSV summaries and two JSON summaries against the reference files. The chart script reads the split-interval rows from the checked-in scenario summary and writes the overview PNG under `docs/assets/`. [Reference replay outputs](../evidence/offline_replay/reference) · [Chart source data](../evidence/results/interval_summary.csv)

To start from the archived command traces, replace `EXTRACTED_STUDY` with the path to the directory containing the release's `protocol.json`.

```sh
python evidence/offline_replay/extract_counts.py EXTRACTED_STUDY rerun/inputs/window_counts.csv
python evidence/offline_replay/summarize.py rerun/replay --counts rerun/inputs/window_counts.csv
python evidence/forecast_diagnostics/forecast_errors.py EXTRACTED_STUDY rerun/forecast
```

Each replay case identifies a recording and an observation horizon. Weekly cases use whole windows. Midpoint cases use 84 or 168 hours. A first-alarm prefix ends at the first jump or roughness alarm, which can be a false alarm. Each deterministic recording contributes once per window and condition. [Case membership](../evidence/offline_replay/case_windows.csv)

For the external timing check, obtain the original ZIP from COCOON and supply its path. The script aligns flags, factors and measurements across six recorded profiles. Temporal order alone cannot establish that detector feedback caused a waveform change. [COCOON data](https://zenodo.org/records/21476677) · [Reference timing output](../evidence/measurement_timing/temporal_check.json)

```sh
python evidence/measurement_timing/check_timing.py Energy_Community.zip rerun/timing.json
```

## From the numerical archive

Download `TUDTF_ISJ_scientific_release.zip` and its accompanying `ISJ_RELEASE_SHA256.txt` from the release assets. The ZIP contains causal energy extracts, selected waveform arrays, frozen parameters, command traces and the environment record. Extract it and open a shell in the folder containing `FILES_SHA256.json`. Verify the files before running experiments. [18 November 2026 numerical release](https://github.com/Devchandrasen/tudt-f-smart-energy/releases/tag/isj-revision-2026-11-18)

```sh
python -m pip install -r requirements-revision.txt
python verify_release.py
python -m pytest tests -q
```

Use a separate writable extraction to recompute results. Development and sensitivity overwrite their outputs. Primary and physical jobs resume completed work unless invoked with `--rerun`. The model caches rebuild from the causal extracts.

```sh
python audit_experiments.py develop
python audit_experiments.py primary --workers 4 --rerun
python audit_experiments.py physical --workers 3 --rerun
python pv_followup.py --rerun
python audit_experiments.py sensitivity
python audit_analysis.py
```

The interval and synchronization follow-ups use the same release inputs. Their checked-in references include window-level results, scenario means and the calibration-timestamp overlap counts for both adaptive initializations. [Interval and alarm results](../evidence/results) · [Adaptive results](../evidence/adaptive_results)

```sh
python evidence/experiments.py . --output rerun/followup --workers 4
python evidence/adaptive_pool.py . --output rerun/adaptive --workers 4
```

## Study records

- The archive covers 18,000 primary controller-window runs and 11,016 physical runs. The nine months are the temporal units for inference. The extended policy comparisons and fallback studies were exploratory follow-ups on previously inspected periods. [Frozen design](../protocol.json) · [Policy comparison amendment](../audit_amendment.json) · [PV fallback amendment](../pv_followup_amendment.json)
- The numerical release records the local scientific revision `f1464d8a2724a2b799a90a66bb36866da54077bc` and the historical public implementation. The release tag and checksum identify the distributed artifacts. [Historical public commit](https://github.com/Devchandrasen/tudt-f-smart-energy/commit/94a9e8360a9c3c86eb99a2a5494e7fc0ba7f1a8e) · [Release manifest](../RELEASE_MANIFEST.json)
- The archive's `results/environment.json` records the software environment. Its `data/rte/selection.json` identifies the selected recordings. The OPSD fetcher verifies the source release hash. GESL appears in the source ledger as a collection requiring authentication. Its measurements were unavailable for this evaluation. [Input fetcher](../fetch_inputs.py) · [Source ledger](../provenance/source_ledger.json) · [Data and licensing](../README.md#data-and-provenance)
