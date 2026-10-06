# Contributing

Reproducibility reports, clearly scoped fixes and additional benchmark evidence
help improve the project. Please include Python/dependency versions, the exact
command, input/output directory and expected versus observed behaviour in an issue.

Create a branch for changes and run `python -m unittest discover -s tests -v`.
For controller or forecasting changes, also run the quick benchmark and verify
its records. Changes affecting published numerical results need a complete rerun
and a comparison with the archived reference study.

Run `python -m tudtf compare --results results/full` after a complete rerun. A result
change should be reported explicitly rather than increasing tolerances to hide it.

Preserve chronological train/calibration/test boundaries and the separation of
controller information from future physical outcomes. Apply the same physical
violation definition to all controllers, including holds. Report cost and coverage
beside violation rates. Do not use test-set results to silently tune fixed thresholds.

Reference data represent an executed experiment. Keep new output in a separate
directory. A new scientific result should receive its own version, metadata,
checksums and documented assumptions rather than replacing records without trace.

Retain source attribution and applicable licences. Project code contributions
use the existing MIT licence; benchmark-derived databases retain their terms.
