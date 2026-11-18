TUDT-F interval and synchronization experiments

Download the numerical archive from the versioned release and extract it:
https://github.com/Devchandrasen/tudt-f-smart-energy/releases/tag/isj-revision-2026-11-18

Run the following commands from the extracted directory containing
data_pipeline.py. The environment uses Python 3.12 and the versions in
requirements-revision.txt. The study directory must be writable so the
chronological model cache can be rebuilt.

  python evidence/experiments.py . --workers 4 --output rerun/followup
  python evidence/adaptive_pool.py . --workers 4 --output rerun/adaptive
  python evidence/figure.py results/audit_v2/interval_authorization.csv rerun/followup/interval_summary.csv figure4.pdf

The design records the exploratory recency-weighted, stale-state and joint
detector-miss comparisons. Reference values appear in evidence/results.
The synchronization experiment compares paired current and replay packets
and counts each deterministic window once.

The adaptive initialization sensitivity retains the original calibration
scores. In the adaptive arm the quantile also uses the latest 168 completed
source-time scores. In the adaptive_distinct arm only later target timestamps
can enter the recent-score buffer. Clipping, point forecasts, received
observations, action rules and gate parameters are common to both runs.

The scenario means, individual runs, initial overlap counts and input/output
hashes for this comparison appear in evidence/adaptive_results. Runs store
deterministic scenarios once with their seed multiplicity. The nine monthly
windows receive equal weight. The regression check compares ten outcomes
in eight scenarios with the archived adaptive arm.
