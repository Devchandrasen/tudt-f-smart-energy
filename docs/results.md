# Interpret the executed results

The v3.0.0 study has 1,959,552 controller records, but its physical time samples are
24 non-overlapping blocks. Seeds and controllers share operating periods. Rates
include all commands, including holds. The IEEE and two SimBench representations
use a common imposed voltage band of 0.95–1.05 p.u., which differs from native
SimBench bus limits. The original asset profiles are also forecast spatially.

## Main controller comparison

| Representation | Ungated failures | Agreement, three regimes | Composite, three regimes | Always half |
| :--- | ---: | ---: | ---: | ---: |
| IEEE 33 | 206 (0.442%) | 76 (0.163%) | 85 (0.182%) | 0 (0.000% observed) |
| SimBench aggregate | 136 (0.291%) | 73 (0.156%) | 63 (0.135%) | 54 (0.116%) |
| SimBench asset profiles | 10,407 (22.306%) | 10,146 (21.746%) | 10,136 (21.725%) | 9,779 (20.960%) |

Every cell has a denominator of 46,656 hours. Composite gating improves on ungated
dispatch but has higher mean cost: +EUR 26.22, +EUR 138.83 and +EUR 25.08 per block.
The paired asset-profile cost interval spans zero. Always-half has lower observed
violations and lower cost than composite gating in each representation. The score's
value consequently cannot be inferred from the ungated comparison alone.

Agreement-versus-composite comparisons use the same binary or three-regime policies.
Their three-regime physical differences are small and do not establish consistent
superiority. Removing any one evidence channel also produces no uniform ordering
across networks, costs and failures. All fourteen configurations are retained in
[overall.csv](../results/epsr-summary/overall.csv).

## Conditional selection information

Each representation has 40,608 degraded-condition ungated proposals with a current
reference. At matched 50% coverage, agreement/composite risk is 0.364%/0.443% for
IEEE, 0.207%/0.167% for aggregate MV and 21.218%/17.711% for asset profiles. The
asset composite advantage persists at 75% coverage but disappears near 90%.
This retrospective ranking fixes proposal and energy state; it does not replace
the separately tested development cutoffs. See
[matched_coverage.csv](../results/epsr-summary/matched_coverage.csv).

## Voltage exposure and energy capability

All observed failures are voltage events: under-voltage on IEEE and over-voltage
on SimBench. No thermal limit is crossed. The maximum within-block failure runs
are four hours on IEEE, two on aggregate MV and 55 with original asset profiles.
The asset comparison changes the full operating/forecast/measurement representation;
it is not a single-factor causal experiment isolating spatial diversity alone.

The thirteen-action realised-state diagnostic finds at least one feasible tested
power in every one of 6,912 unique states before energy screening. For composite
gating with asset profiles, 6,610 failures are energy-limited and 3,526 have an
energy-admissible same-hour alternative. Energy-limited failures are 65.21% of its
10,136 failures. Choosing a different action changes future state of charge, so
these counts are not the number preventable by a deployable oracle trajectory.
They identify energy-headroom management as a central extension to the one-hour
dispatch kernel. See [feasibility_overall.csv](../results/epsr-summary/feasibility_overall.csv).

## Uncertainty and numerical consistency

Ten thousand stratified paired bootstrap draws retain each complete physical block,
its conditions and seeds. Their intervals describe the selected benchmark periods.
Stored-energy valuation at 90/130/170 EUR/MWh changes accounting on fixed trajectories;
it does not rerun dispatch. Fresh uncached checks cover 114 realised and 114 forecast
command points, with zero label disagreements and maximum physical-metric difference
6.25e-8. Full-record arithmetic audits find no battery bound, regime or score errors
outside rounding tolerance. The independent profile conversion test covers 8,784
hours against the official SimBench helper. The
[numerical supplement](../manuscript/EPSR_Numerical_Supplement.pdf) gives the complete tables.
