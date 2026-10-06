# Methodology

The experiment separates the information available to the controller from the
realised next-hour network state. This timing connects forecast construction,
measurement-quality evidence, command selection and physical assessment.

## Timing and forecasting

At epoch $t$, the primary stream supplies load and generation factors $z_t$.
Features contain the current factors, lags of one and 24 hours, and sine/cosine
calendar terms. A 60-tree random forest predicts the factors at $t+1$.

Training uses January to August 2016. Calibration uses September and October.
Selected 72-hour blocks start on 1, 11 and 21 November and December. Both an
observation and its next-hour target remain inside their assigned partition.
There is no random train/test shuffle. Joint interval widths use calibration
residuals normalised by scales fitted on training residuals. Serial dependence
limits any distribution-free coverage interpretation.

The source profiles are weighted by installed active power and averaged from
quarter-hourly to hourly resolution. Both networks use the resulting common
factors. See [data provenance](data-provenance.md).

## Evidence and regimes

Four channels enter the score: freshness $F_t$, six-hour availability $C_t$,
completed-residual forecast performance $M_t$ and current reference agreement $A_t$.

$$T_t=0.30F_t+0.25C_t+0.25M_t+0.20A_t.$$

$$F_t=\exp(-a_t/2),\qquad
A_t=\exp\left[-\frac{\operatorname{mean}|z_t-r_t|}{0.10}\right].$$

The maturity channel combines joint interval coverage and mean absolute error
over up to 24 completed reference residuals. A forecast for $t$ can enter this
history only once the reference observation at $t$ has arrived. Fault labels and
future plant factors are excluded from the score.

| Score | Regime | Requested command |
| :--- | :--- | :--- |
| $T_t \geq 0.85$ | Full | Candidate command |
| $0.65 \leq T_t < 0.85$ | Cautious | Half the candidate command |
| $T_t < 0.65$ | Hold | Zero battery power |

A metadata gate and an agreement-only gate provide simpler comparisons. Each
controller has its own battery state trajectory but shares model, input factors,
candidate grid, tariffs and physical outcome definition.

## Dispatch and AC checks

Positive power discharges the battery. Candidate ratios are
$-1,-0.75,\ldots,0.75,1$ times the network-specific rated power.
Charge and discharge efficiency are both 0.94, and the one-hour SOC transition is

$$x_{t+1}=x_t-\frac{u_t^+}{0.94E}+\frac{0.94u_t^-}{E},$$

with $u_t^+=\max(u_t,0)$, $u_t^-=\max(-u_t,0)$ and capacity $E$ in MWh.
Each block starts at SOC 0.55 and requires SOC between 0.15 and 0.90.

Linear AC sensitivities rank candidates using a study cost and constraint penalty.
Acceptance uses a full AC power-flow check. The gated command is checked again
because scaling may remove useful voltage support. An infeasible nonzero command
becomes a hold, with the hold's own feasibility flag retained.

Physical outcomes use the realised factors at $t+1$. Voltages must lie between
0.95 and 1.05 pu, and line/transformer loading must not exceed 100%, with the same
numerical tolerance for every controller. A hold can violate these limits.
Forecast-model feasibility therefore does not establish plant feasibility.

## Conditions and assessment

Missing measurements use causal forward carry with 12% random absence. Delayed
measurements have two-hour age. Noise has factor standard deviation 0.08.
Bias changes primary load/generation by factors 1.5/0.5 during two six-hour bursts.
The operating disturbance raises actual load by 28% and lowers generation by 45%
for 24 hours, visible to both meters. The reference stream is assumed available
and independent, with factor noise standard deviation 0.01.

Cost includes import/export energy, throughput and a terminal stored-energy
correction. Conditional one-step regret compares feasible issued actions with
feasible discrete alternatives at the same SOC. It excludes infeasible actions.
The cost is a study objective, not an observed market bill.

Selective-risk replay holds SOC at 0.55 and uses one common proposal for both scores.
Whole-block bootstrap resampling uses 10,000 replicates and six temporal clusters,
retaining conditions and seeds within each cluster. [Results](results.md) explain
why full-policy gains do not establish superior selective ranking.
