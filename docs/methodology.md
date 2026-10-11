# Causal dispatch method

At hour t the controller receives primary factors z(t), current reference r(t),
availability and age metadata, and its own battery state of charge. The next-hour
plant state is available only to the independent evaluator after command issue.
A forecast made at t−1 can be assessed against a received reference at t. A forecast
made at t cannot use r(t+1) when its command is selected.

## Prediction and physical proposal

The frozen random forest uses current, lag-1 and lag-24 factors and calendar sine
and cosine features. Aggregate cases forecast two factors; original asset profiles
forecast 32 factors, mapped to each load's active/reactive power and each generator.
A jointly calibrated box assesses completed aggregate load/generation residuals.
Coverage is empirical; temporal dependence and seasonal shifts remain in the data.

Nine powers from −1 to +1 times rated power, in increments of 0.25, are screened
against battery bounds. A local electrical approximation ranks them by the one-hour
energy-adjusted objective plus a voltage/thermal severity penalty. Exact AC checks
accept the first feasible candidate at the forecast state. The selected regime
then uses the full proposal, half proposal or zero. A separate AC check tests that
actual command, including its fallback flag. Realised feasibility is evaluated
at the subsequently revealed plant factors, separately from forecast feasibility.

## Evidence and policies

| Channel | Causal calculation |
| :--- | :--- |
| Freshness F | exp(−age/2), with age in hours |
| Completeness C | Fraction of received primary vectors over the latest six hours |
| Completed performance M | Recent joint coverage and residual error over at most 24 completed reference pairs |
| Agreement A | exp(−mean absolute aggregate meter discrepancy / 0.10) |

Composite evidence is T = 0.30F + 0.25C + 0.25M + 0.20A. It is operating evidence,
not a feasibility probability. Agreement and composite scores each receive binary
and three-regime action policies. Lower and upper cutoffs are development quantiles
at 0.15 and 0.50, targeting 85% authorisation and 50% full dispatch conditional on
a current reference. Test coverage can differ. Reference-dependent gates hold when
the current reference is unavailable. Channel removal renormalises retained weights
and fits its own development cutoffs. A fixed 0.65/0.85 composite control is retained.

The remaining controls are ungated, always half, metadata-only, reference-fed
prediction and finite-scenario stress screening. The reference-fed control uses
the same trained predictor with reference history and primary fallback on outage.
The stress screen checks the point forecast and four jointly signed calibrated
load/generation perturbations. It does not enumerate the 32-dimensional uncertainty
box or certify nonlinear feasibility inside a continuous set.

## Trajectories and evaluation

Each controller begins each 72-hour block at state of charge 0.55 and follows its
own issued power. Charging/discharging efficiencies are 0.94 and energy limits
are 0.15–0.90. The common voltage band is 0.95–1.05 p.u.; original SimBench bands
are different. Authorisation, full regime and non-zero activity are recorded
separately. Paired intervals resample 24 physical blocks within summer/winter
strata and retain all their scenarios and measurement seeds.

Common-proposal ranking holds ungated proposals, battery state and realised outcome
identical across scores. It tests information at matched 50%, 75% and 90% selection
coverage, without changing deployed development thresholds. The retrospective
feasibility diagnostic solves all 13 possible full/half issued power levels at
realised states, then screens them at each controller's recorded energy state.
It distinguishes missing physical alternatives from energy-limited hours and
failed selection despite an available same-hour alternative. It does not build
an implementable multi-hour oracle controller.
