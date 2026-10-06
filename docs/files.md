# File dictionary

All reference paths below are relative to `results/reference/`. These files connect
reported aggregate results to individual decisions and model settings.

`results/migration_check.json` records the code-migration comparison in the original
runtime. `results/reproduction_check.json` records the complete comparison in the
compatible installation environment. Both report numeric tolerances and per-column
differences without replacing the original archived results.

| File | Contents |
| :--- | :--- |
| `hourly_profiles.csv` | 8,784 time/load/generation factor pairs |
| `decisions.csv` | 62,208 decisions with observations, forecasts, plant state and outcomes |
| `block_results.csv` | 864 network/block/condition/seed/controller summaries |
| `summary.csv` | Per-condition summaries averaged across blocks and seeds |
| `overall_results.csv` | Equal-weight overall summaries by network and controller |
| `proposal_replay.csv` | Fixed-SOC proposals and evidence for selective risk |
| `risk_coverage.csv` | Score thresholds and conditional-risk curves |
| `paired_uncertainty.csv` | Paired differences and whole-block bootstrap intervals |
| `score_sensitivity.csv` | Alternative channel weights and thresholds |
| `matched_coverage.csv` | Trust and agreement risk at matched retained fractions |
| `metadata.json` | Versions, networks, parameters, partitions and execution duration |
| `validation.json` | Numerical checks and fresh AC comparison |
| `forecast_validation.json` | Selected-block forecast error and joint coverage |
| `SHA256.json` | Byte integrity of the archived files |

## Decision columns

The identity is `(network, block, scenario, seed, hour, controller)`. `hour` starts
at zero inside a block. Factors are dimensionless; powers are MW; costs are EUR;
SOC is a fraction; voltage is pu; thermal utilisation is a fraction of rated loading.

| Group | Important columns |
| :--- | :--- |
| Evidence | `freshness`, `completeness`, `maturity`, `agreement`, `trust` |
| Current readings | `observed_*_factor`, `reference_*_factor` |
| Next-hour forecast | `forecast_*_factor` |
| Realised next-hour state | `plant_*_factor` |
| Commands and battery | `proposal_mw`, `command_mw`, `regime`, `soc`, `next_soc` |
| Model acceptance | `model_feasible` |
| Physical assessment | `plant_feasible`, `vmin`, `vmax`, `max_thermal_utilisation`, `severity` |
| Energy and economics | `grid_mw`, `loss_mw`, `cost_eur`, `regret_eur` |
| Summary flags | `violation`, `nonzero`, `authorised` |

Realised plant columns are assessment outputs. They do not enter the trust score.
A nonzero command must pass model checks, but model acceptance and plant feasibility
can differ. `regret_eur` is missing when the issued action is physically infeasible
or no feasible oracle alternative exists. Authorisation includes zero proposals
allowed by a gate; it is different from nonzero dispatch.

## Manuscript assets

The manuscript directory contains a compiled PDF, one self-contained editable
`.tex` source, a graphical abstract, and three figure sources with high-resolution
PNG exports and captions. Submission correspondence and private preparation notes
are excluded from the public project.
