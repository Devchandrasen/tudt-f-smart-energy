# Data provenance and reuse

The study uses public benchmark networks and synthetic benchmark profiles.
No participant data, customer meter records or confidential utility data are included.
Source attribution connects the saved hourly inputs to a reproducible transformation.

## Sources

| Source | Use | Attribution |
| :--- | :--- | :--- |
| SimBench 1.6.3 | Rural MV network and 2016 load/renewable profiles | [Dataset](https://simbench.de/en/download/datasets/), [paper](https://doi.org/10.3390/en13123290) |
| pandapower 3.4.0 | AC solver and IEEE `case33bw` reconstruction | [Software](https://github.com/e2nIEE/pandapower), [paper](https://doi.org/10.1109/TPWRS.2018.2829021) |
| Baran and Wu | Original 33-bus feeder source | [Paper](https://doi.org/10.1109/61.25627) |

## Input transformation

`profiles()` loads SimBench code `1-MV-rural--0-sw`. Aggregate factors use installed
active-power weights and average each four quarter-hour samples into one hour.
Original asset operation retains ten active-load, ten reactive-load and twelve
generation profile groups. Each element's original group multiplies its installed
P or Q separately. Official SimBench absolute values agree over all 8,784 hours
to 1e-12. Both hourly input tables are retained in `results/epsr-summary/`.

IEEE 33 receives three 0.6 MW generators and a study current rating of 0.4 kA.
These are study modifications. Storage placement, tariffs and the uniform
0.95–1.05 p.u. voltage band are also study assumptions. Original SimBench bus
bands include 0.90–1.10 and 0.965–1.055 p.u. Networks are reconstructed from pinned
dependencies; raw network databases are not duplicated in this repository.

## Database terms

SimBench's version-specific [licence](https://github.com/e2nIEE/simbench/blob/v1.6.3/LICENSE)
makes its database available under the
[Open Database Licence 1.0](https://opendatacommons.org/licenses/odbl/1-0/)
and individual database contents under the
[Database Contents Licence 1.0](https://opendatacommons.org/licenses/dbcl/1-0/).
Its code uses BSD-3-Clause. The version-specific notice is retained in
[SimBench-LICENSE.txt](../licenses/SimBench-LICENSE.txt).

The benchmark-derived tabular database in `results/reference/`, `results/epsr-summary/` and the full numerical release, `results/epsr-summary/` and the full numerical release, `results/epsr-summary/` and the full numerical release, `results/epsr-summary/` and the full numerical release is distributed
under ODbL 1.0 with the SimBench attribution retained. This scope includes the
hourly-factor transformation and associated decision/summary tables. It does not
replace upstream rights or terms. Original study code remains MIT. Manuscript
and figure rights are separate as set out in [NOTICE](../NOTICE).

The code and recorded settings describe the transformation. Cite the source
dataset and retain the attribution when redistributing these factors or tables.
Dependencies retain their own licences; installing them does not transfer their
ownership to this project.

## Integrity and chronology

Each version's SHA-256 manifest covers its archived files with canonical LF line
endings. The current summer origin trains January–April, calibrates May, develops
cutoffs in June and tests eight July blocks. Winter trains January–August,
calibrates September, develops October and tests eight blocks in each of November
and December. Every block contains 72 commands; seeds 11, 29 and 47 govern paired
measurement conditions. No future plant state enters a controller's information.
The v2.0.0 reference archive retains its original six-block design and manifests.
