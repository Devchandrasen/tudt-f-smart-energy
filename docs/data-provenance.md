# Data provenance and reuse

The study uses public benchmark networks and synthetic benchmark profiles.
No participant data, customer meter records or confidential utility data are included.
Source attribution connects the saved hourly inputs to a reproducible transformation.

## Sources

| Source | Use | Attribution |
| :--- | :--- | :--- |
| SimBench 1.6.3 | Rural MV network and 2016 load/renewable profiles | [Dataset](https://simbench.de/en/download/datasets/), [paper](https://doi.org/10.3390/en13123290) |
| pandapower 3.3.1 | AC solver and IEEE `case33bw` reconstruction | [Software](https://github.com/e2nIEE/pandapower), [paper](https://doi.org/10.1109/TPWRS.2018.2829021) |
| Baran and Wu | Original 33-bus feeder source | [Paper](https://doi.org/10.1109/61.25627) |

## Input transformation

`profiles()` loads SimBench code `1-MV-rural--0-sw`. Each load/renewable profile
receives its installed active-power weight. The weighted aggregate is divided
by total installed active power. Four successive quarter-hour factors are averaged
into one hourly factor. This produces 8,784 paired factors for leap year 2016.

`hourly_profiles.csv` stores the resulting time, load factor and generation factor.
The same pair scales installed network load and generation on both networks.
IEEE 33 receives three 0.6 MW generators and a study current rating of 0.4 kA.
These are study modifications. The script reconstructs the networks from installed
dependencies; raw network databases are not duplicated in this repository.

## Database terms

SimBench's version-specific [licence](https://github.com/e2nIEE/simbench/blob/v1.6.3/LICENSE)
makes its database available under the
[Open Database Licence 1.0](https://opendatacommons.org/licenses/odbl/1-0/)
and individual database contents under the
[Database Contents Licence 1.0](https://opendatacommons.org/licenses/dbcl/1-0/).
Its code uses BSD-3-Clause. The version-specific notice is retained in
[SimBench-LICENSE.txt](../licenses/SimBench-LICENSE.txt).

The benchmark-derived tabular database in `results/reference/` is distributed
under ODbL 1.0 with the SimBench attribution retained. This scope includes the
hourly-factor transformation and associated decision/summary tables. It does not
replace upstream rights or terms. Original study code remains MIT. Manuscript
and figure rights are separate as set out in [NOTICE](../NOTICE).

The code and recorded settings describe the transformation. Cite the source
dataset and retain the attribution when redistributing these factors or tables.
Dependencies retain their own licences; installing them does not transfer their
ownership to this project.

## Integrity and chronology

[SHA256.json](../results/reference/SHA256.json) covers each archived data or metadata
file using canonical LF line endings. January to August is training, September to
October is calibration, and six November/December blocks are evaluation. Seeds
11, 29 and 47 govern measurement conditions. The archive is separate from generated
run destinations so that a rerun cannot silently replace the published record.
