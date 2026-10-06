# Executed reference study

These files are the archived benchmark behind the accompanying manuscript.
They contain 62,208 controller records from two networks, six 72-hour blocks,
six conditions and three seeds. Each record is a controller evaluation; records
within time blocks are correlated.

Run `python -m tudtf verify` from the project root to verify data integrity and
numerical invariants. For fresh AC checks, run
`python -m tudtf analyse --results results/reference --output results/audit`.

Generated results belong in `results/full/` or `results/quick/`. This directory
is protected against normal run or analysis output. See the
[file dictionary](../../docs/files.md) and [data provenance](../../docs/data-provenance.md).

Benchmark-derived tabular data retain SimBench attribution and ODbL 1.0 terms.
