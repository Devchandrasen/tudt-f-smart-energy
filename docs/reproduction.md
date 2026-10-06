# Reproduce the benchmark

The reference results were produced with Python 3.12.14, SciPy 1.18.0 and pandapower
3.3.1. The installable project uses SciPy 1.16.3 and pandapower 3.4.0 to satisfy the
dependencies declared by pandapower and SimBench 1.6.3. The other direct study
dependencies retain their original versions. The full-rerun comparison is recorded
in `results/reproduction_check.json`.
Install the project from its root in a Python 3.12
virtual environment. SimBench includes the benchmark data in its Python distribution;
a separate OPSD download is not required.

```bash
python -m venv .venv
# PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -e .
python -m tudtf --help
python -m tudtf verify
```

`verify` checks SHA-256 integrity, record counts, battery equations, trust scores,
regimes, physical labels, command feasibility and block summaries. It uses the
saved records; `analyse` adds fresh AC solves. Run all commands from the project root.

`requirements-reproduction.txt` also records the resolved dependency versions from
the tested installation. To request that complete version set, install it before
the editable project with `python -m pip install -r requirements-reproduction.txt`
and `python -m pip install --no-deps -e .`.

## Execution check

A short run uses both networks, the clean and bias conditions, one seed and one
24-hour block. It generates 384 controller records without replacing the archive.

```bash
python -m tudtf run --quick
python -m tudtf verify --results results/quick --skip-hashes
python -m tudtf analyse --results results/quick
```

The plot retains the six condition labels; conditions omitted by the short run
have no estimate. Short-run bootstrap intervals contain only one temporal cluster
and cannot be used for scientific inference.

## Full experiment

The full run reconstructs 8,784 hourly factor pairs, fits the predictor on earlier
partitions, evaluates every controller and saves intermediate and final records.

```bash
python -m tudtf run
python -m tudtf analyse --results results/full
python -m tudtf verify --results results/full --skip-hashes
python -m tudtf compare --results results/full
```

Expected output is 62,208 decisions and 864 network/block/condition/seed/controller
summaries. The runtime depends on the machine; the archived experiment records its
measured runtime in `metadata.json`. Random forest fitting uses two worker threads.
New runs may differ slightly with numerical-library builds or platform versions.

`--output PATH` changes the run destination. The analysis accepts `--results PATH`
and a separate `--output PATH`. To audit the published records without writing
into their directory:

```bash
python -m tudtf analyse --results results/reference --output results/audit
```

Files inside `results/reference/` are protected as analysis and run destinations.
Normal runs overwrite their generated output files, so choose a new output directory
when retaining an earlier run. Verification without `--skip-hashes` requires a manifest.

## Tests

The tests use Python's standard `unittest` runner and installed study dependencies.
They check future-data isolation, partition boundaries, meter corruption, AC-label
logic, battery equations, gate thresholds and the entire reference archive.

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs these tests on Linux and Windows and executes the short benchmark
on Linux. It does not rerun the full scientific experiment on every push.

## Manuscript source

`manuscript/SEGAN_Manuscript.tex` is a self-contained article with TikZ/PGFPlots
artwork, editable equations, tables and bibliography. An existing LaTeX distribution
with those packages can compile it with two `pdflatex` passes. Figure sources and
high-resolution PNG artwork are included in `manuscript/figures/`.
