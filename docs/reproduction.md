# Reproduce the EPSR study

The tested environment is Python 3.12.14, pandapower 3.4.0, SimBench 1.6.3 and
scikit-learn 1.7.2. Other numerical dependencies are pinned in `pyproject.toml`.
Run commands from the repository root after installing with `pip install -e .`.

## Verify the saved study

```bash
python -m tudtf verify-extended
python -m unittest discover -s tests -v
```

The first command verifies repository-summary digests, the executed study and
diagnostic source hashes, and saved audit counts. It does not rerun all AC solves.
The test suite includes profile conversion against official SimBench values over
all 8,784 benchmark hours. `python -m tudtf verify` separately verifies v2.0.0.

The full numerical archive is available at the
[v3.0.0 release](https://github.com/Devchandrasen/tudt-f-smart-energy/releases/tag/v3.0.0).
Download `EPSR_v3_Full_Numerical_Archive.zip`, extract its `epsr-2026/` folder into
`results/`, then run:

```bash
python -m tudtf verify-extended --results results/epsr-2026
```

The 1,959,552-row `decisions.csv` is about 1.42 GB uncompressed. SHA-256 verification
streams the file in small chunks. Summary analysis reads numeric/categorical
columns in memory; allow several gigabytes of free memory and enough disk space.

## Execute a small pipeline check

```bash
python -m tudtf extended --quick --network "SimBench assets" --output results/ci-extended
python -m tudtf analyse-extended --results results/ci-extended
```

This runs all fourteen controls for 24 hours on one July block, one seed, Clean and
Reference outage conditions. It still fits the original chronological predictor
and development cutoffs. It produces 672 records. Use `--network all` for all three
representations, producing 2,016 records. A small run is not the published experiment.

## Execute the full study

Use a new output directory so that an existing record is preserved:

```bash
python -m tudtf extended --output results/epsr-reproduction
python -m tudtf analyse-extended --results results/epsr-reproduction
python -m tudtf.feasibility_diagnostic --results results/epsr-reproduction
python -m tudtf verify-extended --results results/epsr-reproduction
```

The archived full controller run took about 38 minutes on the preparation computer;
the retrospective thirteen-action diagnostic took about 8.5 minutes. These are
observed runtimes rather than requirements for another machine. Forecasting uses
two worker threads. Cached AC states reduce repeated solves, while fresh uncached
checks audit numerical reuse. Full runs retain 24 non-overlapping 72-hour blocks,
nine conditions, three paired seeds and fourteen controllers for each representation.
The study refuses to overwrite an existing `decisions.csv`; the diagnostic refuses
to overwrite its completed block table. Remove or rename only a deliberately
discarded run, or choose another destination.

## Regenerate publication figures

```bash
python -m tudtf.extended_figures --results results/epsr-2026 --output figure-export
```

Figures 2–5 are native Matplotlib PDF plots with matching PNG previews. Figure 5
uses the prespecified noisy asset-profile block beginning on 1 November, seed 11.
Figure 1 has editable native draw.io objects. Its first page retains the supplied
image reconstruction; its second page is the EPSR publication diagram. Export that
second page as a cropped PDF. The graphical abstract has standalone TikZ source.

Compile `manuscript/EPSR_Manuscript.tex` from the manuscript folder with the
Elsevier `elsarticle` class and its supplied `epsr_figures/` directory. Compile
`EPSR_Numerical_Supplement.tex` separately. Two LaTeX passes resolve numbered
references. The saved paper uses the same values as the repository summaries.

## Files and deterministic settings

`metadata.json` retains partitions, profile mapping, network/storage settings,
cutoffs, calibration half-widths, software versions and executed source hash.
`validation.json` records arithmetic and fresh-solve checks. The action diagnostic
has a separate source hash in `feasibility_validation.json`. Bootstrap resampling
uses seed 20261011; common-proposal ties use seed 1729; forests use seed 42. Small
floating-point differences can arise across supported operating systems. Physical
labels use the documented tolerance of 1e-7. Compare outcomes and arithmetic within
their specified tolerance, while archive digests verify the exact saved files.
