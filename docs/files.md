# File guide

| File or directory | Purpose |
| :--- | :--- |
| `src/tudtf/extended.py` | Executed rolling-origin, score-policy and asset-profile study |
| `src/tudtf/extended_analysis.py` | Arithmetic audit, fresh AC checks, paired uncertainty and tables |
| `src/tudtf/feasibility_diagnostic.py` | Retrospective thirteen-action feasibility and energy partition |
| `src/tudtf/extended_verify.py` | Streaming checksums and executed-source verification |
| `src/tudtf/extended_figures.py` | Publication Figures 2–5 and fixed trace data |
| `tests/test_extended.py` | Reference handling, causality, action rules and original profile mapping |
| `results/epsr-summary/` | Current summaries, inputs, metadata and audit manifests |
| `EPSR_v3_Full_Numerical_Archive.zip` | Release asset with every per-hour controller record |
| `manuscript/EPSR_Manuscript.pdf` / `.tex` | Current paper and editable mathematical source |
| `manuscript/EPSR_Numerical_Supplement.pdf` / `.tex` | Complete numerical tables |
| `manuscript/epsr_figures/Figure_1.drawio` | Original native reconstruction and EPSR publication page |
| `manuscript/epsr_figures/Figure_1.pdf` | Vector architecture used in the paper |
| `manuscript/epsr_figures/Figure_2–5.pdf` | Native result plots and matching PNG previews |
| `manuscript/EPSR_Graphical_Abstract.pdf` / `.tex` | Standalone conceptual graphic and TikZ source |
| `manuscript/EPSR_Highlights.docx` | Editable publication highlights |
| `results/reference/` | Historical v2.0.0 archive, protected from overwrite |

`SHA256.json` in the summary covers its local files. `ARCHIVE_SHA256.json` records
the complete release's numerical-file digests, including files deliberately kept
outside Git. The full archive has its own `SHA256.json`. Historical SEGAN manuscript
files retain their original design and should not be used with EPSR result tables.
Private author declarations and rejection correspondence are not included here.
