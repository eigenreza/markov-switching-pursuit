# Reproduction

Everything runs on a laptop CPU from one Python package. The commands below are the ones the
results were produced with, in order; each writes files that the next reads. Commands marked
"raw data" need the downloads under `data/raw/`; the others run from the committed derived
tables and result files.

## Environment

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# or: .venv/bin/python -m pip install -r requirements.txt
python tasks.py setup
```

Python 3.11 with the pinned versions in `requirements.txt`. On a console that is not UTF-8
(Windows cp1252) run Python with `-X utf8`.

## The pipeline

| command | needs | writes |
|---|---|---|
| `python tasks.py data` | internet | `data/raw/<dataset>/` with verified hashes |
| `python tasks.py audit` | raw data | `results/audit/<dataset>.json` |
| `python tasks.py simulate` | nothing | the synthetic suite under `data/derived/synthetic/` |
| `python tasks.py recover` | the suite | `results/recovery/` (classification, detection, switch timing, parameter recovery, calibration, the realistic block, `limits.md`) |
| `python tasks.py fit` | raw data | `results/fits/<dataset>/` (delay profile, fits, bootstrap intervals, held-out comparison, the parametric null with 200 replicates, posteriors, event-aligned curves, parameters by bin) and the per-frame posteriors under `data/derived/` |
| `python tasks.py evaluate` | the fits | `results/evaluate/<dataset>/` (distance from optimal play, capture times, the two-player test, the evader library) |
| `python tasks.py followup` | raw data | `results/fits_persistence/<dataset>/fit_report.json`, the post-freeze library with a persistence state |
| `python tasks.py sensitivity` | raw data | the fit report of the arena-fit scale run of the primary dataset (`results/fits/michaiel2020_arena_scale/`) and the leave-one-animal-out table of the replication set |
| `python tasks.py tables` | result files | `results/tables/` |
| `python tasks.py figures` | result files | `results/figures/` (PDF and PNG) |
| `python tasks.py animate` | result files | `results/animations/` (GIF) |
| `python tasks.py paper` | the paper's LaTeX source (not published), result files, a TeX distribution | `paper/main.pdf` |
| `python tasks.py demo` | result files | the Streamlit app on a local port |
| `python tasks.py test`, `lint`, `check` | nothing | the test suite, the linter, and the consistency and style checks |

`fit` accepts `--null-b N` to change the replicate count of the parametric null and `--jobs N`
to run the replicates in parallel processes (about 700 MB per process on the mouse-and-mouse
data). The mouse-and-mouse null is the slow step: about 2.4 minutes per replicate on one
laptop core.

## Seeds and determinism

Every random draw takes its seed from `configs/model.yaml` or `configs/synthetic.yaml`; reruns
on the same platform reproduce the result files to the printed precision. Bootstrap and
permutation counts are in the same files.

## What the checks guarantee

`python tasks.py check` fails on any em dash, any date pattern and any forbidden attribution
string in a public file. `python tasks.py test` includes a test that holds the README's numbers
to the result files and, where the paper's LaTeX source is present, tests that regenerate the
paper's numbers and tables from the result files and fail if they differ, so no number in the
paper can drift from the results. The source is not published; the built paper is
`paper/main.pdf`.
