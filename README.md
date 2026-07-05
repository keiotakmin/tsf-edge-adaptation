# When Does Online Adaptation Pay on the Edge?

Reproducibility package for *"When Does Online Adaptation Pay on the Edge? A Leakage-Free,
Warmup- and Tuning-Fair Study of the Accuracy–Memory–Compute Frontier for Building-Energy
Forecasting"* (under review, IEEE BigData 2026).

This repository contains (a) the complete experiment harness, (b) the **result artifacts the
paper is built from** (including the full 360-cell optimizer grid `grid.jsonl` and the
LR-fairness grid `lr_fairness.jsonl`), and (c) the single-source pipeline that turns those
artifacts into every number and figure in the paper.

## Layout

```
experiments/tsf_edge/    the harness (see "Reproduction map" below)
    data/                datasets: bdg2.csv shipped; ETT + Appliances via get_data.sh
results/tsf_edge/        the paper's result artifacts (data files + generated macros/figures)
```

## Quickstart 1 — rebuild the paper's numbers & figures WITHOUT a GPU (seconds)

Every number in the paper is a LaTeX macro generated from the shipped artifacts, and every
figure is drawn from them; no hand-typed results anywhere:

```bash
pip install -r requirements.txt
python experiments/tsf_edge/gen_macros.py    # -> results/tsf_edge/macros.tex (380+ macros)
python experiments/tsf_edge/paper_figs.py    # -> results/tsf_edge/*_paper.pdf (all 5 figures)
```

## Quickstart 2 — rerun the experiments (GPU)

```bash
pip install -r requirements.txt
bash experiments/tsf_edge/data/get_data.sh   # fetch ETT x4 + UCI Appliances (bdg2.csv shipped)
python experiments/tsf_edge/combined_grid.py # e.g. the 360-cell grid
```

## Reproduction map (paper item -> script -> artifact)

| Paper item | Script | Artifact | Runtime (1x A100) |
|---|---|---|---|
| Table I + Fig. 1 (C1a warmup confound) | `warmup_confound.py` | `warmup_confound.json` | ~1.5 h |
| Fig. 2 (C1c deployable protocol) | `validation_protocol.py` | `validation_protocol.json` | ~1 h |
| C1b leak inflation numbers | `leakage_check.py` | `leakage_check.json` | ~10 min |
| Fig. 3 (C2 frontier) | `frontier.py --recompute` | `frontier_data.json` | ~30 min |
| Fig. 4 (staleness, SGD + Adam rows) | `staleness.py` / `staleness.py --strategy full_adam` | `staleness_patchtst.json` / `staleness_patchtst_full_adam.json` | ~15 min each |
| C3 default-rate statistics (the confound at scale) | `combined_grid.py` | `grid.jsonl` (360 cells) | ~13 h |
| Fig. 5 + Table `lrfair` (C3 LR-fairness: three readings, plateaus) | `lr_fairness.py` (`--L/--H/--seeds`) | `lr_fairness.jsonl` (full 360-cell design) | ~10 h total |
| C1 strategy-generality paragraph | `m6_strategies.py` | `m6_strategies.json` | ~2 h |
| Discussion: BDG2 meter-selection study | `prep_bdg2_subsets.py`, then `lr_fairness.py --datasets bdg2_fox,bdg2_panther,bdg2_rat_worst` | `bdg2_*.csv` + rows in `lr_fairness.jsonl` | ~40 min |
| Every number in the paper | `gen_macros.py` | `macros.tex` | seconds, no GPU |
| Every figure in the paper | `paper_figs.py` | `*_paper.pdf` | seconds, no GPU |

`grid.jsonl`: one JSON line per cell (6 datasets x 2 backbones x H in {24,48,96} x
L in {96,192} x 5 seeds), with the fair-warmup selection, static/adapted results for
full-SGD and full-Adam, optimizer-state bytes, and the three optimizer-independent probes
(P1 noise, P2 gradient cosine, P3 drift; P3 is post hoc — it uses the test region).
`lr_fairness.jsonl`: one JSON line per cell with the full 8-point online-LR sweep
(validation-rehearsal MSE + test MSE + benefit per rate, both optimizers) and the
val-selected / test-oracle readings per optimizer.

## Protocol notes (what makes the evaluation fair)

- **Leakage-free streaming**: non-overlapping windows at stride = horizon; every target is
  scored before it can enter any gradient (`online_eval.py:stream_eval`). `leakage_check.py`
  reproduces the inflation caused by the leaky stride-1 alternative.
- **Warmup-fair baselines**: the warmup budget is picked by early-stopping on a held-out
  pre-drift validation slice (`online_eval.py:warm_and_select`), never on test data. All
  downstream measurements share this one selection procedure.
- **Tuning-fair optimizers**: each strategy's online learning rate is picked by *rehearsing*
  online adaptation on the same pre-drift validation slice (`online_eval.py:select_online_lr`),
  never on test data. `lr_fairness.py` shows that skipping this — running both optimizers at a
  shared default rate — reverses the SGD-vs-Adam verdict (the paper's third confound).

## Environment

Python 3.11, and the versions pinned in `requirements.txt` (the ones used for the paper:
torch 2.7.0+cu126 on a single NVIDIA A100 80GB, CUDA 12.5). Nearby versions are expected to
work; exact numerical reproduction assumes the pinned versions and the shipped data files
(`experiments/tsf_edge/data/checksums.sha256`).

## Data

See `experiments/tsf_edge/data/README.md` for sources, licenses, and the BDG2 preprocessing
specification. `bdg2.csv` (a processed subset of the MIT-licensed Building Data Genome 2
corpus) is shipped; ETT and UCI Appliances are downloaded by `get_data.sh` and verified
against the checksums of the exact files used in the paper.

## License / citation

Code: MIT (see `LICENSE`). Datasets keep their original licenses (see the data README).

Citation: to be added upon publication.
<!-- TODO before publishing: author names in LICENSE, citation entry, paper/arXiv link. -->
