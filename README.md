# When Does Online Adaptation Pay on the Edge?

Reproducibility package for *"When Does Online Adaptation Pay on the Edge? A Leakage-Free
Evaluation of Warmup, Learning-Rate Selection, and Resource Trade-offs for Time-Series
Forecasting"* (under review, IEEE BigData 2026).

Takumi Fujimoto, Hiroaki Nishi — Graduate School of Science and Technology, Keio University.

This repository contains (a) the complete experiment harness, (b) the **result artifacts the
paper is built from** (including the 360-cell optimizer grid `grid.jsonl` and the
learning-rate grid `lr_fairness.jsonl`), and (c) the single-source pipeline that turns those
artifacts into every number and figure in the paper.

## Layout

```
experiments/tsf_edge/    the harness (see "Reproduction map" below)
    data/                datasets: bdg2*.csv shipped; ETT + Appliances via get_data.sh
results/tsf_edge/        the paper's result artifacts (data files + generated macros/figures)
```

## Quickstart 1 — rebuild the paper's numbers & figures WITHOUT a GPU (seconds)

Every number in the paper is a LaTeX macro generated from the shipped artifacts, and every
figure is drawn from them; no hand-typed results anywhere:

```bash
pip install -r requirements.txt
bash experiments/tsf_edge/data/get_data.sh   # download only, no GPU (see the note below)
python experiments/tsf_edge/gen_macros.py    # -> results/tsf_edge/macros.tex (842 macros)
python experiments/tsf_edge/paper_figs.py    # -> results/tsf_edge/*_paper.pdf (the paper's 4 figures + 1 supplementary)
```

`get_data.sh` is needed here even though this step uses no GPU: five of the 842 macros describe
the *composition* of the Appliances stream (`\ApplChannels` and friends) and are computed from
`appliances.csv` itself rather than from a result artifact. Without it `gen_macros.py` prints
`WARNING: dataset composition skipped` and emits 837 macros, four of which the paper references
— so `main.tex` would not compile. Every other macro comes from the shipped artifacts.

## Quickstart 2 — rerun the experiments (GPU)

```bash
pip install -r requirements.txt
bash experiments/tsf_edge/data/get_data.sh   # fetch ETT x4 + UCI Appliances (bdg2*.csv shipped)
python experiments/tsf_edge/combined_grid.py # e.g. the 360-cell grid
```

## Reproduction map (paper item -> script -> artifact)

| Paper item | Script | Artifact | Runtime (1x A100) |
|---|---|---|---|
| Fig. 1 + Table I (C1 warmup sensitivity and validation-only selection; one 2x3 panel grid) | `warmup_confound.py` (static/adapted/benefit curves) **and** `validation_protocol.py` (validation pick + test-selected oracle reference) | `warmup_confound_sgdm.json`, `validation_protocol_sgdm.json` | ~1.5 h each |
| C1 leak-inflation numbers (the stride-1 alternative) | `leakage_check.py` | `leakage_check_sgdm.json` | ~10 min |
| Fig. 2 + Table II (C2 learning-rate selection: default / rehearsed / oracle readings) | `lr_fairness.py` (`--L/--H/--seeds`) | `lr_fairness.jsonl` (10-rate grid over the 360-cell design) | ~16 h total |
| C2 shared-default statistics at scale | `combined_grid.py` | `grid.jsonl` (360 cells) | ~13 h |
| C2 guard: are collapsed high rates a startup transient or a steady-state failure? | `lr_transient_check.py` | `lr_transient.json` | — |
| Fig. 3 (C3 accuracy--memory--compute frontier, 5 seeds) | `frontier_seeds.py` | `frontier_seeds.jsonl` (30 points x 5 seeds; `frontier_data.json` = the retired seed-0 run) | ~30 min |
| Per-update wall-clock (Fig. 3 compute axis) | `frontier_timing.py` | `frontier_timing.json` | ~2 min |
| Fig. 4 (staleness: one row of dataset panels, hue = optimizer, shade = schedule) | `staleness.py` / `staleness.py --strategy full_adam` | `staleness_patchtst_full_sgdm.json` / `staleness_patchtst_full_adam.json` | ~15 min each |
| Table III (BDG2 meter-selection & scale study) | `prep_bdg2_subsets.py`, then `lr_fairness.py --datasets bdg2_fox,bdg2_panther,bdg2_rat_worst,bdg2_rat_all,bdg2_fleet` | `bdg2_*.csv` + the 30 subset rows in `lr_fairness.jsonl` | ~40 min (15-meter subsets); hours for site/fleet |
| Per-update wall-clock vs channel count (the scale numbers in §IV-C) | `scale_timing.py` | `scale_timing_sgdm.json` | — |
| Supplementary: the warmup confound across four adaptation strategies (not in the paper — cut for page budget; figure `m6_strategies_paper.pdf`) | `m6_strategies.py` | `m6_strategies.json` (figure), `m6_strategies_sgdm.json` (macros) | ~2 h |
| Every number in the paper | `gen_macros.py` | `macros.tex` (842 macros) | seconds, no GPU |
| Every figure in the paper | `paper_figs.py` | `*_paper.pdf` | seconds, no GPU |

Runtimes are the measured wall-clock of the paper's runs on one A100; `—` = not recorded.

Two scripts are shipped but **superseded**, kept so the earlier readings stay reproducible:
`frontier.py` (single-seed C3 frontier, replaced by `frontier_seeds.py`) and `regime_figure.py`
(standalone C2 figure, replaced by `paper_figs.regime_paper`).

`grid.jsonl`: one JSON line per cell (6 datasets x 2 backbones x H in {24,48,96} x
L in {96,192} x 5 seeds = 360), with the validation-selected warmup, static/adapted results for
full-SGD and full-Adam, optimizer-state bytes, and the three optimizer-independent probes
(P1 noise, P2 gradient cosine, P3 drift; P3 is post hoc — it uses the test region).
`lr_fairness.jsonl`: 390 lines — the same 360-cell design plus 30 BDG2 meter-selection rows
(5 subsets x 2 backbones x 3 seeds at H=24, L=96) for Table III. Each line carries the full
10-point online-LR sweep (a `{1,3}x10^k` grid from `3e-6` to `1e-1`: validation-rehearsal MSE +
test MSE + benefit per rate, per optimizer) and the rehearsed / test-oracle readings.

## Protocol notes (what makes the evaluation fair)

- **Leakage-free streaming**: non-overlapping windows at stride = horizon; every target is
  scored before it can enter any gradient (`online_eval.py:stream_eval`). `leakage_check.py`
  reproduces the inflation caused by the leaky stride-1 alternative.
- **Validation-selected warmup**: the warmup budget is picked by early-stopping on a held-out
  pre-drift validation slice (`online_eval.py:warm_and_select`), never on test data. All
  downstream measurements share this one selection procedure.
- **Validation-selected online LR (rehearsal)**: each strategy's online learning rate is picked
  by *rehearsing* online adaptation on the same pre-drift validation slice
  (`online_eval.py:select_online_lr`), never on test data. `lr_fairness.py` shows that skipping
  this — running both optimizers at a shared default rate — reverses the SGD-vs-Adam verdict
  (the paper's third confound).
- **SGD means SGD with momentum.** `torch.optim.SGD`'s `momentum` is a free argument, so
  "the torch default" does not imply momentum-free SGD, and nobody deploys the momentum-free
  form. The paper's SGD-family arm is therefore `sgdm` (`online_eval.SGD_STRAT`). The
  pre-migration momentum-free results are kept in the artifacts (`*_sgd*` fields,
  `staleness_patchtst.json`, `leakage_check.json`, ...) so the retired readings stay
  reproducible, but they are not what the paper reports.
- **Per-update wall-clock is measured separately** (`frontier_timing.py`), not taken from the
  evaluation stream: at batch 1 the update is launch-latency bound, so a single sequential
  pass per strategy measures host contention and GPU warm-up rather than the optimizer. The
  published estimator is a median over updates after a discarded warm-up prefix, minimised
  over repeats that interleave every strategy, and it carries a sanity gate (Adam must never
  measure faster than SGD+momentum at the same strategy).

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

Citation: this repository accompanies a paper under review; the citation entry and the paper
link will be added once the venue is decided. Until then, please cite the repository URL and
the commit hash you used.
