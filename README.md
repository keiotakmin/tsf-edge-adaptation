# When Does Online Adaptation Pay on the Edge?

Reproducibility package for *"When Does Online Adaptation Pay on the Edge? A Leakage-Free
Evaluation of Warmup, Learning-Rate Selection, and Resource Trade-offs for Time-Series
Forecasting"* (under review, IEEE BigData 2026).

This repository contains (a) the complete experiment harness, (b) the **result artifacts the
paper is built from** (including the 360-cell optimizer grid `grid.jsonl` and the
learning-rate grid `lr_fairness.jsonl`), and (c) the single-source pipeline that turns those
artifacts into every number and figure in the paper.

It also carries the harness and artifacts of a **follow-up study** that measures 23 update
rules on the same 216-cell grid and proposes a stateless one. That study is under review; its
paper is not public yet, so this repository describes what the code does rather than what the
paper says, and the citation will be added once the venue is decided. Its layer is the
`stage0_*` scripts and result files; see [The follow-up study](#the-follow-up-study) below. The
two layers share the protocol, the data and `online_eval.py`, and are otherwise independent:
the conference layer's scripts import nothing from the follow-up's.

## Layout

```
experiments/tsf_edge/    the harness (see "Reproduction map" below)
    data/                datasets: bdg2*.csv shipped; ETT + Appliances via get_data.sh
    online_optimizers.py the follow-up study's update rules, ObSign among them
    stage0_*.py          the follow-up study's harness, pooling and tables
results/tsf_edge/        both studies' result artifacts (data files + generated macros/figures)
    macros.tex           conference paper: every number, generated
    macros_ext.tex       follow-up study: every number, generated (\Ext... namespace)
    stage0*_optimizers*.jsonl   the follow-up study's cells
```

## Quickstart 1 — rebuild the paper's numbers & figures WITHOUT a GPU (seconds)

Every number in the paper is a LaTeX macro generated from the shipped artifacts, and every
figure is drawn from them; no hand-typed results anywhere:

```bash
pip install -r requirements.txt
bash experiments/tsf_edge/data/get_data.sh   # download only, no GPU (see the note below)
python experiments/tsf_edge/gen_macros.py    # -> results/tsf_edge/macros.tex (842 macros)
python experiments/tsf_edge/paper_figs.py    # -> results/tsf_edge/*_paper.pdf (the paper's 5 figures + 1 supplementary)
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
| Fig. 1 + Table I (C1a warmup sensitivity, 2x3 panels) | `warmup_confound.py` | `warmup_confound_sgdm.json` | ~1.5 h |
| Fig. 2 (C1c validation-only selection, same 2x3 grid as Fig. 1) | `validation_protocol.py` | `validation_protocol_sgdm.json` | ~1.5 h |
| C1 leak-inflation numbers (the stride-1 alternative) | `leakage_check.py` | `leakage_check_sgdm.json` | ~10 min |
| Fig. 3 + Table II (C2 learning-rate selection: default / rehearsed / oracle readings) | `lr_fairness.py` (`--L/--H/--seeds`) | `lr_fairness.jsonl` (10-rate grid over the 360-cell design) | ~16 h total |
| C2 shared-default statistics at scale | `combined_grid.py` | `grid.jsonl` (360 cells) | ~13 h |
| C2 guard: are collapsed high rates a startup transient or a steady-state failure? | `lr_transient_check.py` | `lr_transient.json` | — |
| Fig. 4 (C3 accuracy--memory--compute frontier, 5 seeds) | `frontier_seeds.py` | `frontier_seeds.jsonl` (30 points x 5 seeds; `frontier_data.json` = the retired seed-0 run) | ~30 min |
| Per-update wall-clock (Fig. 4 compute axis) | `frontier_timing.py` | `frontier_timing.json` | ~2 min |
| Fig. 5 (staleness: one row of dataset panels, hue = optimizer, shade = schedule) | `staleness.py` / `staleness.py --strategy full_adam` | `staleness_patchtst_full_sgdm.json` / `staleness_patchtst_full_adam.json` | ~15 min each |
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

## The follow-up study

**Status: under review; the paper is not public and is not named here.** What follows describes
the code and the artifacts in this repository, which stand on their own.

The follow-up takes the third confound of the conference paper — that comparisons across
optimizers are usually run at one rate inherited from one of them — and turns it into a
requirement a deployment imposes: the rate has to be fixed once, before the site is seen. Together with a bound on optimizer
state and the requirement that adapting not be worse than leaving the model frozen, that gives
three requirements, and the study measures 23 update rules against them on the same 216
cells. No existing design class meets all three; ObSign, which caps a sign step at a fixed
fraction of each parameter's own RMS, does, with no optimizer state.

Same rule as the conference layer: **no number is typed by hand.** `gen_macros_stage0.py`
regenerates all of `macros_ext.tex` from the shipped `stage0*.jsonl`, and
`stage0_figs.py` regenerates the two tables and three figures.

```bash
pip install -r requirements.txt                              # includes the two extra deps
python experiments/tsf_edge/gen_macros_stage0.py             # -> results/tsf_edge/macros_ext.tex
python experiments/tsf_edge/stage0_pool.py                   # the pooled tables, as markdown
python experiments/tsf_edge/test_online_optimizers.py        # the update rules' own assertions
```

| Paper item | Script | Artifact | Runtime (1x A100) |
|---|---|---|---|
| the memory-light and learning-rate-free contenders | `stage0_optimizers.py --stage 0` (per `--L/--H` slice) | `stage0_optimizers.jsonl` (216 cells) | — |
| the rules built for non-stationary streams | `run_stage0b.sh` | `stage0b_optimizers.jsonl` | ~27 h |
| ObSign and its ablation | `run_stage0c.sh` | `stage0c_optimizers.jsonl` | ~10 h |
| bracketing fill-in (two top rates; a shared grid is only fair if it brackets every rule's optimum) | `run_stage0_fillin.sh` on two hosts, then `merge_stage0_fillin.py --write` | `stage0_fillin_*.jsonl` | — |
| seeds 3-4 for the leading rules (216 -> 360 cells) | `run_stage0_seeds.sh`, then `merge_stage0_seeds.py --write` | `stage0_seeds34*.jsonl` | — |
| the deployment (parameter-efficient) configurations | `run_g2_peft.sh` | `stage0_optimizers_{calib,head}.jsonl` | ~2.5 h |
| the guard level tau, swept finely enough to locate the no-harm crossing | `run_stage0d.sh` | `stage0d_optimizers.jsonl` | ~7.3 h |
| comparison with PETSA's modules and loss | `petsa_compare.py` (uses `petsa_calib.py`) | rows inside the PEFT artifacts | — |
| Table II (deployability at every shipped rate) | `stage0_figs.requirement_table()` | `requirement_table.tex` | seconds |
| Table III (all rules, both readings) | `stage0_figs.optimizer_table()` | `optimizer_table.tex` | seconds |
| Figs. 1-3 | `stage0_figs.{requirement_gap_paper,knee_paper,lr_response_paper}()` | `*_paper.pdf` | seconds |
| every number in the follow-up study | `gen_macros_stage0.py` | `macros_ext.tex` | seconds, no GPU |
| the palette's colour-blind separation | `check_palette.py` | printed report | seconds |

Runtimes are the measured wall-clock of the runs the paper reports, on one A100; `—` = not
recorded (the fill-in and seed passes ran split across two other hosts).

Reading the artifacts: one JSON line per cell, keyed by (dataset, backbone, L, H, seed), with
each rule's full sweep over the shared 10-rate grid (`{val, test, benefit}` per rate), the
validation-selected and test-oracle rates, and the measured optimizer-state bytes. The four
stage files are separate on purpose — every added run got a new prefix rather than rewriting the
previous artifact — and `stage0_pool.load_cells()` merges them per cell. `stage0_pool.py` is the
single implementation of every pooled statistic the paper quotes; the tables, the figures and
the macros all call it, so they cannot disagree.

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

Citation: this repository accompanies the conference paper named above (under review, IEEE
BigData 2026; preprint: [arXiv:2609.01126](https://arxiv.org/abs/2609.01126)) and a follow-up
study whose paper is also under review and not yet public. Citation entries and links will be
added as the venues are decided. Until then, please cite the repository URL and the commit hash
you used.
