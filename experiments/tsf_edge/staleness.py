"""Staleness axis: the forecast-quality vs update-frequency frontier, and whether
DRIFT-TRIGGERED adaptation Pareto-dominates PERIODIC (fewer updates for the same error).

For each dataset and SEED (referee W2: formerly single-seed) we warm the base model at the
FAIR warmup (C1 deployable protocol, `warm_and_select`), rehearsal-select the ONLINE LR on the
same pre-drift validation slice (`select_online_lr`, the fair-LR protocol), and sweep periodic
every-k and drift-triggered (error-spike) schedules on clones of that checkpoint, all at the
selected LR. Curves are aggregated ACROSS SEEDS per schedule knob (per k / per tau, not per
update-fraction, which varies by seed for the drift trigger); the drift-vs-periodic win% at
matched update budget is computed per seed and reported mean +/- std. Plot MSE vs update
fraction (lower-left = better). Default strategy full_sgd (zero optimizer state); --strategy
full_adam measures the fair-LR quality winner and writes to a suffixed output
(staleness_<bb>_full_adam.json) so the SGD file keeps its identity.
"""
from __future__ import annotations
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from online_eval import VAL_FRAC, load_csv, prep, select_online_lr, stream_eval, warm_and_select

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--backbone", default="patchtst")           # high-adaptation regime by default
ap.add_argument("--datasets", default="ETTh2,ETTm2,appliances")
ap.add_argument("--strategy", default="full_sgd")
ap.add_argument("--seeds", default="0,1,2")                 # referee W2: 3 seeds
ARGS = ap.parse_args()
STRAT = ARGS.strategy
SUFFIX = "" if STRAT == "full_sgd" else f"_{STRAT}"
DATASETS = ARGS.datasets.split(",")
BB = ARGS.backbone
SEEDS = [int(s) for s in ARGS.seeds.split(",")]
KS = [1, 2, 4, 8, 16]
TAUS = [1.1, 1.3, 1.5, 2.0, 3.0]
L, H, dev = 96, 24, "cuda"


def sweep_seed(data, seed):
    d, n_warm, C = prep(data, device=dev)
    n_train = int(n_warm * (1 - VAL_FRAC))
    model, wstep, _ = warm_and_select(BB, L, H, C, d, n_train, n_warm, seed)   # fair warmup
    lr, _ = select_online_lr(model, d, BB, n_train, n_warm, L, H, STRAT, device=dev)  # fair LR
    static = stream_eval(model, d, BB, n_warm, L, H, "static", device=dev)["mse"]
    per = [(r["update_frac"], r["mse"]) for r in
           (stream_eval(model, d, BB, n_warm, L, H, STRAT, lr=lr, device=dev,
                        schedule="every", k=k) for k in KS)]                   # aligned by k
    dri = [(r["update_frac"], r["mse"]) for r in
           (stream_eval(model, d, BB, n_warm, L, H, STRAT, lr=lr, device=dev,
                        schedule="drift", tau=t) for t in TAUS)]               # aligned by tau
    return static, per, dri, wstep, lr


def pareto_win(per, dri):
    """At each drift point, is its MSE below the periodic curve interpolated at the same
    update fraction? Return mean MSE improvement of drift over periodic at matched budget."""
    pf = np.array(sorted(per)); df = np.array(dri)
    diffs = []
    for uf, mse in df:
        if uf < pf[:, 0].min() or uf > pf[:, 0].max():
            continue
        per_mse = np.interp(uf, pf[:, 0], pf[:, 1])
        diffs.append((per_mse - mse) / per_mse)          # >0 => drift better at same budget
    return float(np.mean(diffs)) * 100 if diffs else float("nan")


def agg_curve(curves):
    """curves: per-seed lists aligned by schedule knob -> [(mean uf, mean mse, std mse)]."""
    A = np.array(curves)                                  # [seeds, knobs, 2]
    return [(float(A[:, j, 0].mean()), float(A[:, j, 1].mean()), float(A[:, j, 1].std()))
            for j in range(A.shape[1])]


fig, axes = plt.subplots(1, len(DATASETS), figsize=(4 * len(DATASETS), 3.6))
print(f"{'dataset':8s} {'warm':>17s} {'lr':>23s} {'static MSE':>10s} "
      f"{'drift vs periodic @matched budget':>34s}")
dump = {}                                   # -> staleness_{BB}{SUFFIX}.json for gen_macros.py
for ax, name in zip(np.atleast_1d(axes), DATASETS):
    data = load_csv(os.path.join(ROOT, "experiments/tsf_edge/data", f"{name}.csv"))
    runs = [sweep_seed(data, s) for s in SEEDS]
    statics = [r[0] for r in runs]
    per = agg_curve([r[1] for r in runs])
    dri = agg_curve([r[2] for r in runs])
    wins = [pareto_win(r[1], r[2]) for r in runs]         # per-seed, then mean +/- std
    wins = [w for w in wins if w == w]
    win_m = float(np.mean(wins)) if wins else None
    win_s = float(np.std(wins)) if wins else None
    warms = [r[3] for r in runs]; lrs = [r[4] for r in runs]
    sm, ss = float(np.mean(statics)), float(np.std(statics))

    ax.axhline(sm, color="0.6", ls=":", lw=1, label="static (no adapt)")
    for curve, col, lab, mk in [(per, "#1f77b4", "periodic every-k", "o"),
                                (dri, "#d62728", "drift-triggered", "s")]:
        pts = sorted(curve)
        u, m, s = zip(*pts)
        ax.plot(u, m, mk + "-", color=col, ms=4, label=lab)
        ax.fill_between(u, np.array(m) - s, np.array(m) + s, color=col, alpha=0.15)
    tag = "n/a" if win_m is None else f"{win_m:+.1f}±{win_s:.1f}%"
    ax.set_title(f"{name}  (drift {tag} @budget)"); ax.set_xlabel("update fraction")
    ax.set_ylabel("online MSE"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    best = min(min(m for _, m, _ in per), min(m for _, m, _ in dri))
    print(f"{name:8s} {str(warms):>17s} {str([f'{x:g}' for x in lrs]):>23s} {sm:>10.4f} "
          f"{tag:>34s}", flush=True)
    dump[name] = dict(seeds=len(SEEDS), warm=warms, lr=lrs, static=sm, static_std=ss,
                      best=best, win_pct=win_m, win_pct_std=win_s, periodic=per, drift=dri)

fig.suptitle(f"Quality vs update-frequency frontier ({BB} {STRAT}, fair warmup + fair LR, "
             f"{len(SEEDS)} seeds mean±std): periodic vs drift-triggered (lower-left better)",
             fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = os.path.join(ROOT, "results", "tsf_edge")
os.makedirs(out, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(out, f"staleness_frontier_{BB}{SUFFIX}.{ext}"), dpi=150, bbox_inches="tight")
json.dump(dump, open(os.path.join(out, f"staleness_{BB}{SUFFIX}.json"), "w"), indent=2)
print("\nsaved", os.path.join(out, f"staleness_frontier_{BB}{SUFFIX}.png"))
print("Read: drift +x% @budget means drift-triggered gets x% lower MSE than periodic at the "
      "same update fraction (same compute/energy), mean±std over seeds. Positive => drift wins.")
