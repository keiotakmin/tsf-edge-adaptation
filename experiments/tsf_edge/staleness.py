"""Staleness axis: the forecast-quality vs update-frequency frontier, and whether
DRIFT-TRIGGERED adaptation Pareto-dominates PERIODIC (fewer updates for the same error).

For each dataset we warm the base model ONCE at the FAIR warmup (C1 deployable protocol:
held-out pre-drift validation early-stopping, `warm_and_select`; formerly a fixed 2000-step
warmup, which C1 shows can skew the measured benefit), then rehearsal-select the ONLINE LR on
the same pre-drift validation slice (`select_online_lr`, M1 fair-LR protocol; formerly a fixed
lr=1e-3, which sat outside Adam's safety plateau) and sweep periodic every-k and
drift-triggered (error-spike) schedules on clones of that one checkpoint, all at the selected
LR. Plot MSE vs update fraction (lower-left = better) and report the compute/energy saving of
drift-triggering at matched quality. Default strategy full_sgd (zero optimizer state -- the
memory-free adapter); --strategy full_adam measures the fair-LR quality winner and writes to
a suffixed output (staleness_<bb>_full_adam.json) so the SGD file keeps its identity.
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
ARGS = ap.parse_args()
STRAT = ARGS.strategy
SUFFIX = "" if STRAT == "full_sgd" else f"_{STRAT}"
DATASETS = ARGS.datasets.split(",")
BB = ARGS.backbone
KS = [1, 2, 4, 8, 16]
TAUS = [1.1, 1.3, 1.5, 2.0, 3.0]
L, H, SEED, dev = 96, 24, 0, "cuda"


def sweep(name):
    data = load_csv(os.path.join(ROOT, "experiments/tsf_edge/data", f"{name}.csv"))
    d, n_warm, C = prep(data, device=dev)
    n_train = int(n_warm * (1 - VAL_FRAC))
    model, wstep, _ = warm_and_select(BB, L, H, C, d, n_train, n_warm, SEED)   # fair warmup, once
    lr, _ = select_online_lr(model, d, BB, n_train, n_warm, L, H, STRAT, device=dev)  # fair LR,
    static = stream_eval(model, d, BB, n_warm, L, H, "static", device=dev)["mse"]     # once
    per = [(r["update_frac"], r["mse"]) for r in
           (stream_eval(model, d, BB, n_warm, L, H, STRAT, lr=lr, device=dev,
                        schedule="every", k=k) for k in KS)]
    dri = [(r["update_frac"], r["mse"]) for r in
           (stream_eval(model, d, BB, n_warm, L, H, STRAT, lr=lr, device=dev,
                        schedule="drift", tau=t) for t in TAUS)]
    return static, sorted(per), sorted(dri), wstep, lr


def pareto_win(per, dri):
    """At each drift point, is its MSE below the periodic curve interpolated at the same
    update fraction? Return mean MSE improvement of drift over periodic at matched budget."""
    pf = np.array(per); df = np.array(dri)
    diffs = []
    for uf, mse in df:
        if uf < pf[:, 0].min() or uf > pf[:, 0].max():
            continue
        per_mse = np.interp(uf, pf[:, 0], pf[:, 1])
        diffs.append((per_mse - mse) / per_mse)          # >0 => drift better at same budget
    return float(np.mean(diffs)) * 100 if diffs else float("nan")


fig, axes = plt.subplots(1, len(DATASETS), figsize=(4 * len(DATASETS), 3.6))
print(f"{'dataset':8s} {'warm':>6s} {'lr':>7s} {'static MSE':>10s} {'best MSE':>9s} {'drift vs periodic @matched budget':>34s}")
dump = {}                                   # -> staleness_{BB}{SUFFIX}.json for gen_macros.py
for ax, name in zip(axes, DATASETS):
    static, per, dri, wstep, lr = sweep(name)
    win = pareto_win(per, dri)
    ax.axhline(static, color="0.6", ls=":", lw=1, label="static (no adapt)")
    ax.plot([u for u, _ in per], [m for _, m in per], "o-", color="#1f77b4", ms=4, label="periodic every-k")
    ax.plot([u for u, _ in dri], [m for _, m in dri], "s-", color="#d62728", ms=4, label="drift-triggered")
    ax.set_title(f"{name}  (drift {win:+.1f}% @budget)"); ax.set_xlabel("update fraction")
    ax.set_ylabel("online MSE"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    best = min(min(m for _, m in per), min(m for _, m in dri))
    print(f"{name:8s} {wstep:>6d} {lr:>7g} {static:>10.4f} {best:>9.4f} {win:>33.1f}%", flush=True)
    dump[name] = dict(warm=wstep, lr=lr, static=static, best=best,
                      win_pct=None if np.isnan(win) else win, periodic=per, drift=dri)

fig.suptitle(f"Quality vs update-frequency frontier ({BB} {STRAT}, fair warmup + fair LR): does "
             "drift-triggered adaptation Pareto-dominate periodic? (lower-left better)", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = os.path.join(ROOT, "results", "tsf_edge")
os.makedirs(out, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(out, f"staleness_frontier_{BB}{SUFFIX}.{ext}"), dpi=150, bbox_inches="tight")
json.dump(dump, open(os.path.join(out, f"staleness_{BB}{SUFFIX}.json"), "w"), indent=2)
print("\nsaved", os.path.join(out, f"staleness_frontier_{BB}{SUFFIX}.png"))
print("Read: drift +x% @budget means drift-triggered gets x% lower MSE than periodic at the "
      "same update fraction (i.e. same compute/energy). Positive => drift-triggering wins.")
