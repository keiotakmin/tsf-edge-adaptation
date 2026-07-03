"""Staleness axis: the forecast-quality vs update-frequency frontier, and whether
DRIFT-TRIGGERED adaptation Pareto-dominates PERIODIC (fewer updates for the same error).

For each dataset we warm the base model ONCE at the FAIR warmup (C1 deployable protocol:
held-out pre-drift validation early-stopping, `warm_and_select`; formerly a fixed 2000-step
warmup, which C1 shows can skew the measured benefit), then sweep periodic every-k and
drift-triggered (error-spike) schedules on clones of that one checkpoint. Plot MSE vs update
fraction (lower-left = better) and report the compute/energy saving of drift-triggering at
matched quality. Uses full_sgd (zero optimizer state -- the memory-free adapter that
milestone 2 found competitive on drift-heavy data).
"""
from __future__ import annotations
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from online_eval import VAL_FRAC, load_csv, prep, stream_eval, warm_and_select

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--backbone", default="patchtst")           # high-adaptation regime by default
ap.add_argument("--datasets", default="ETTh2,ETTm2,appliances")
ARGS = ap.parse_args()
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
    static = stream_eval(model, d, BB, n_warm, L, H, "static", device=dev)["mse"]
    per = [(r["update_frac"], r["mse"]) for r in
           (stream_eval(model, d, BB, n_warm, L, H, "full_sgd", device=dev,
                        schedule="every", k=k) for k in KS)]
    dri = [(r["update_frac"], r["mse"]) for r in
           (stream_eval(model, d, BB, n_warm, L, H, "full_sgd", device=dev,
                        schedule="drift", tau=t) for t in TAUS)]
    return static, sorted(per), sorted(dri), wstep


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
print(f"{'dataset':8s} {'warm':>6s} {'static MSE':>10s} {'best MSE':>9s} {'drift vs periodic @matched budget':>34s}")
dump = {}                                   # -> staleness_{BB}.json for gen_macros.py
for ax, name in zip(axes, DATASETS):
    static, per, dri, wstep = sweep(name)
    win = pareto_win(per, dri)
    ax.axhline(static, color="0.6", ls=":", lw=1, label="static (no adapt)")
    ax.plot([u for u, _ in per], [m for _, m in per], "o-", color="#1f77b4", ms=4, label="periodic every-k")
    ax.plot([u for u, _ in dri], [m for _, m in dri], "s-", color="#d62728", ms=4, label="drift-triggered")
    ax.set_title(f"{name}  (drift {win:+.1f}% @budget)"); ax.set_xlabel("update fraction")
    ax.set_ylabel("online MSE"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    best = min(min(m for _, m in per), min(m for _, m in dri))
    print(f"{name:8s} {wstep:>6d} {static:>10.4f} {best:>9.4f} {win:>33.1f}%", flush=True)
    dump[name] = dict(warm=wstep, static=static, best=best,
                      win_pct=None if np.isnan(win) else win, periodic=per, drift=dri)

fig.suptitle(f"Quality vs update-frequency frontier ({BB}, fair warmup): does drift-triggered "
             "adaptation Pareto-dominate periodic? (lower-left better)", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = os.path.join(ROOT, "results", "tsf_edge")
os.makedirs(out, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(out, f"staleness_frontier_{BB}.{ext}"), dpi=150, bbox_inches="tight")
json.dump(dump, open(os.path.join(out, f"staleness_{BB}.json"), "w"), indent=2)
print("\nsaved", os.path.join(out, f"staleness_frontier_{BB}.png"))
print("Read: drift +x% @budget means drift-triggered gets x% lower MSE than periodic at the "
      "same update fraction (i.e. same compute/energy). Positive => drift-triggering wins.")
