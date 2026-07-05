"""M6 (referee): does the C1 warmup confound generalize beyond full-SGD?

Mechanistically, the STATIC baseline's U-shape is strategy-independent (the baseline does not
depend on the adaptation strategy), so the confound's existence is generic by construction;
what varies per strategy is the benefit magnitude at each warmup budget. Here we replicate the
C1 trajectory experiment on the two strongest panels (ETTm2/PatchTST, Appliances/PatchTST;
3 seeds, milestones -> 50k) with FOUR adaptation strategies, each at a plateau-appropriate
fixed rate (per M1, 1e-3 sits outside Adam's safety plateau, so full-Adam runs at 1e-4;
SGD-family strategies run at 1e-3 exactly as in the C1 experiment):
  full_sgd@1e-3, full_adam@1e-4, head_sgd@1e-3, calib_sgd@1e-3.
Outputs results/tsf_edge/m6_strategies.json (consumed by gen_macros.py) and
m6_strategies.{png,pdf}. Improvement% = 100*(static-adapted)/static (positive-good).
"""
from __future__ import annotations
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from online_eval import VAL_FRAC, build_model, load_csv, prep, stream_eval

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MILESTONES = [50, 100, 200, 500, 1000, 2000, 4000, 8000, 20000, 50000]
DATASETS = ["ETTm2", "appliances"]
STRATS = [("full_sgd", 1e-3), ("full_adam", 1e-4), ("head_sgd", 1e-3), ("calib_sgd", 1e-3)]
SEEDS = [0, 1, 2]
BB, L, H, dev = "patchtst", 96, 24, "cuda"
COLORS = {"full_sgd": "#1f77b4", "full_adam": "#d62728",
          "head_sgd": "#2ca02c", "calib_sgd": "#9467bd"}


def trajectory(data, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    d, n_warm, C = prep(data, device=dev)
    n_train = int(n_warm * (1 - VAL_FRAC))       # same split as warmup_confound.py
    model = build_model(BB, L, H, C).to(dev); model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    stat = []
    imp = {s: [] for s, _ in STRATS}
    for step in range(1, max(MILESTONES) + 1):
        ii = np.random.randint(L, n_train - H, size=32)
        x = torch.stack([d[i - L:i] for i in ii]); y = torch.stack([d[i:i + H] for i in ii])
        opt.zero_grad(); F.mse_loss(model(x), y).backward(); opt.step()
        if step in MILESTONES:
            s = stream_eval(model, d, BB, n_warm, L, H, "static", device=dev)["mse"]
            stat.append(s)
            for strat, lr in STRATS:
                a = stream_eval(model, d, BB, n_warm, L, H, strat, lr=lr, device=dev)["mse"]
                imp[strat].append(100 * (s - a) / s)
    return np.array(stat), {s: np.array(v) for s, v in imp.items()}


fig, axes = plt.subplots(1, len(DATASETS), figsize=(6.2 * len(DATASETS), 4.4))
dump = {}
print(f"{'dataset':11s} {'strategy':10s} | improvement% at: {'under(50)':>10s} {'sweet':>14s} {'over(50k)':>10s}")
for ax, name in zip(axes, DATASETS):
    data = load_csv(os.path.join(ROOT, "experiments/tsf_edge/data", f"{name}.csv"))
    res = [trajectory(data, s) for s in SEEDS]
    S = np.stack([r[0] for r in res])
    sm, ss = S.mean(0), S.std(0)
    j = int(np.argmin(sm))                                  # sweet spot = min static test error
    entry = dict(milestones=MILESTONES, static_mean=sm.tolist(), static_std=ss.tolist(),
                 sweet_idx=j, sweet_step=MILESTONES[j], strategies={})
    ax.plot(MILESTONES, sm, "o-", color="0.35", label="static (left)")
    ax.fill_between(MILESTONES, sm - ss, sm + ss, color="0.35", alpha=0.15)
    ax.axvline(MILESTONES[j], color="green", ls=":", lw=1.2)
    ax.set_xscale("log"); ax.set_xlabel("warmup steps"); ax.set_ylabel("static online MSE")
    ax.set_title(name); ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    for strat, lr in STRATS:
        I = np.stack([r[1][strat] for r in res])
        im, ist = I.mean(0), I.std(0)
        entry["strategies"][strat] = dict(lr=lr, imp_mean=im.tolist(), imp_std=ist.tolist(),
                                          under=float(im[0]), sweet=float(im[j]),
                                          over=float(im[-1]),
                                          under_infl=float(im[0] - im[j]),
                                          over_infl=float(im[-1] - im[j]))
        ax2.plot(MILESTONES, im, "^--", color=COLORS[strat], lw=1.1, ms=3,
                 label=f"{strat}@{lr:g}")
        print(f"{name:11s} {strat:10s} | {im[0]:>+9.1f}% {im[j]:>+8.1f}%@{MILESTONES[j]:<5d} "
              f"{im[-1]:>+9.1f}%")
    ax2.set_ylabel("adaptation improvement % (right)")
    ax2.legend(fontsize=7, loc="center right")
    dump[f"{name}|{BB}"] = entry

fig.suptitle("M6: the warmup confound is strategy-generic (PatchTST, 3 seeds; static U-shape "
             "shared, improvement inflated at both ends for every strategy)", fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.93))
out = os.path.join(ROOT, "results", "tsf_edge")
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(out, f"m6_strategies.{ext}"), dpi=150, bbox_inches="tight")
json.dump(dump, open(os.path.join(out, "m6_strategies.json"), "w"), indent=2)
print("\nsaved", os.path.join(out, "m6_strategies.png"), "and m6_strategies.json")
