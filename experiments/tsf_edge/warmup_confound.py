"""HEADLINE experiment (3-seed): the measured online-adaptation benefit is NON-MONOTONE in
the base model's warmup budget -- a two-sided confound. We warm along one trajectory per seed
and, at each warmup milestone, evaluate static (no-adapt) and full_sgd (adapt) online MSE on a
clone. The static baseline's test error is U-shaped in warmup: UNDER-warming (undertrained
baseline) and OVER-warming (baseline overfits the PRE-DRIFT warmup segment) both inflate the
reported adaptation benefit; it is honest only at the warmup sweet spot (min static test error).
"""
from __future__ import annotations
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from online_eval import build_model, stream_eval, load_csv, prep

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MILESTONES = [50, 100, 200, 500, 1000, 2000, 4000, 8000, 20000, 50000]
DATASETS = ["ETTm2", "appliances", "bdg2"]   # + BDG2 (2nd real building-energy dataset)
BACKBONES = ["dlinear", "patchtst"]
SEEDS = [0, 1, 2]
L, H, dev = 96, 24, "cuda"


VAL_FRAC = 0.2   # reserve the last 20% of pre-drift as validation -> IDENTICAL split to
                 # validation_protocol.py, so the two figures' static-test curves are comparable


def trajectory(data, backbone, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    d, n_warm, C = prep(data, device=dev)
    n_train = int(n_warm * (1 - VAL_FRAC))       # warm on TRAIN only (matches the protocol split)
    model = build_model(backbone, L, H, C).to(dev); model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    stat, adap = [], []
    for step in range(1, max(MILESTONES) + 1):
        ii = np.random.randint(L, n_train - H, size=32)
        x = torch.stack([d[i - L:i] for i in ii]); y = torch.stack([d[i:i + H] for i in ii])
        opt.zero_grad(); F.mse_loss(model(x), y).backward(); opt.step()
        if step in MILESTONES:
            stat.append(stream_eval(model, d, backbone, n_warm, L, H, "static", device=dev)["mse"])
            adap.append(stream_eval(model, d, backbone, n_warm, L, H, "full_sgd", device=dev)["mse"])
    return np.array(stat), np.array(adap)


fig, axes = plt.subplots(len(DATASETS), len(BACKBONES), figsize=(5 * len(BACKBONES), 4 * len(DATASETS)))
print(f"{'dataset':11s} {'backbone':9s} | benefit% at: {'under('+str(MILESTONES[0])+')':>10s} "
      f"{'sweet-spot':>18s} {'over('+str(MILESTONES[-1])+')':>11s}")
dump = {}                                   # -> warmup_confound.json for gen_macros.py
for r, name in enumerate(DATASETS):
    data = load_csv(os.path.join(ROOT, "experiments/tsf_edge/data", f"{name}.csv"))
    for c, bb in enumerate(BACKBONES):
        res = [trajectory(data, bb, s) for s in SEEDS]                 # (static, adapted) per seed
        S = np.stack([r[0] for r in res]); A = np.stack([r[1] for r in res])
        sm, ss = S.mean(0), S.std(0); am, asd = A.mean(0), A.std(0)
        ben = 100 * (A - S) / S
        bm, bs = ben.mean(0), ben.std(0)
        j = int(np.argmin(sm))                                         # sweet spot = min static test error
        print(f"{name:11s} {bb:9s} | {'':12s} {bm[0]:>+8.1f}% {bm[j]:>+8.1f}%@{MILESTONES[j]:<6d} {bm[-1]:>+8.1f}%")
        dump[f"{name}|{bb}"] = dict(milestones=MILESTONES, static_mean=sm.tolist(),
                                    static_std=ss.tolist(), adapted_mean=am.tolist(),
                                    benefit_mean=bm.tolist(), benefit_std=bs.tolist(),
                                    sweet_idx=j, sweet_step=MILESTONES[j],
                                    under=float(bm[0]), sweet=float(bm[j]), over=float(bm[-1]))
        ax = axes[r, c]
        ax.plot(MILESTONES, sm, "o-", color="0.35", label="static (no adapt)")
        ax.fill_between(MILESTONES, sm - ss, sm + ss, color="0.35", alpha=0.15)
        ax.plot(MILESTONES, am, "s-", color="#1f77b4", label="full_sgd (adapt)")
        ax.fill_between(MILESTONES, am - asd, am + asd, color="#1f77b4", alpha=0.15)
        ax.axvline(MILESTONES[j], color="green", ls=":", lw=1, label="sweet spot")
        ax.set_xscale("log"); ax.set_ylabel("online MSE"); ax.set_xlabel("warmup steps")
        ax.set_title(f"{name} / {bb}"); ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="best")
        ax2 = ax.twinx()
        ax2.plot(MILESTONES, bm, "^--", color="#d62728", lw=1.6)
        ax2.fill_between(MILESTONES, bm - bs, bm + bs, color="#d62728", alpha=0.12)
        ax2.set_ylabel("adaptation benefit %", color="#d62728"); ax2.tick_params(axis="y", labelcolor="#d62728")

fig.suptitle("Adaptation benefit is NON-MONOTONE in warmup budget (3 seeds): under-warming "
             "(undertrained) and over-warming (overfits pre-drift segment) both inflate it; "
             "honest only at the sweet spot", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = os.path.join(ROOT, "results", "tsf_edge")
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(out, f"warmup_confound.{ext}"), dpi=150, bbox_inches="tight")
json.dump(dump, open(os.path.join(out, "warmup_confound.json"), "w"), indent=2)
print("\nsaved", os.path.join(out, "warmup_confound.png"))
