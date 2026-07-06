"""C2 figure (v2; contribution renumbered C3→C2 on 2026-07-07 when the Results order became
C1 warmup → C2 LR-confound → C3 frontier+recipe) — the ONLINE-LR DEFAULT is a third
evaluation confound.
Built from lr_fairness.jsonl (the M1 fair-LR grid: 72 cells = 6 datasets x 2 backbones x
L in {96,192} x 3 seeds, H=24, 8-point LR grid, per-optimizer val-rehearsed LR). Two panels:
  (A) adaptation benefit vs online LR (median + IQR across all cells): BOTH optimizers have an
      LR safety plateau; the fixed default 1e-3 sits INSIDE SGD's plateau and OUTSIDE Adam's --
      that placement, not optimizer intrinsics, manufactured the old "SGD safe / Adam
      over-adapts" asymmetry of the 360-cell grid.
  (B) per-cell benefit at the fixed default (x) vs at the val-rehearsed LR (y): Adam's
      negative-at-default cells are rescued into the upper-left quadrant; SGD hugs the diagonal.
The old drift x noise regime panel (grid.jsonl) is DEMOTED to text: it described where the
default exits Adam's plateau, not which optimizer is better.
"""
from __future__ import annotations
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CORE = {"appliances", "bdg2", "ETTm2", "ETTh2", "ETTm1", "ETTh1"}
rows = [json.loads(l) for l in open(os.path.join(ROOT, "results/tsf_edge/lr_fairness.jsonl"))]
# The FULL fair-LR grid (core datasets; bdg2_* M5 extension subsets excluded from C3 stats).
rows = [r for r in rows if r["dataset"] in CORE]
LRS = sorted(rows[0]["lrs"])
COL = {"sgd": "#1f77b4", "adam": "#d62728"}
LAB = {"sgd": "full-SGD", "adam": "full-Adam"}

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2))

# ---- Panel A: the two LR safety plateaus ----
for o in ("sgd", "adam"):
    M = np.array([[r[o][f"{lr:g}"]["benefit"] for lr in LRS] for r in rows])
    med = np.median(M, axis=0)
    q1, q3 = np.percentile(M, [25, 75], axis=0)
    axA.plot(LRS, med, "o-", color=COL[o], lw=2, ms=6, label=f"{LAB[o]} (median)", zorder=3)
    axA.fill_between(LRS, q1, q3, color=COL[o], alpha=0.15, label=f"{LAB[o]} IQR")
    neg = (M < 0).sum(axis=0)
    for x, n in zip(LRS, neg):                       # negative-cell counts along the curve
        if n:
            axA.annotate(f"{n}", (x, -66), color=COL[o], fontsize=8.5, ha="center",
                         va="bottom" if o == "sgd" else "top",
                         xytext=(0, 2 if o == "sgd" else -2), textcoords="offset points")
axA.text(LRS[0], -66, "cells < static:", fontsize=8.5, color="0.3", ha="left", va="center")
axA.axhline(0, color="0.4", lw=1)
axA.axvline(1e-3, color="0.25", ls="--", lw=1.4)
axA.annotate("the fixed default\n(360-cell grid ran here)", (1e-3, 38), fontsize=9.5,
             ha="right", va="top", xytext=(-6, 0), textcoords="offset points", color="0.15")
axA.set_xscale("log")
axA.set_ylim(-72, 45)
axA.set_xlabel("online learning rate", fontsize=11)
axA.set_ylabel(f"adaptation benefit %  (median + IQR over {len(rows)} cells)", fontsize=11)
axA.set_title("(A) Both optimizers have an LR safety plateau;\n"
              "the default sits inside SGD's and outside Adam's", fontsize=12)
axA.legend(fontsize=9, loc="lower left", framealpha=0.95)
axA.grid(alpha=0.3, which="both")

# ---- Panel B: per-cell rescue at the rehearsed LR ----
for r in rows:
    mk = "o" if r["L"] == 96 else "^"
    for o in ("sgd", "adam"):
        axB.scatter(r[o]["0.001"]["benefit"], r[f"sel_benefit_{o}"], marker=mk, s=48,
                    c=COL[o], alpha=0.7, edgecolors="none", zorder=3)
lo, hi = -50, 62
axB.plot([lo, hi], [lo, hi], color="0.6", lw=1, ls=":")
axB.axhline(0, color="0.4", lw=1)
axB.axvline(0, color="0.4", lw=1)
n_ad_fix = sum(r["adam"]["0.001"]["benefit"] < 0 for r in rows)
n_ad_sel = sum(r["sel_benefit_adam"] < 0 for r in rows)
n_sg_fix = sum(r["sgd"]["0.001"]["benefit"] < 0 for r in rows)
n_sg_sel = sum(r["sel_benefit_sgd"] < 0 for r in rows)
axB.text(0.02, 0.98, "rescued by LR rehearsal:\n"
         f"Adam  {n_ad_fix}/{len(rows)} neg @default → {n_ad_sel}/{len(rows)}\n"
         f"SGD   {n_sg_fix}/{len(rows)} neg @default → {n_sg_sel}/{len(rows)}",
         transform=axB.transAxes, ha="left", va="top", fontsize=9.5,
         bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.7"))
axB.set_xlim(lo, hi); axB.set_ylim(-8, hi)
axB.set_xlabel("benefit % at the fixed default LR ($10^{-3}$)", fontsize=11)
axB.set_ylabel("benefit % at the val-rehearsed LR", fontsize=11)
axB.set_title("(B) Per-cell effect of fair LR selection:\nAdam is rescued, SGD barely moves",
              fontsize=12)
axB.legend(handles=[
    Line2D([], [], marker="s", color="w", markerfacecolor=COL["sgd"], markersize=10, label="full-SGD"),
    Line2D([], [], marker="s", color="w", markerfacecolor=COL["adam"], markersize=10, label="full-Adam"),
    Line2D([], [], marker="o", color="w", markerfacecolor="0.5", markersize=9, label="L=96"),
    Line2D([], [], marker="^", color="w", markerfacecolor="0.5", markersize=9, label="L=192"),
    Line2D([], [], ls=":", color="0.6", label="y = x"),
], fontsize=9, loc="lower right", framealpha=0.95)
axB.grid(alpha=0.3)

_ds = len({r["dataset"] for r in rows})
_sd = len({r["seed"] for r in rows})
_hs = ",".join(str(h) for h in sorted({r["H"] for r in rows}))
fig.suptitle(f"C2: the online-LR default is a third evaluation confound "
             f"({len(rows)} cells = {_ds} datasets × 2 backbones × H∈{{{_hs}}} × L∈{{96,192}} "
             f"× {_sd} seeds; per-optimizer LR rehearsed on the pre-drift validation slice)",
             fontsize=11.5)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = os.path.join(ROOT, "results", "tsf_edge")
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(out, f"regime.{ext}"), dpi=150, bbox_inches="tight")
print("saved", os.path.join(out, "regime.png"))
sel_gap = np.mean([r["sel_benefit_adam"] - r["sel_benefit_sgd"] for r in rows])
print(f"pooled: Adam neg @default {n_ad_fix}/{len(rows)} -> @rehearsed {n_ad_sel}/{len(rows)}; "
      f"mean sel gap Adam-SGD {sel_gap:+.2f} pt")
