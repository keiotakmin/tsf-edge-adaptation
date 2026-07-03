"""C3 regime figure from grid.jsonl (dimensions read from the file). Two honest panels:
  (A) the online-optimizer winner in the 2-D drift x noise plane -- shows Adam wins the
      high-drift / moderate-noise corner but OVER-ADAPTS (goes negative) elsewhere, so drift alone
      (P3) is not a clean 1-D threshold (the ETTh1 high-noise counterexample is annotated);
  (B) the decisive asymmetry -- SGD's benefit never drops below ~0 (robust floor, zero optimizer
      state) while Adam has a heavy negative tail (worse than static).
Readable by construction: large fonts, explicit legend, annotated exceptions (addresses the earlier
C2 legibility complaint).
"""
from __future__ import annotations
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
rows = [json.loads(l) for l in open(os.path.join(ROOT, "results/tsf_edge/grid.jsonl"))]
def winner(r): return "SGD" if r["benefit_sgd"] >= r["benefit_adam"] else "Adam"

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2))

# ---- Panel A: winner in the drift x noise plane ----
for r in rows:
    x, y = r["p3_drift"], r["p1_noise"]
    w = winner(r)
    gap = r["benefit_adam"] - r["benefit_sgd"]          # >0 => Adam better
    adam_neg = r["benefit_adam"] < 0
    m = "s" if w == "Adam" else "o"
    col = "#d62728" if w == "Adam" else "#1f77b4"
    edge = "black" if adam_neg else col
    axA.scatter(x, y, marker=m, s=90 + 6 * abs(gap), c=col, edgecolors=edge,
                linewidths=1.8 if adam_neg else 0.6, alpha=0.85, zorder=3)
# annotate the sharp counterexample: ETTh1/patchtst (high drift AND high noise -> Adam over-adapts)
_ann = [r for r in rows if r["dataset"] == "ETTh1" and r["backbone"] == "patchtst" and r["H"] == 96]
if _ann:
    r = _ann[0]                                              # annotate once (not once per seed)
    worst = min(x["benefit_adam"] for x in _ann)
    axA.annotate(f"ETTh1/PatchTST:\nhigh drift + high noise,\nyet Adam over-adapts ({worst:.0f}%)",
                 (r["p3_drift"], r["p1_noise"]), xytext=(1.05, 0.165), fontsize=8.5,
                 arrowprops=dict(arrowstyle="->", color="black", lw=1), ha="left")
axA.axvline(1.0, color="0.6", ls="--", lw=1)
axA.text(1.03, 0.135, "drift=1\n(test=val)", fontsize=8, color="0.4", rotation=90, va="center")
axA.set_xlabel("P3  drift strength  (static test-MSE / val-MSE)", fontsize=11)
axA.set_ylabel("P1  noise  (first-difference variance)", fontsize=11)
axA.set_title("(A) Which optimizer wins, in the drift × noise plane", fontsize=12)
from matplotlib.lines import Line2D
axA.legend(handles=[
    Line2D([], [], marker="o", color="w", markerfacecolor="#1f77b4", markersize=11, label="SGD wins"),
    Line2D([], [], marker="s", color="w", markerfacecolor="#d62728", markersize=11, label="Adam wins"),
    Line2D([], [], marker="s", color="w", markerfacecolor="#d62728", markeredgecolor="black",
           markeredgewidth=1.8, markersize=11, label="Adam < static (over-adapt)"),
], fontsize=9, loc="upper right", framealpha=0.95)
axA.grid(alpha=0.3)

# ---- Panel B: the asymmetry -- SGD floor vs Adam negative tail ----
bs = np.array([r["benefit_sgd"] for r in rows])
ba = np.array([r["benefit_adam"] for r in rows])
xj = np.random.default_rng(0).uniform(-0.13, 0.13, size=len(rows))
axB.scatter(0 + xj, bs, c="#1f77b4", s=45, alpha=0.75, label="full-SGD")
axB.scatter(1 + xj, ba, c="#d62728", s=45, alpha=0.75, label="full-Adam")
axB.axhline(0, color="0.4", lw=1.2, ls="-")
axB.hlines(bs.min(), -0.25, 0.25, color="#1f77b4", lw=2.5)
axB.hlines(ba.min(), 0.75, 1.25, color="#d62728", lw=2.5)
axB.text(0, bs.min() - 4, f"SGD floor {bs.min():+.1f}%", ha="center", va="top", fontsize=9,
         color="#1f77b4", fontweight="bold")
axB.text(1, ba.min() - 4, f"Adam worst {ba.min():+.1f}%", ha="center", va="top", fontsize=9,
         color="#d62728", fontweight="bold")
axB.text(0.02, 0.98, f"{(ba<0).sum()}/{len(rows)} Adam cells\nworse than static",
         transform=axB.transAxes, ha="left", va="top", fontsize=9, color="#d62728",
         bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
axB.set_ylim(ba.min() - 12, max(bs.max(), ba.max()) + 4)
axB.set_xticks([0, 1]); axB.set_xticklabels(["full-SGD\n(0 optimizer state)", "full-Adam\n(2× state)"], fontsize=10)
axB.set_ylabel("adaptation benefit % (fair warmup)", fontsize=11)
axB.set_title("(B) The recipe asymmetry: SGD never diverges, Adam has a heavy negative tail", fontsize=11.5)
axB.grid(alpha=0.3, axis="y")

_ds, _bb = len({r["dataset"] for r in rows}), len({r["backbone"] for r in rows})
_hs = ",".join(str(h) for h in sorted({r["H"] for r in rows}))
_ls = ",".join(str(l) for l in sorted({r["L"] for r in rows}))
_sd = len({r["seed"] for r in rows})
fig.suptitle(f"C3 regime analysis ({len(rows)} cells = {_ds} datasets × {_bb} backbones × "
             f"H∈{{{_hs}}} × L∈{{{_ls}}} × {_sd} seeds): "
             "P3-drift is the best single indicator but the split is 2-D; SGD is the safe default",
             fontsize=11.5)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = os.path.join(ROOT, "results", "tsf_edge")
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(out, f"regime.{ext}"), dpi=150, bbox_inches="tight")
print("saved", os.path.join(out, "regime.png"))
print(f"SGD wins {sum(winner(r)=='SGD' for r in rows)}, Adam wins {sum(winner(r)=='Adam' for r in rows)}; "
      f"SGD floor {bs.min():+.1f}%, Adam worst {ba.min():+.1f}%, Adam-negative {(ba<0).sum()}/{len(rows)}")
