"""Paper-specific figure variants for ieee_big_data/main.tex, regenerated PURELY from the
result data files (no GPU, no experiment rerun): landscape layouts designed at the final IEEE
full-text width (figure* = 7.1 in), fonts sized for print, and NO matplotlib suptitles (the
LaTeX captions carry the message; suptitles would duplicate them).

Sources: warmup_confound.json / validation_protocol.json / frontier_data.json /
staleness_patchtst.json / grid.jsonl.  Outputs: results/tsf_edge/<name>_paper.{pdf,png}.
Run via ieee_big_data/sync_assets.sh (which also regenerates macros and copies assets)."""
from __future__ import annotations
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results", "tsf_edge")
TEXTWIDTH = 7.1                                    # IEEE two-column text width (inches)
PRETTY = {"appliances": "Appliances", "bdg2": "BDG2", "dlinear": "DLinear",
          "patchtst": "PatchTST", "ETTm2": "ETTm2", "ETTh2": "ETTh2",
          "ETTm1": "ETTm1", "ETTh1": "ETTh1"}

plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.3,
    "lines.linewidth": 1.1, "lines.markersize": 3.2,
})


def load(fname):
    return json.load(open(os.path.join(RES, fname)))


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(RES, f"{name}.pdf"))


def warmup_confound_paper():
    """C1a, 2 rows (backbones) x 3 cols (datasets) landscape; static +/- std, adapted,
    benefit on the right axis (red), sweet-spot vline."""
    wc = load("warmup_confound.json")
    datasets, backbones = ["ETTm2", "appliances", "bdg2"], ["dlinear", "patchtst"]
    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH, 3.9))
    for r, bb in enumerate(backbones):
        for c, ds in enumerate(datasets):
            d, ax = wc[f"{ds}|{bb}"], axes[r, c]
            m = d["milestones"]
            sm, ss = np.array(d["static_mean"]), np.array(d["static_std"])
            ax.plot(m, sm, "o-", color="0.35", label="static (no adapt)")
            ax.fill_between(m, sm - ss, sm + ss, color="0.35", alpha=0.15)
            ax.plot(m, d["adapted_mean"], "s-", color="#1f77b4", label="adapted (full-SGD)")
            ax.axvline(d["sweet_step"], color="green", ls=":", lw=1.1, label="sweet spot")
            ax.set_xscale("log"); ax.grid(alpha=0.3)
            ax.set_title(f"{PRETTY[ds]} / {PRETTY[bb]}")
            if c == 0:
                ax.set_ylabel("online MSE")
            if r == 1:
                ax.set_xlabel("warmup steps")
            ax2 = ax.twinx()
            # paper-wide positive-good sign convention (minor 1): improvement = -benefit
            im, ist = -np.array(d["benefit_mean"]), np.array(d["benefit_std"])
            ax2.plot(m, im, "^--", color="#d62728", lw=1.0, ms=2.5)
            ax2.fill_between(m, im - ist, im + ist, color="#d62728", alpha=0.12)
            ax2.tick_params(axis="y", labelcolor="#d62728", labelsize=6)
            if c == len(datasets) - 1:
                ax2.set_ylabel("adaptation improvement %", color="#d62728")
    handles, _ = axes[0, 0].get_legend_handles_labels()
    handles.append(Line2D([], [], color="#d62728", ls="--", marker="^", ms=2.5,
                          label="improvement % (right axis, higher = larger apparent benefit)"))
    fig.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.05),
               frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save(fig, "warmup_confound_paper")


def validation_protocol_paper():
    """C1c, 1x4 landscape: static TEST MSE (left) vs held-out pre-drift VAL MSE (right,
    rescaled -- only its argmin matters), with the oracle and validation picks as vlines."""
    vp = load("validation_protocol.json")
    order = ["ETTm2|dlinear", "ETTm2|patchtst", "appliances|dlinear", "appliances|patchtst"]
    fig, axes = plt.subplots(1, 4, figsize=(TEXTWIDTH, 1.9))
    for ax, key in zip(axes, order):
        d = vp[key]
        ds, bb = key.split("|")
        m = d["milestones"]
        ax.plot(m, d["static_mean"], "o-", color="0.35")
        ax.axvline(d["oracle_step"], color="0.35", ls=":", lw=1.2)
        ax.set_xscale("log"); ax.grid(alpha=0.3)
        ax.set_title(f"{PRETTY[ds]} / {PRETTY[bb]}")
        ax.set_xlabel("warmup steps")
        ax2 = ax.twinx()
        ax2.plot(m, d["val_mean"], "s--", color="#1f77b4")
        ax2.axvline(d["val_step"], color="#1f77b4", ls="--", lw=1.0)
        ax2.set_yticks([])                       # scale irrelevant: only the argmin matters
    axes[0].set_ylabel("static TEST MSE")
    handles = [Line2D([], [], marker="o", color="0.35", label="static TEST MSE (left)"),
               Line2D([], [], ls=":", color="0.35", lw=1.2, label="oracle sweet spot"),
               Line2D([], [], marker="s", ls="--", color="#1f77b4",
                      label="held-out pre-drift VAL MSE (right, rescaled)"),
               Line2D([], [], ls="--", color="#1f77b4", lw=1.0, label="validation early-stop pick")]
    fig.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13),
               frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "validation_protocol_paper")


def frontier_paper():
    """C2, 2x2 (datasets x {memory, compute}); styling constants shared with frontier.py.
    Memory axis = adaptation memory (gradients + optimizer state), which separates full-Adam
    (12 B/param) from full-SGD (4 B/param) at equal trainable-parameter count."""
    from frontier import COMBOS, DATASETS, adapt_mem_bytes, pareto
    data = load("frontier_data.json")
    style = {lab: (mk, col) for _, _, lab, mk, col in COMBOS}
    fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH, 5.2))
    for r, name in enumerate(DATASETS):
        rows = data[name]
        for c, (xget, xlab, xname) in enumerate(
                [(adapt_mem_bytes,
                  "adaptation memory: grads + opt. state (B, $\\downarrow$ better)", "memory"),
                 (lambda row: row["ms"],
                  "per-update compute (ms, $\\downarrow$ better)", "compute")]):
            ax = axes[r, c]
            for row in rows:
                mk, col = style[row["label"]]
                ax.scatter(xget(row), row["benefit"], marker=mk, s=55, color=col,
                           edgecolor="k", lw=0.6, zorder=3)
            if xname == "memory":
                pf = pareto([(xget(row), row["benefit"]) for row in rows])
                ax.plot([p[0] for p in pf], [p[1] for p in pf], "--", color="0.5", lw=1.0,
                        zorder=1)
                ax.set_xscale("log")
            ax.axhline(0, color="0.7", ls=":", lw=0.8)
            ax.set_xlabel(xlab)
            if c == 0:
                ax.set_ylabel("adaptation benefit % ($\\uparrow$ better)")
            ax.set_title(f"{PRETTY.get(name, name)}: quality vs {xname}")
            ax.grid(alpha=0.3)
    handles = [Line2D([0], [0], marker=mk, color="w", markerfacecolor=col,
                      markeredgecolor="k", markersize=6, label=lab)
               for _, _, lab, mk, col in COMBOS]
    handles.append(Line2D([0], [0], ls="--", color="0.5", label="Pareto frontier (memory)"))
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.045),
               frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save(fig, "frontier_paper")


def staleness_paper():
    """Staleness, 2x3 ({full-SGD, full-Adam} x datasets), fair warmup + rehearsed LR:
    online MSE vs update fraction, periodic vs drift-triggered. Two optimizer rows because
    the drift-vs-periodic sign is small and optimizer-dependent (the demoted claim)."""
    variants = [("full-SGD", load("staleness_patchtst.json"))]
    adam_p = os.path.join(RES, "staleness_patchtst_full_adam.json")
    if os.path.exists(adam_p):
        variants.append(("full-Adam", json.load(open(adam_p))))
    names = ["ETTh2", "ETTm2", "appliances"]
    fig, axes = plt.subplots(len(variants), 3, figsize=(TEXTWIDTH, 1.9 * len(variants)),
                             squeeze=False)
    for r, (vlab, st) in enumerate(variants):
        for c, name in enumerate(names):
            ax, d = axes[r][c], st[name]
            ax.axhline(d["static"], color="0.6", ls=":", lw=0.9, label="static (no adapt)")
            for key, col, lab, mk in (("periodic", "#1f77b4", "periodic every-$k$", "o"),
                                      ("drift", "#d62728", "drift-triggered", "s")):
                pts = sorted(d[key])
                u = [p[0] for p in pts]
                m = np.array([p[1] for p in pts])
                ax.plot(u, m, mk + "-", color=col, label=lab)
                if len(pts[0]) > 2:                       # multi-seed schema: +/- std band
                    s = np.array([p[2] for p in pts])
                    ax.fill_between(u, m - s, m + s, color=col, alpha=0.15)
            win, std = d["win_pct"], d.get("win_pct_std")
            tag = ("" if win is None else
                   f" (drift {win:+.1f}%)" if std is None else
                   f" (drift {win:+.1f}$\\pm${std:.1f}%)")
            ax.set_title(f"{PRETTY.get(name, name)} $\\cdot$ {vlab}{tag}", fontsize=7)
            if r == len(variants) - 1:
                ax.set_xlabel("update fraction")
            ax.grid(alpha=0.3)
        axes[r][0].set_ylabel("online MSE")
    axes[0][0].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, "staleness_paper")


def regime_paper():
    """C3 (v2), 1x2 from lr_fairness.jsonl — the online-LR default is a third confound:
    (A) benefit vs LR, median+IQR (the two safety plateaus and the default's placement);
    (B) per-cell benefit at the default vs at the val-rehearsed LR (Adam rescued)."""
    rows = [json.loads(l) for l in open(os.path.join(RES, "lr_fairness.jsonl"))]
    core = {"appliances", "bdg2", "ETTm2", "ETTh2", "ETTm1", "ETTh1"}
    # the FULL fair-LR grid (bdg2_* M5 extension subsets excluded from C3 stats)
    rows = [r for r in rows if r["dataset"] in core]
    n = len(rows)
    lrs = sorted(rows[0]["lrs"])
    COLS = {"sgd": "#1f77b4", "adam": "#d62728"}
    LABS = {"sgd": "full-SGD", "adam": "full-Adam"}

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.8))
    for o in ("sgd", "adam"):
        M = np.array([[r[o][f"{lr:g}"]["benefit"] for lr in lrs] for r in rows])
        med, (q1, q3) = np.median(M, axis=0), np.percentile(M, [25, 75], axis=0)
        axA.plot(lrs, med, "o-", color=COLS[o], label=f"{LABS[o]} (median)", zorder=3)
        axA.fill_between(lrs, q1, q3, color=COLS[o], alpha=0.15)
        for x, n in zip(lrs, (M < 0).sum(axis=0)):     # negative-cell counts along the curve
            if n:
                axA.annotate(f"{n}", (x, -64), color=COLS[o], fontsize=5.5, ha="center",
                             va="bottom" if o == "sgd" else "top",
                             xytext=(0, 1.5 if o == "sgd" else -1.5),
                             textcoords="offset points")
    axA.text(lrs[0], -64, "cells $<$ static:", fontsize=5.5, color="0.3", va="center")
    axA.axhline(0, color="0.4", lw=0.8)
    axA.axvline(1e-3, color="0.25", ls="--", lw=1.0)
    axA.annotate("fixed default\n(the grid ran here)", (1e-3, 36), fontsize=6, ha="right",
                 va="top", xytext=(-3, 0), textcoords="offset points")
    axA.set_xscale("log"); axA.set_ylim(-70, 44)
    axA.set_xlabel("online learning rate")
    axA.set_ylabel(f"benefit % (median+IQR, {n} cells)")
    axA.set_title("(A) LR safety plateaus vs the default")
    axA.legend(loc="lower left", framealpha=0.9)
    axA.grid(alpha=0.3, which="both")

    for r in rows:
        mk = "o" if r["L"] == 96 else "^"
        for o in ("sgd", "adam"):
            axB.scatter(r[o]["0.001"]["benefit"], r[f"sel_benefit_{o}"], marker=mk, s=13,
                        c=COLS[o], alpha=0.7, edgecolors="none", zorder=3)
    lo, hi = -50, 62
    axB.plot([lo, hi], [lo, hi], color="0.6", lw=0.7, ls=":")
    axB.axhline(0, color="0.4", lw=0.8); axB.axvline(0, color="0.4", lw=0.8)
    n_ad_fix = sum(r["adam"]["0.001"]["benefit"] < 0 for r in rows)
    n_ad_sel = sum(r["sel_benefit_adam"] < 0 for r in rows)
    n_sg_fix = sum(r["sgd"]["0.001"]["benefit"] < 0 for r in rows)
    n_sg_sel = sum(r["sel_benefit_sgd"] < 0 for r in rows)
    axB.text(0.03, 0.97, f"negative cells @default $\\to$ @rehearsed:\n"
             f"Adam {n_ad_fix}/{n} $\\to$ {n_ad_sel}/{n};  SGD {n_sg_fix}/{n} $\\to$ {n_sg_sel}/{n}",
             transform=axB.transAxes, ha="left", va="top", fontsize=6,
             bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.7", lw=0.5))
    axB.set_xlim(lo, hi); axB.set_ylim(-8, hi)
    axB.set_xlabel("benefit % at the fixed default ($10^{-3}$)")
    axB.set_ylabel("benefit % at rehearsed LR")
    axB.set_title("(B) fair LR rescues Adam; SGD barely moves")
    axB.legend(handles=[
        Line2D([], [], marker="s", color="w", markerfacecolor=COLS["sgd"], markersize=4.5,
               label="full-SGD"),
        Line2D([], [], marker="s", color="w", markerfacecolor=COLS["adam"], markersize=4.5,
               label="full-Adam"),
        Line2D([], [], marker="o", color="w", markerfacecolor="0.5", markersize=4, label="L=96"),
        Line2D([], [], marker="^", color="w", markerfacecolor="0.5", markersize=4, label="L=192"),
    ], loc="lower right", framealpha=0.9)
    axB.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "regime_paper")


def m6_strategies_paper():
    """C1 (M6), 1x2: the warmup confound is strategy-generic and distorts strategy RANKINGS.
    Static U-shape (gray, left axis; shared across strategies by construction) + each
    strategy's improvement (right axis): under-warming inflates all four, over-warming
    inflates full-model but deflates PEFT (head/calib)."""
    m6 = load("m6_strategies.json")
    strats = [("full_sgd", "full$\\cdot$SGD @$10^{-3}$", "#1f77b4"),
              ("full_adam", "full$\\cdot$Adam @$10^{-4}$", "#d62728"),
              ("head_sgd", "head$\\cdot$SGD @$10^{-3}$", "#2ca02c"),
              ("calib_sgd", "calib$\\cdot$SGD @$10^{-3}$", "#9467bd")]
    order = ["ETTm2|patchtst", "appliances|patchtst"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.5))
    for i, (ax, key) in enumerate(zip(axes, order)):
        d = m6[key]
        ds, bb = key.split("|")
        m = d["milestones"]
        sm, ss = np.array(d["static_mean"]), np.array(d["static_std"])
        ax.plot(m, sm, "o-", color="0.35")
        ax.fill_between(m, sm - ss, sm + ss, color="0.35", alpha=0.15)
        ax.axvline(d["sweet_step"], color="green", ls=":", lw=1.1)
        ax.set_xscale("log"); ax.grid(alpha=0.3)
        ax.set_title(f"{PRETTY[ds]} / {PRETTY[bb]}")
        ax.set_xlabel("warmup steps")
        if i == 0:
            ax.set_ylabel("static online MSE")
        ax2 = ax.twinx()
        for strat, _, col in strats:
            im = np.array(d["strategies"][strat]["imp_mean"])
            ax2.plot(m, im, "^--", color=col, lw=1.0, ms=2.5)
        if i == len(order) - 1:
            ax2.set_ylabel("adaptation improvement %")
    handles = [Line2D([], [], marker="o", color="0.35", label="static (no adapt; left axis)"),
               Line2D([], [], ls=":", color="green", lw=1.1, label="sweet spot")]
    handles += [Line2D([], [], marker="^", ls="--", color=col, ms=2.5, label=lab)
                for _, lab, col in strats]
    fig.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12),
               frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "m6_strategies_paper")


if __name__ == "__main__":
    warmup_confound_paper()
    validation_protocol_paper()
    frontier_paper()
    staleness_paper()
    regime_paper()
    m6_strategies_paper()
