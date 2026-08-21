"""Paper-specific figure variants for ieee_big_data/main.tex, regenerated PURELY from the
result data files (no GPU, no experiment rerun): landscape layouts designed at the final IEEE
full-text width (figure* = 7.1 in), fonts sized for print, and NO matplotlib suptitles (the
LaTeX captions carry the message; suptitles would duplicate them).

Sources: warmup_confound.json / validation_protocol.json / frontier_data.json /
staleness_patchtst_full_sgdm.json / grid.jsonl.  Outputs: results/tsf_edge/<name>_paper.{pdf,png}.
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


# ---------------------------------------------------------------------------------------
# ONE palette, one meaning per colour, shared by every figure (R3: colour previously meant
# three different things across Figs. 1-5 -- blue was "adapted model", "validation MSE",
# "full-SGD" and "periodic" in turn, which is exactly the inconsistency a reader carries
# between figures). The rule now: the three reserved hues encode the OPTIMIZER and nothing
# else; anything that is not an optimizer takes a role colour outside them; backbone and
# parameter-subset are carried by marker SHAPE, not by hue.
# Validated with check_palette.py (OKLab dE under normal/deuter/protan/tritanopia): every
# co-occurring set PASSes, except static-grey vs Adam-red at protanopia (dE 6.7), which is
# legal because the static reference is always a DOTTED rule against solid marker curves.
# WHEN YOU CHANGE A HUE HERE, also grep ieee_big_data/main.tex for the colour words in the
# figure captions ("blue", "green", ...): captions name colours in prose, so nothing in the
# macro pipeline catches a caption left describing the previous palette. This has bitten
# three captions already (Figs. 1, 2 and 4).
PAL = {
    "sgd":      "#1f77b4",   # reserved: optimizer identity
    "adam":     "#d62728",
    "sgdm":     "#ff7f0e",
    "sgd_alt":  "#9ecae1",   # same optimizer, second series in one panel (Fig. 5 schedules):
    "adam_alt": "#fc9272",   # same hue, lighter, so "blue = SGD" still reads
    "static":   "0.35",      # the no-adaptation reference (never a series identity)
    "derived":  "#2ca02c",   # a derived diagnostic on a secondary axis (improvement %, val MSE).
                             # Green, not the former dark purple: at OKLab L 62 vs 46 it reads
                             # as a light accent against the grey/blue series while still
                             # clearing 3:1 on white, and it lifts the worst CVD pair in Fig. 1
                             # (vs. the blue adapted curve) from dE 8 to 15.
    "pick":     "0.15",      # a selected/oracle budget, drawn as a rule rather than a series
    "val":      "#e7298a",   # the held-out pre-drift VALIDATION diagnostic (curve + its pick) in
                             # the merged Fig. 1. Purple (#7570b3) was tried first and FAILS
                             # check_palette -- dE 0.2 against the blue adapted curve under
                             # protanopia. This magenta PASSes normal/deuteranopia and is WARN
                             # (dE 8.0 vs static grey) under protanopia, where the dash pattern
                             # and diamond markers carry the difference. It is NOT an optimizer
                             # hue: no Adam series appears in this figure.
}


def warmup_paper():
    """C1a and C1c in ONE 2x3 grid (rows = backbones, cols = datasets), which is legitimate
    because both studies read the same milestone grid and the same static-test curve (verified
    identical between the two dumps). Per panel: static +/- std and adapted on the left axis,
    adaptation benefit % on the right, the held-out pre-drift validation curve rescaled onto the
    left axis (only its argmin matters), and the two picks as rules.
    Source is warmup_confound_SGDM -- the file gen_macros reads. Until 2026-08-12 this figure
    read warmup_confound.json, the pre-migration momentum-free run, so every benefit curve
    disagreed with Table I (e.g. Appliances/DLinear under-warming +13.5% drawn against +33.4%
    tabulated). Both dumps are kept; only the *_sgdm one is the paper's."""
    wc, vp = load("warmup_confound_sgdm.json"), load("validation_protocol_sgdm.json")
    datasets, backbones = ["ETTm2", "appliances", "bdg2"], ["dlinear", "patchtst"]
    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH, 3.6))
    for r, bb in enumerate(backbones):
        for c, ds in enumerate(datasets):
            d, v, ax = wc[f"{ds}|{bb}"], vp[f"{ds}|{bb}"], axes[r, c]
            m = d["milestones"]
            sm, ss = np.array(d["static_mean"]), np.array(d["static_std"])
            ax.plot(m, sm, "o-", color=PAL["static"], label="static (no adapt)", zorder=3)
            ax.fill_between(m, sm - ss, sm + ss, color=PAL["static"], alpha=0.15)
            ax.plot(m, d["adapted_mean"], "s-", color=PAL["sgd"], zorder=4,
                    label="adapted (full·SGD+m)")
            vm = np.array(v["val_mean"])                 # affine rescale onto the static range
            ax.plot(m, sm.min() + (vm - vm.min()) * (sm.max() - sm.min()) / np.ptp(vm),
                    "d--", color=PAL["val"], lw=0.9, ms=2.4, zorder=2,
                    label="held-out pre-drift VAL MSE (rescaled)")
            ax.axvline(v["oracle_step"], color=PAL["pick"], ls=":", lw=1.1, zorder=1,
                       label="test-selected oracle reference")
            ax.axvline(v["val_step"], color=PAL["val"], ls="--", lw=1.1, zorder=1,
                       label="validation early-stop pick")
            ax.set_xscale("log"); ax.grid(alpha=0.3)
            ax.set_title(f"{PRETTY[ds]} / {PRETTY[bb]}")
            if c == 0:
                ax.set_ylabel("online MSE")
            if r == 1:
                ax.set_xlabel("warmup steps")
            ax2 = ax.twinx()
            # paper-wide positive-good sign convention (minor 1): reported benefit = -(file benefit)
            im, ist = -np.array(d["benefit_mean"]), np.array(d["benefit_std"])
            ax2.plot(m, im, "^--", color=PAL["derived"], lw=1.0, ms=2.5, zorder=2)
            ax2.fill_between(m, im - ist, im + ist, color=PAL["derived"], alpha=0.12)
            ax2.tick_params(axis="y", labelcolor=PAL["derived"], labelsize=6)
            if c == len(datasets) - 1:
                ax2.set_ylabel("adaptation benefit %", color=PAL["derived"])
    handles, _ = axes[0, 0].get_legend_handles_labels()
    handles.append(Line2D([], [], color=PAL["derived"], ls="--", marker="^", ms=2.5,
                          label="benefit % (right axis)"))
    fig.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.10),
               frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    save(fig, "warmup_paper")


def frontier_paper():
    """C3, 2x2 (datasets x {memory, compute}), FIVE SEEDS (frontier_seeds.jsonl; R2):
    mean +/- std error bars per strategy point. Memory axis = adaptation memory (gradients +
    optimizer state), which separates full-Adam (12 B/param) from full-SGD+m (8 B/param) at
    equal trainable-parameter count. calib is drawn as an OPEN marker on top of head (their
    adaptation memory is nearly identical, so filled markers would occlude each other).
    NOTE: opt_state_bytes MUST be carried into the per-point dict -- adapt_mem_bytes falls
    back to a label rule that bills anything without "Adam" in its name at 0 state, i.e. it
    silently plots SGD+momentum at plain-SGD memory (half its true footprint, and half of
    what gen_macros emits for the same point, which does pass the measured field)."""
    from frontier import COMBOS, DATASETS, adapt_mem_bytes, pareto
    from frontier_timing import load_timing
    timing = load_timing()          # controlled per-update wall-clock; see frontier_timing.py
    seeds = [json.loads(l) for l in open(os.path.join(RES, "frontier_seeds.jsonl"))]
    style = {lab: (mk, col) for _, _, lab, mk, col in COMBOS}
    # 2x2 at text width gives panels about 1.5x wider than tall; the extra height keeps the
    # log-memory axis from squashing the Pareto staircase into a flat band.
    fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH, 5.0))
    for r, name in enumerate(DATASETS):
        rows = []
        for _, _, lab, mk, col in COMBOS:
            rs = [x for x in seeds if x["dataset"] == name and x["label"] == lab]
            rows.append(dict(label=lab, params=rs[0]["params"],
                             opt_state_bytes=rs[0].get("opt_state_bytes"),
                             ms=timing.get((name, lab), float(np.mean([x["ms"] for x in rs]))),
                             benefit=float(np.mean([x["benefit"] for x in rs])),
                             std=float(np.std([x["benefit"] for x in rs]))))
        for c, (xget, xlab, xname) in enumerate(
                [(adapt_mem_bytes,
                  "adaptation memory: grads + opt. state (B, $\\downarrow$ better)", "memory"),
                 (lambda row: row["ms"],
                  "per-update compute (ms, $\\downarrow$ better)", "compute")]):
            ax = axes[r, c]
            for row in rows:
                mk, col = style[row["label"]]
                open_mk = "calib" in row["label"]          # drawn hollow, on top of head
                ax.errorbar(xget(row), row["benefit"], yerr=row["std"], fmt="none",
                            ecolor=col, elinewidth=0.9, capsize=2, zorder=2)
                ax.scatter(xget(row), row["benefit"], marker=mk, s=55,
                           facecolor="none" if open_mk else col, color=col,
                           edgecolor=col if open_mk else "k", lw=1.2 if open_mk else 0.6,
                           zorder=4 if open_mk else 3)
            if xname == "memory":
                # static (no adaptation: 0 B, 0 %) is always available, so any below-zero
                # point is dominated by it and cannot sit on the frontier
                pf = pareto([(xget(row), row["benefit"]) for row in rows
                             if row["benefit"] >= 0])
                ax.plot([p[0] for p in pf], [p[1] for p in pf], "--", color="0.5", lw=1.0,
                        zorder=1)
                ax.set_xscale("log")
            ax.axhline(0, color="0.7", ls=":", lw=0.8)
            ax.set_xlabel(xlab)
            if c == 0:
                ax.set_ylabel("adaptation benefit %")
            ax.set_title(f"{PRETTY.get(name, name)}: quality vs {xname}")
            ax.grid(alpha=0.3)
    # Legend layout: matplotlib fills a multi-column legend COLUMN-major (11 entries over
    # ncol=4 => columns of 3/3/3/2), so the entries are handed over in COLUMN order. The
    # result is a grid: each column is one backbone x optimizer group, each ROW one parameter
    # subset (full / head / calib). Previously the list was simply COMBOS order, which put the
    # four later-added points at the end and scattered the pairs across the block. COMBOS
    # (= plotting order) is deliberately left alone; only the legend is reordered.
    LEGEND_ORDER = [                                          # None = the Pareto entry, which
        "PatchTST full·SGD+m", "PatchTST head·SGD+m", "PatchTST calib·SGD+m",
        "PatchTST full·Adam",  "PatchTST head·Adam",  "PatchTST calib·Adam",
        "DLinear full·SGD+m",  "DLinear head·SGD+m",  None,   # pads the short DLinear column
        "DLinear full·Adam",   "DLinear head·Adam",           # so the rows stay aligned
    ]
    assert set(LEGEND_ORDER) - {None} == set(style), "LEGEND_ORDER must cover every COMBOS label"
    handles = []
    for lab in LEGEND_ORDER:
        if lab is None:
            handles.append(Line2D([0], [0], ls="--", color="0.5",
                                  label="nondominated (memory axis)"))
            continue
        mk, col = style[lab]
        handles.append(Line2D([0], [0], marker=mk, color="w",
                              markerfacecolor="none" if "calib" in lab else col,
                              markeredgecolor=col if "calib" in lab else "k",
                              markersize=6, label=lab))
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.038),
               frameon=False, fontsize=6.4, handletextpad=0.4, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    save(fig, "frontier_paper")


def staleness_paper():
    """Staleness in ONE row of dataset panels (was 2 optimizer rows x 3 datasets): hue is the
    optimizer, the lighter shade within a hue is the drift-triggered schedule. Merging the rows
    makes the actual claim visible inside a panel -- the drift-vs-periodic sign tracks the
    dataset, not the optimizer -- instead of asking the reader to compare across rows."""
    sg = load("staleness_patchtst_full_sgdm.json")
    ad = json.load(open(os.path.join(RES, "staleness_patchtst_full_adam.json")))
    names = ["ETTh2", "ETTm2", "appliances"]
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 1.95), squeeze=False)
    for c, name in enumerate(names):
        ax = axes[0][c]
        ax.axhline(sg[name]["static"], color=PAL["static"], ls=":", lw=0.9,
                   label="static (no adapt)")
        for st, base, olab in ((sg, "sgd", "SGD+m"), (ad, "adam", "Adam")):
            d = st[name]
            for key, col, mk, slab in (("periodic", PAL[base], "o", "periodic every-$k$"),
                                       ("drift", PAL[base + "_alt"], "s", "drift-triggered")):
                pts = sorted(d[key])
                u = [p[0] for p in pts]
                mm = np.array([p[1] for p in pts])
                ax.plot(u, mm, mk + "-", color=col, ms=2.6, label=f"{olab}, {slab}")
                if len(pts[0]) > 2:                       # multi-seed schema: +/- std band
                    s_ = np.array([p[2] for p in pts])
                    ax.fill_between(u, mm - s_, mm + s_, color=col, alpha=0.13)
        # label the quantity, not the phenomenon: these are margins over the periodic
        # schedule, not a measure of how much the series drifts.
        ax.set_title(f"{PRETTY.get(name, name)}  ($\\Delta$ vs periodic: "
                     f"{sg[name]['win_pct']:+.1f}% / {ad[name]['win_pct']:+.1f}%)", fontsize=7.2)
        ax.set_xlabel("update fraction"); ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("online MSE")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, ncol=5, loc="upper center",
               bbox_to_anchor=(0.5, 1.15), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "staleness_paper")


def regime_paper():
    """C3 (v2), 1x2 from lr_fairness.jsonl — the online-LR default is a third confound:
    (A) benefit vs LR, median+IQR (the two nonnegative-benefit ranges and the default's
        placement);
    (B) per-cell benefit at the default vs at the val-rehearsed LR."""
    rows = [json.loads(l) for l in open(os.path.join(RES, "lr_fairness.jsonl"))]
    core = {"appliances", "bdg2", "ETTm2", "ETTh2", "ETTm1", "ETTh1"}
    # the FULL fair-LR grid (bdg2_* M5 extension subsets excluded from C3 stats)
    rows = [r for r in rows if r["dataset"] in core]
    n = len(rows)
    lrs = sorted(rows[0]["lrs"])
    # R3: the SGD-family arm is SGD+momentum; momentum-free SGD is no longer reported.
    COLS = {"sgdm": PAL["sgd"], "adam": PAL["adam"]}
    LABS = {"sgdm": "full·SGD+m", "adam": "full·Adam"}

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.55))
    for o in ("sgdm", "adam"):
        M = np.array([[r[o][f"{lr:g}"]["benefit"] for lr in lrs] for r in rows])
        M[~np.isfinite(M)] = -1e9          # diverged stream (nan/-inf) = very negative; the
        med, (q1, q3) = np.median(M, axis=0), np.percentile(M, [25, 75], axis=0)  # curve just
        # leaves the axis (ylim floor) at those rates, and the counts row still includes them
        axA.plot(lrs, med, "o-", color=COLS[o], label=f"{LABS[o]} (median)", zorder=3)
        axA.fill_between(lrs, q1, q3, color=COLS[o], alpha=0.15)
        for x, ncur in zip(lrs, (M < 0).sum(axis=0)):  # negative-cell counts along the curve
            if ncur:
                axA.annotate(f"{ncur}", (x, -64), color=COLS[o], fontsize=5.5, ha="center",
                             va="bottom" if o == "sgdm" else "top",
                             xytext=(0, 1.5 if o == "sgdm" else -1.5),
                             textcoords="offset points")
    axA.text(lrs[0], -64, "cells $<$ static:", fontsize=5.5, color="0.3", va="center")
    axA.axhline(0, color="0.4", lw=0.8)
    axA.axvline(1e-3, color="0.25", ls="--", lw=1.0)
    axA.annotate("fixed default\n(the grid ran here)", (1e-3, 36), fontsize=6, ha="right",
                 va="top", xytext=(-3, 0), textcoords="offset points")
    axA.set_xscale("log"); axA.set_ylim(-70, 44)
    axA.set_xlabel("online learning rate")
    axA.set_ylabel(f"benefit % (median+IQR, {n} cells)")
    axA.set_title("(A) Empirical nonnegative-benefit ranges\nand the shared default")
    axA.legend(loc="lower left", framealpha=0.9)
    axA.grid(alpha=0.3, which="both")

    lo, hi, y_floor = -50, 62, -42        # sel benefits below y_floor (incl. the diverged
    for r in rows:                        # rehearsed-SGD picks) are clipped to the bottom edge
        mk = "o" if r["L"] == 96 else "^"
        for o in ("sgdm", "adam"):
            y = r[f"sel_benefit_{o}"]
            y = y_floor if not (y >= y_floor) else y
            axB.scatter(r[o]["0.001"]["benefit"], y, marker=mk, s=13,
                        c=COLS[o], alpha=0.7, edgecolors="none", zorder=3)
    axB.plot([lo, hi], [lo, hi], color="0.6", lw=0.7, ls=":")
    axB.axhline(0, color="0.4", lw=0.8); axB.axvline(0, color="0.4", lw=0.8)
    n_ad_fix = sum(r["adam"]["0.001"]["benefit"] < 0 for r in rows)
    n_ad_sel = sum(r["sel_benefit_adam"] < 0 for r in rows)
    n_sg_fix = sum(r["sgdm"]["0.001"]["benefit"] < 0 for r in rows)
    n_sg_sel = sum(r["sel_benefit_sgdm"] < 0 for r in rows)
    axB.text(0.03, 0.97, f"negative cells @default $\\to$ @rehearsed:\n"
             f"Adam {n_ad_fix}/{n} $\\to$ {n_ad_sel}/{n};  SGD+m {n_sg_fix}/{n} $\\to$ {n_sg_sel}/{n}",
             transform=axB.transAxes, ha="left", va="top", fontsize=6,
             bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.7", lw=0.5))
    axB.set_xlim(lo, hi); axB.set_ylim(y_floor - 3, hi)
    axB.set_xlabel("benefit % at the fixed default ($10^{-3}$)")
    axB.set_ylabel("benefit % at rehearsed LR")
    axB.set_title("(B) Validation-based rehearsal substantially\nreduces Adam's below-static cells")
    axB.legend(handles=[
        Line2D([], [], marker="s", color="w", markerfacecolor=COLS["sgdm"], markersize=4.5,
               label="full·SGD+m"),
        Line2D([], [], marker="s", color="w", markerfacecolor=COLS["adam"], markersize=4.5,
               label="full·Adam"),
        Line2D([], [], marker="o", color="w", markerfacecolor="0.5", markersize=4, label="L=96"),
        Line2D([], [], marker="^", color="w", markerfacecolor="0.5", markersize=4, label="L=192"),
    ], loc="lower right", framealpha=0.9)
    axB.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "regime_paper")


def m6_strategies_paper():
    """C1 (M6), 1x2: the warmup confound is strategy-generic and distorts strategy RANKINGS.
    Static U-shape (gray, left axis; shared across strategies by construction) + each
    strategy's improvement (right axis): under-warming inflates all four, while over-warming
    splits them -- it inflates the full-model arms and deflates PEFT (head/calib) on
    Appliances, the more drift-heavy panel.
    Reads m6_strategies_SGDM -- the file gen_macros reads. Until 2026-08-21 this figure read
    m6_strategies.json, the pre-migration momentum-free run, so the drawn curves disagreed
    with the \MSix* macros (e.g. ETTm2 head over-warm inflation -5.5 drawn against +0.0
    tabulated). Both dumps are kept; only the *_sgdm one is current. Same fix as the one
    applied to Fig. 1 on 2026-08-12."""
    m6 = load("m6_strategies_sgdm.json")
    strats = [("full_sgdm", "full·SGD+m @$10^{-3}$", "#1f77b4"),
              ("full_adam", "full·Adam @$10^{-4}$", "#d62728"),
              ("head_sgdm", "head·SGD+m @$10^{-3}$", "#2ca02c"),
              ("calib_sgdm", "calib·SGD+m @$10^{-3}$", "#9467bd")]
    order = ["ETTm2|patchtst", "appliances|patchtst"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.5))
    for i, (ax, key) in enumerate(zip(axes, order)):
        d = m6[key]
        ds, bb = key.split("|")
        m = d["milestones"]
        sm, ss = np.array(d["static_mean"]), np.array(d["static_std"])
        ax.plot(m, sm, "o-", color="0.35")
        ax.fill_between(m, sm - ss, sm + ss, color=PAL["static"], alpha=0.15)
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
               Line2D([], [], ls=":", color="green", lw=1.1,
                      label="test-selected oracle reference")]
    handles += [Line2D([], [], marker="^", ls="--", color=col, ms=2.5, label=lab)
                for _, lab, col in strats]
    fig.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12),
               frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "m6_strategies_paper")


if __name__ == "__main__":
    warmup_paper()                      # C1a + C1c merged (was warmup_confound + validation_protocol)
    frontier_paper()
    staleness_paper()
    regime_paper()
    m6_strategies_paper()
