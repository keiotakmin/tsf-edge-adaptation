"""Stage-0 extension variants of the two paper figures, WITH Lion and AdaFactor.
Deliberately written as separate outputs (*_stage0_paper.{pdf,png}) so the frozen
ieee_big_data submission figures (frontier_paper / regime_paper) are untouched.

  lr_response_paper   -- the C2 LR-confound figure, 4 optimizers on the SAME 216 cells
                           (6 datasets x 2 backbones x L{96,192} x H{24,48,96} x seeds 0-2):
                           sgd/adam surfaces from lr_fairness.jsonl restricted to the stage-0
                           cell set; lion/adafactor from stage0_optimizers.jsonl.
  frontier_stage0_paper -- the C3 frontier figure + PatchTST full-Lion / full-AdaFactor
                           points (seed 0, L96/H24, same warm_and_select protocol; memory =
                           4 B/param gradients + MEASURED optimizer-state bytes).

Identity is never color-alone: each optimizer keeps one marker shape across both figures
(SGD o, Adam s, Lion v, AdaFactor D).
"""
from __future__ import annotations
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from paper_figs import PRETTY, RES, TEXTWIDTH, save          # also applies paper rcParams

# ieeeaccess.cls: \textwidth 177.53mm, \columnsep 6.95mm, so one column is 85.29mm = 3.36in.
# A figure's canvas has to match the width it is INCLUDED at, or every label is scaled by the
# ratio between them -- at 7.1in drawn and 3.36in included, that is 47% and the axes stop being
# readable. requirement_gap_paper is the one figure the paper places in a column, so it is the
# one drawn at COLWIDTH; the other two are drawn at TEXTWIDTH and included at \textwidth.
COLWIDTH = 3.36

# TERMINOLOGY (2026-09-03, and the project glossary is the canonical list).  The paper calls
# the held-out
# pre-drift rate selection a TUNED rate; the code calls it `rehearsed` (stage0_pool.rehearsed,
# the \ExtReh* macros, the `rehearsed` JSONL keys).  The rename stopped at the reader-facing
# strings on purpose: "rehearsal" means REPLAY in continual learning, which this paper cites
# and compares against, so the word had to leave the captions -- but renaming 112 macros and a
# canonical result key buys nothing a reader sees, and every such rename is a chance for a
# number to keep its name while changing its value.  Keep the two names in step through this
# comment, not by renaming.

# FAMILY PALETTE. Okabe & Ito's colour-blind-safe eight, the set most journals' figure
# guidelines point at and the one behind matplotlib's tableau-colorblind10 and seaborn's
# "colorblind". Adopted 2026-09-03 in place of an ad-hoc mix of matplotlib tab10 defaults
# (tab10 blue / orange / cyan / brown), which read as unharmonious in print.
#
# ONE CONSTRAINT DECIDED THE ASSIGNMENT: no family may take a colour that already means a
# DIFFERENT thing in lr_response_paper, where colour encodes the individual optimizer. That
# rules out #009E73 (AdaFactor there, but a member of the "reduced state" family here) and
# #d62728 (Adam). #E69F00 is kept for "reduced state" precisely because it is signSGD's colour
# there and signSGD is in that family, so the two figures agree rather than clash.
#
# check_palette.py verdict: normal 20.0 / deuteranopia 10.1 / protanopia 9.0 / tritanopia 9.8,
# PASS in all four views with no pair relying on the secondary encoding.
FAM = {"base": ("#0072B2", "standard baselines"),                   # Okabe-Ito blue
       "gp":   ("#E69F00", "general-purpose, reduced state"),       # orange
       "ns":   ("#CC79A7", "designed for non-stationary streams"),  # reddish purple
       "lrf":  ("#56B4E9", "learning-rate-free"),                   # sky blue
       "new":  ("#333333", "ObSign (proposed)")}                    # near-black, unchanged

# optimizer -> (key, label, colour, marker) for lr_response_paper.
#
# ARMS (changed 2026-08-31). The five arms are the ones the extension's argument runs through:
#   sgdm       1x    the paper's SGD-family reference (plain SGD was retired from the paper)
#   adam       2x    the other de facto reference
#   signsgd    0x    0 state, Adam-level quality when tuned -- and a CLIFF at the default
#   adafactor  0.54x the same quality with NO cliff at the default: the relative update clip
#   obsign_t1e3 0x   that mechanism transplanted onto the sign direction at zero state
# Lion is dropped from this figure (its curve is a near-duplicate of signSGD's, one state class
# up) and lives in the table instead.
#
# COLOURS. In THIS figure colour encodes the individual optimizer, not the family: there are
# only five curves and each needs its own identity. sgdm-blue and adam-red are fixed across
# every figure of the conference paper and are kept. This set PASSES check_palette in all four
# views; marker shape is a second, redundant encoding, and panel A's below-static table is
# keyed by row label.
SERIES = [("sgdm", "SGD+m", "#1f77b4", "P"),
          ("adam", "Adam", "#d62728", "s"),
          ("signsgd", "signSGD", "#E69F00", "^"),
          ("adafactor", "AdaFactor", "#009E73", "D"),
          ("obsign_t1e3", "ObSign $\\tau$=1e-3", "#333333", "d")]
CORE = {"appliances", "bdg2", "ETTm2", "ETTh2", "ETTm1", "ETTh1"}


def load_jsonl(path):
    rows = {}
    for line in open(os.path.join(RES, path)):
        try:
            r = json.loads(line)
            rows[(r["dataset"], r["backbone"], r["L"], r["H"], r["seed"])] = r
        except json.JSONDecodeError:
            pass
    return rows


def lr_response_paper():
    """C2 (the LR confound) re-run over the extension's optimizer set, on the 216 cells.

    Panel A is the mechanism: signSGD and the two references fall off a cliff above their own
    optimum, while AdaFactor and ObSign do not -- both clip the update relative to the
    parameter scale, so past the knee the rate cancels out of the step. Panel B is the
    consequence for deployment: how many cells sit below the static baseline at the shared
    default, and how many the tuning pass has to rescue.

    Every arm is read on SHARED_LR so no arm gets a wider search than the references (ObSign's
    grid runs two rates higher for the bracketing check; those are excluded here). sgdm/adam
    come from lr_fairness.jsonl, the rest from the stage files, merged by stage0_pool.

    NOTE (2026-08-31): AdaFactor's surface is one of the two the P0-1 fill-in extends, so this
    figure must be regenerated after that pass merges."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import stage0_pool as pool
    from stage0_optimizers import _restrict, sel_oracle

    cells, ref = pool.load_cells()
    keys = [k for k, _ in cells]
    rows = dict(cells)
    n = len(keys)
    # Panel A compares curves point for point, so it can only use rates EVERY plotted arm
    # actually holds. Right now that is 8 of the shared grid's 10: Stage 0 ran before the top
    # extension to 3e-2/1e-1, which is the P0-1 gap the fill-in pass closes -- after it merges
    # this intersection becomes the full 10 and the panel widens on its own, no edit needed.
    common = None
    for o, *_ in SERIES:
        have = set(pool._sweep(o, rows[keys[0]], ref[keys[0]]))
        common = have if common is None else (common & have)
    lrs = sorted(float(x) for x in common)
    if len(lrs) < len(_restrict(ref[keys[0]]["adam"])):
        print(f"  NOTE: panel A drawn on {len(lrs)} of the shared grid's "
              f"{len(_restrict(ref[keys[0]]['adam']))} rates -- "
              f"{sorted(set(_restrict(ref[keys[0]]['adam'])) - common, key=float)} "
              f"missing for at least one arm (P0-1)")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 3.0))
    # ---- Panel A: benefit vs rate, median + IQR ----
    neg_rows = []
    for o, lab, col, mk in SERIES:
        M = np.array([[pool._sweep(o, rows[k], ref[k])[f"{lr:g}"]["benefit"] for lr in lrs]
                      for k in keys])
        M[~np.isfinite(M)] = -1e9              # a diverged stream leaves the axis rather than
        med = np.median(M, axis=0)             # silently dropping out of the median
        q1, q3 = np.percentile(M, [25, 75], axis=0)
        axA.plot(lrs, med, mk + "-", color=col, label=lab, zorder=3, ms=3.0)
        axA.fill_between(lrs, q1, q3, color=col, alpha=0.12)
        neg_rows.append((lab, col, (M < 0).sum(axis=0)))
    # below-static counts as a mini-table: identity by row label, never by colour alone
    # The mini-table's row names need a left margin of their own: at the panel's old x limit
    # they started 0.1 decades from the first rate column and ran straight into its digits.
    # Widening the axis to the left costs nothing (no curve starts before the first rate) and
    # gives the names somewhere to live.
    y0, dy = -34, 9
    xlab = min(lrs) * 0.14
    axA.text(xlab, y0 + 8, "cells $<$ static:", fontsize=5.8, color="0.3", va="center")
    for i, (lab, col, neg) in enumerate(neg_rows):
        y = y0 - i * dy
        # First word only: the legend above carries the full name, and "ObSign $\tau$=1e-3"
        # spelled out here runs into the first rate column whatever margin it is given.
        axA.text(xlab, y, lab.replace("full-", "").split(" ")[0], fontsize=5.8, color=col,
                 va="center", ha="left")
        for x, c in zip(lrs, neg):
            # ZEROS ARE DRAWN, faintly. Skipping them left the proposed rule's row empty --
            # which is the row the paper is about, and an empty row reads as missing data
            # rather than as "no cell fell below the static model at any rate".
            axA.text(x, y, f"{c}", fontsize=5.8, color=col, ha="center", va="center",
                     alpha=1.0 if c else 0.42)
    axA.axhline(0, color="0.4", lw=0.8)
    axA.axvline(1e-3, color="0.25", ls="--", lw=1.0)
    axA.set_xscale("log")
    axA.set_xlim(min(lrs) * 0.115, max(lrs) * 1.5)   # tracks the rates actually drawn, so
                                                  # the panel widens when P0-1 merges
    axA.set_ylim(-80, 40)
    axA.set_xlabel("online learning rate")
    axA.set_ylabel("benefit %")
    # TITLES are descriptors, not claims. The claim ("signSGD and the references collapse
    # first, AdaFactor later, ObSign not at all") belongs in the caption and the body, where a
    # reader can weigh it; inside the panel it is text competing with the curves that support
    # it. Same for panel B.
    axA.set_title("(A) benefit vs online learning rate", fontsize=8)
    axA.grid(alpha=0.3, which="both")

    # ---- Panel B: the same cells at the default vs at the tuned rate ----
    for o, lab, col, mk in SERIES:
        xs, ys = [], []
        for k in keys:
            sw = pool._sweep(o, rows[k], ref[k])
            xs.append(sw["0.001"]["benefit"])
            ys.append(sel_oracle(sw)[1])
        axB.scatter(np.array(xs), np.array(ys), marker=mk, s=9, c=col, alpha=0.55,
                    edgecolors="none", zorder=3)
    lo, hi = -60, 62
    axB.plot([lo, hi], [lo, hi], color="0.6", lw=0.7, ls=":")
    axB.axhline(0, color="0.4", lw=0.8); axB.axvline(0, color="0.4", lw=0.8)
    # The "cells below static, at the default -> at the tuned rate" counts used to be a
    # text box here. They are per-arm scalars, so they belong in optimizer_table.tex, which
    # carries them for all sixteen arms instead of the five drawn here.
    axB.set_xlim(lo, hi); axB.set_ylim(-8, hi)
    axB.set_xlabel("benefit % at the shared default ($10^{-3}$)")
    axB.set_ylabel("benefit % at the tuned rate")
    axB.set_title("(B) benefit at the default vs at the tuned rate", fontsize=8)
    axB.grid(alpha=0.3)

    # ONE legend for both panels, above the figure, as in the BigData frontier figure. The two
    # in-axes legends it replaces cost real plot area: panel A's sat over the small-rate end of
    # every curve and panel B's over the dense y=x cluster, which is where the rules that need
    # no rescue actually live.
    handles = [Line2D([], [], marker=mk, color=col, markerfacecolor=col, markersize=4.5,
                      lw=1.2, label=lab) for _, lab, col, mk in SERIES]
    handles.append(Line2D([], [], ls=":", color="0.6", label="$y=x$ (B)"))
    fig.legend(handles=handles, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.035),
               frameon=False, fontsize=6.4, handletextpad=0.4, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "lr_response_paper")


def frontier_stage0_paper():
    """RETIRED 2026-08-31 -- do not call, and do not put its output in the extension paper.

    Its right-hand column is a per-update COMPUTE axis, and the extension does not make a
    compute claim. That axis was never trustworthy here anyway: it reads the `adapt_ms` stored
    in stage0_optimizers.jsonl, the sequential single-pass mean that ranks state-free plain SGD
    (4.34 ms) slower than 2x-state Adam (4.26 ms). Re-measuring it under the controlled
    protocol (P0-3) was tried and abandoned: per-update cost at batch 1 is kernel-launch bound,
    so it follows the HOST's CPU, and the two A100 machines available disagree by a factor that
    is not even monotone -- iris/melon is 0.57-0.96 for DLinear/head/calib and 2.13-2.15 for
    full PatchTST. A number that changes rank with the machine cannot carry a claim, so the
    extension drops the axis rather than report it with caveats.

    The left-hand (memory) column is superseded: optimizer_table / memory_benefit_paper
    cover the same axis over all 20 arms including ObSign, where this figure has only Lion and
    AdaFactor and predates Stage 0b/0c entirely. The generated PDFs were deleted so they cannot
    be picked up by mistake. Kept as source because the BigData C3 frontier, which DOES report
    compute, is built by frontier.py from melon measurements and is unaffected by any of this.
    """
    from frontier import COMBOS, DATASETS, adapt_mem_bytes, pareto
    data = json.load(open(os.path.join(RES, "frontier_data.json")))
    s0 = load_jsonl("stage0_optimizers.jsonl")
    NEW = [("lion", "PatchTST full·Lion", "#ff7f0e", "v"),
           ("adafactor", "PatchTST full·AdaFactor", "#9467bd", "D")]
    AF_DEF = "full·AdaFactor @default (untuned)"
    rows_by_ds = {}
    for name in DATASETS:
        rows = [dict(r) for r in data[name]]
        cell = s0[(name, "patchtst", 96, 24, 0)]
        for o, lab, col, mk in NEW:
            res = cell[f"res_{o}"]
            rows.append(dict(label=lab, params=res["n_adapt_params"], ms=res["adapt_ms"],
                             benefit=cell[f"sel_benefit_{o}"], lr=cell[f"sel_lr_{o}"],
                             mem_bytes=4 * res["n_adapt_params"] + res["opt_state_bytes"]))
            print(f"{name}: {lab} lr={cell[f'sel_lr_{o}']:g} benefit={rows[-1]['benefit']:+.1f}% "
                  f"mem={rows[-1]['mem_bytes']:.2e}B ms={res['adapt_ms']:.1f}")
        # AdaFactor's headline reading: its OWN default 1e-3, no tuning at all (open marker)
        res = cell["res_adafactor"]
        rows.append(dict(label=AF_DEF, params=res["n_adapt_params"], ms=res["adapt_ms"],
                         benefit=cell["adafactor"]["0.001"]["benefit"], lr=1e-3,
                         mem_bytes=4 * res["n_adapt_params"] + res["opt_state_bytes"]))
        print(f"{name}: {AF_DEF} benefit={rows[-1]['benefit']:+.1f}%")
        rows_by_ds[name] = rows
    style = {lab: (mk, col) for _, _, lab, mk, col in COMBOS}
    style.update({lab: (mk, col) for _, lab, col, mk in NEW})
    xmem = lambda row: row.get("mem_bytes", adapt_mem_bytes(row))

    fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH, 5.4))
    for r, name in enumerate(DATASETS):
        rows = rows_by_ds[name]
        for c, (xget, xlab, xname) in enumerate(
                [(xmem, "adaptation memory: grads + opt. state (B, $\\downarrow$ better)",
                  "memory"),
                 (lambda row: row["ms"], "per-update compute (ms, $\\downarrow$ better)",
                  "compute")]):
            ax = axes[r, c]
            for row in rows:
                if row["label"] == AF_DEF:
                    ax.scatter(xget(row), row["benefit"], marker="D", s=110,
                               facecolor="none", edgecolor="#9467bd", lw=1.2, zorder=4)
                    continue
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
    handles += [Line2D([0], [0], marker=mk, color="w", markerfacecolor=col,
                       markeredgecolor="k", markersize=6, label=lab)
                for _, lab, col, mk in NEW]
    handles.append(Line2D([0], [0], marker="D", color="w", markerfacecolor="white",
                          markeredgecolor="#9467bd", markeredgewidth=1.2, markersize=6,
                          label=AF_DEF))
    handles.append(Line2D([0], [0], ls="--", color="0.5", label="Pareto frontier (memory)"))
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06),
               frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    save(fig, "frontier_stage0_paper")


if __name__ == "__main__":                # the same set regen_stage0.sh drives
    optimizer_table()
    requirement_table()
    lr_response_paper()
    requirement_gap_paper()
    knee_paper()


# ---------------------------------------------------------------------------------------
# The crowded figures (requirement_gap_paper, memory_benefit_paper) carry ten to fourteen
# methods, which no reader can separate by hue, so there colour encodes the FAMILY and identity
# is carried by marker shape plus a direct text label on every point. Family names say what
# puts a method in the family, not when we ran it: "Stage 0/0b/0c" is our own chronology and
# carries nothing for a reader. They are mutually exclusive and read in one line:
# standard -> reduced state -> built for drift -> learning-rate-free -> ours.
# The two ObSign arms shown are tau = 3e-3 and 1e-3. The tau = 1e-2 arm (+14.02, mis1x 5.14)
# and RelSign, the no-guard ablation (+13.73, mis1x 12.33), are reported in the text rather
# than plotted: every ObSign arm sits at state = 0, so in panel A they would pile up on the
# same x with no added information.
# key -> (label, family, marker); order = drawing order
# arm -> tau.  ONE definition, imported by gen_macros_stage0 as well: tau appears in the
# figure, in the macros and in the section-V design rule, and three copies of "which arm is
# which tau" is how a plot ends up labelling a curve with the wrong guard level.
TAU_OF = {"obsign": 1e-2, "obsign_t5e3": 5e-3, "obsign_t3e3": 3e-3, "obsign_t2e3": 2e-3,
          "obsign_t1p5e3": 1.5e-3, "obsign_t1e3": 1e-3}


def tau_label(arm):
    """tau as maths, matching gen_macros_stage0.ARM_LABEL (a figure that says 1e-3 next to a
    table that says $10^{-3}$ makes a reader check whether they are the same run)."""
    t = TAU_OF[arm]
    e = int(np.floor(np.log10(t) + 1e-9))
    m = t / 10 ** e
    mant = "" if abs(m - 1) < 1e-6 else (f"{m:g}" + r"{\cdot}")
    return f"$\\tau{{=}}{mant}10^{{{e}}}$"


TRADEOFF = [
    ("sgd_default",   "SGD @$10^{-3}$",  "base", "o"),
    ("sgd",           "SGD",             "base", "o"),
    ("adam",          "Adam",            "base", "s"),
    ("sgdm",          "SGD+mom.",        "base", "P"),
    ("lion",          "Lion",            "gp",   "v"),
    ("adafactor",     "AdaFactor",       "gp",   "D"),
    ("signsgd",       "signSGD",         "gp",   "^"),
    ("obgd",          "ObGD",            "ns",   "*"),
    ("adaptive_obgd", "Ada-ObGD",        "ns",   "X"),
    ("dons",          "disc. ONS",       "ns",   "<"),
    ("upgd",          "UPGD",            "ns",   ">"),
    ("idbd",          "IDBD",            "ns",   "h"),
    ("autostep",      "Autostep",        "ns",   "p"),
    ("obsign_t3e3",   "ObSign " + tau_label("obsign_t3e3"), "new", "H"),
    ("obsign_t1e3",   "ObSign " + tau_label("obsign_t1e3"), "new", "d"),
]
# Table-only rows: both are ObSign ablations that answer "is the guard level right?" and "is
# the guard needed at all?". They sit at state 0 like every other ObSign arm, so plotting them
# adds points without adding an axis; the table has the room the figure does not.
ABLATIONS = [
    ("obsign",  "ObSign " + tau_label("obsign") + " (guard too loose)", "new", "8"),
    ("relsign", "RelSign (no guard)",                                  "new", "x"),
]


def _tau_sweep_rows(cells):
    """The Stage-0d tau arms, as table rows -- ONE list, derived from the data.

    These are margin-sweep points, not candidate defaults, so they belong in the table and not
    in the figures (six curves of the same shape hide the shape). Built from TAU_OF rather than
    written out, and filtered through pool.covered(), so that (a) an arm cannot appear in the
    table under a tau it was not run with, and (b) a half-finished run contributes nothing: the
    macro \ExtArms counts the same population this returns, and a 108-cell row next to a
    216-cell row looks like a number rather than like an unfinished job.
    """
    import stage0_pool as pool
    listed = {k for k, *_ in TRADEOFF + ABLATIONS}
    return [(a, f"ObSign {tau_label(a)} (margin sweep)", "new", "8")
            for a in sorted(pool.covered(list(TAU_OF), cells), key=lambda a: -TAU_OF[a])
            if a not in listed]
# Hand-placed label offsets (points). Several methods share an x (state is 0/1/2/3x) or a y
# (~12% benefit), so automatic placement collides -- these were set by rendering and looking.
LABEL_OFF = {                      # key: (panel A dx,dy), (panel B dx,dy)
    "signsgd":       ((9, -4), (4, 2)),
    "sgd":           ((6, 1), (-3, -12)),
    "sgd_default":   ((6, -10), None),
    "obgd":          ((8, -11), (8, -3)),
    "adafactor":     ((6, 2), (5, 2)),
    "lion":          ((6, 2), (4, 2)),
    "upgd":          ((6, 3), (-8, 7)),
    "sgdm":          ((6, -10), (-2, 7)),
    "dons":          ((6, -3), (0, -13)),
    "adaptive_obgd": ((8, -3), (8, -3)),
    "adam":          ((6, 2), (4, 2)),
    "idbd":          ((6, 1), (5, 1)),
    "autostep":      ((6, 2), (-4, 7)),
    "obsign_t3e3":   ((9, 5), (5, 3)),
    "obsign_t1e3":   ((9, -13), (5, -24)),
}


def _tradeoff_stats(spec=None):
    """Per-method (mean benefit + bootstrap CI, LR-miss cost, state/param, negative cells)
    over the 216 Stage-0b cells, plus the figures' presentation fields (label/family/marker).

    The pooled math itself lives in stage0_pool.rehearsed/fixed -- the SINGLE implementation
    the table, these figures and gen_macros_stage0.py all read, so a rerun can no longer move
    the table without moving the figure. `sgd_default` is the one row that is not a tuned
    reading: it is plain SGD at the fixed 1e-3 default, the recipe's no-tuning fallback, and
    comes from stage0_pool.fixed."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import stage0_pool as pool

    spec = spec or TRADEOFF
    cells, ref = pool.load_cells()
    arms = sorted({("sgd" if k == "sgd_default" else k) for k, *_ in spec})
    reh = pool.rehearsed(arms, cells, ref)
    fix = pool.fixed(["sgd"], 1e-3, cells, ref) if any(k == "sgd_default" for k, *_ in spec) else {}

    out = {}
    for key, label, fam, mk in spec:
        s = fix["sgd"] if key == "sgd_default" else reh.get(key)
        if s is None:
            continue
        out[key] = dict(key=key, label=label, fam=fam, mk=mk, mean=s["mean"], worst=s["worst"],
                        lo=s["lo"], hi=s["hi"], miss=s.get("mis1x", np.nan),
                        state=s["state"], mem=s.get("mem_bytes", np.nan),
                        neg=s["neg"], n=s["n"])
    return out


def optimizer_table():
    """The ONE table the extension's per-arm numbers live in.

    It carries all three requirements for all sixteen arms:
      R1  state (memory is 4*N*(1+state) bytes; N is fixed by the configuration, so the
          multiplier is the configuration-independent way to write it)
      R2  mis1x, and the benefit at the shared default with no tuning
      R3  negative cells under BOTH readings, and the worst non-diverged cell

    WHY A TABLE AND NOT A FIGURE. Two figures were tried on this material and both failed.
    A quality-vs-state scatter puts up to six methods on one vertical line, because state takes
    only five values -- separating them needs an x offset that means nothing. A quality-vs-
    memory scatter with the two readings joined by a segment (memory_benefit_paper, retired
    2026-09-03) fixed the ordering but not the reading: sixteen arms over five memory columns
    still overplot, and the reader has to decode a two-point glyph before any number arrives.
    The values here are discrete, per-arm and want to be compared digit by digit, which is what
    a table is for.

    Generated into results/tsf_edge/optimizer_table.tex for \input (never hand-typed, per
    CLAUDE.md) and echoed as markdown here."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import stage0_pool as pool

    cells, ref = pool.load_cells()
    spec = [t for t in TRADEOFF + ABLATIONS + _tau_sweep_rows(cells) if t[0] != "sgd_default"]
    S = _tradeoff_stats(spec)
    keys = [k for k, *_ in spec if k in S]
    FX = pool.fixed(keys, 1e-3, cells, ref)          # the untuned reading, same 216 cells

    classes = sorted({round(S[k]["state"], 2) for k in keys})
    # Pareto on the memory axis: minimise state, maximise quality
    pf, best = set(), -1e18
    for k in sorted(keys, key=lambda k: (S[k]["state"], -S[k]["mean"])):
        if S[k]["mean"] > best:
            pf.add(k); best = S[k]["mean"]
    top = max(keys, key=lambda k: S[k]["mean"])           # ONE bold number, per CLAUDE.md
    top_fx = max(keys, key=lambda k: FX[k]["mean"] if k in FX else -1e18)

    rows = [(c, sorted([k for k in keys if round(S[k]["state"], 2) == c],
                       key=lambda k: -S[k]["mean"])) for c in classes]

    mis = lambda s: ("---" if s["miss"] != s["miss"] else f"{s['miss']:.2f}")
    fx = lambda k, f: (f"{FX[k]['mean']:+.2f}" if k in FX else "---")
    fxn = lambda k: (str(FX[k]["neg"]) if k in FX else "---")
    plain = lambda t: (t.replace("$\\tau$", "tau").replace("$10^{-3}$", "1e-3")
                        .replace("$\\times$", "x"))
    print("\n| state | optimizer | tuned % (95% CI) | neg | worst % | mis1x | "
          "@1e-3 % | neg | Pareto |")
    print("|---|---|---|---|---|---|---|---|---|")
    for c, members in rows:
        for n, k in enumerate(members):
            s = S[k]
            print(f"| {f'{c:g}x' if n == 0 else ''} | {plain(s['label'])} | "
                  f"{s['mean']:+.2f} [{s['lo']:+.2f}, {s['hi']:+.2f}] | {s['neg']} | "
                  f"{s['worst']:+.1f} | {mis(s)} | {fx(k, FX)} | {fxn(k)} | "
                  f"{'yes' if k in pf else ''} |")

    n_ref = int(round(S[top]["mem"] / 4 / (1 + S[top]["state"]))) if S[top]["mem"] == S[top]["mem"] else 0
    tex = [r"% AUTO-GENERATED by stage0_figs.optimizer_table() -- do not edit.",
           r"\begin{table*}[t]", r"\centering", r"\footnotesize",
           r"\caption{All rules on the same " + f"{len(cells)}" + r" cells, under both "
           r"readings of Section~\ref{sec:reqs}. \emph{State}: optimizer state per trainable "
           r"parameter; adaptation memory is $4N(1+s)$ bytes with $N="
           + f"{n_ref:,}".replace(",", r"{,}") + r"$ (PatchTST, $L{=}96$, $H{=}24$). "
           r"\emph{Tuned}: rate selected on the held-out pre-drift slice. "
           r"\emph{At $10^{-3}$}: the same cells at the shared default with no selection. "
           r"\emph{neg}: cells below the frozen baseline, including diverged ones (benefit "
           r"$<-100\%$), which are excluded from means and \emph{worst}. \emph{mis1x}: "
           r"benefit lost to a one-decade rate error from each arm's own optimum. $\dagger$ "
           r"marks the state-axis Pareto frontier. A tuned value below its own untuned value is "
           r"not an error: selection lands below the test optimum roughly twice as often as "
           r"above it, and a rule whose high rates are harmless loses benefit by being tuned "
           r"down rather than shipped at the higher fixed rate "
           r"(Section~\ref{sec:tuningpass}).}",
           r"\label{tab:optimizers}",
           r"\begin{tabular}{llrrrrrr}", r"\toprule",
           r"& & \multicolumn{4}{c}{tuned rate} & \multicolumn{2}{c}{at $10^{-3}$, untuned} \\",
           r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}",
           r"state & optimizer & benefit \% (95\% CI) & neg & worst \% & mis1x & "
           r"benefit \% & neg \\"]
    for c, members in rows:
        tex.append(r"\midrule")
        for n, k in enumerate(members):
            s = S[k]
            val = f"{s['mean']:+.2f}"
            if k == top:
                val = r"\textbf{" + val + "}"
            v2 = fx(k, FX)
            if k == top_fx and v2 != "---":
                v2 = r"\textbf{" + v2 + "}"
            lab = s["label"] + (r"$^\dagger$" if k in pf else "")
            cls = ("$%g" % c) + r"\times$" if n == 0 else ""
            tex.append(f"{cls} & {lab} & {val} [{s['lo']:+.2f}, {s['hi']:+.2f}] & "
                       f"{s['neg']} & {s['worst']:+.1f} & {mis(s)} & {v2} & {fxn(k)} " + r"\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    out = os.path.join(RES, "optimizer_table.tex")
    open(out, "w").write("\n".join(tex))
    print(f"\nwrote {out}")


DEPLOY_TABLE_ARMS = [
    # (key, label) -- ONE representative per design class, plus the two arms the proposition
    # turns on. The full sixteen live in optimizer_table.tex; this table answers a different
    # question ("does the class survive a DIFFERENT shipped default?") and a sixteen-row grid
    # of ten rate columns is unreadable.
    ("obsign_t1e3", "ObSign " + tau_label("obsign_t1e3")), ("adafactor", "AdaFactor"),
    ("obsign_t3e3", "ObSign " + tau_label("obsign_t3e3")),
    ("relsign", r"RelSign (no guard)$^\ddagger$"),
    ("signsgd", "signSGD"), ("lion", "Lion"), ("adam", "Adam"),
    ("autostep", "Autostep"), ("upgd", "UPGD"), ("sgd", "SGD"), ("sgdm", "SGD+mom."),
    ("obgd", "ObGD"),
]
# The FULL shared grid, not a readable subset: the band column counts decades, so a reader
# who cannot count the same decades in the row is being asked to trust the number. This is
# the failure the macro post-mortem is about, one table earlier in the pipeline.
def _shared_lr_grid():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from online_eval import LR_GRID
    return list(LR_GRID)


DEPLOY_TABLE_LRS = _shared_lr_grid()


def requirement_table():
    r"""Section IV's evidence: the untuned reading at EVERY rate a deployment might ship.

    The section used to rest on one column (lr = 1e-3), and that column cannot carry it.

      * 1e-3 has no standing as "the" default -- Lion ships 1e-4 -- so judging a class there is
        the conference version's own C3 unfairness pointed the other way. A reviewer asks this
        first, and the honest answer is a sweep, not a defence of the column.
      * "AdaFactor satisfies R2 by accident, not by design" is the load-bearing sentence of the
        section, and at one column it is an assertion. Across the grid it is a measurement:
        AdaFactor is deployable over two decades and falls to +8.63% with 49 negative cells at
        1e-2, while ObSign's rate cancels out of the update above the knee and its row does not
        move for three decades.

    Deployable (marked *) is the TWO-PART R3 -- no cell below the static baseline AND a 95% CI
    that still overlaps the best tuned arm. The no-harm half alone is passed by ten arms
    including plain SGD, which would make this section's proposition false; see
    stage0_pool.requirement_check().
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import stage0_pool as pool

    cells, ref = pool.load_cells()
    keys = [k for k, _ in DEPLOY_TABLE_ARMS]
    D = pool.deployable(keys, cells, ref, grid=DEPLOY_TABLE_LRS)
    order = sorted([k for k in keys if k in D],
                   key=lambda k: (-D[k]["band"], D[k]["state"]))
    hdr = [f"$10^{{{int(round(np.log10(x)))}}}$" if abs(np.log10(x) - round(np.log10(x))) < 1e-9
           else f"$3{{\\cdot}}10^{{{int(np.floor(np.log10(x)))}}}$" for x in DEPLOY_TABLE_LRS]

    def cell(r, tex=True):
        # A rate where every surviving cell diverged has no mean to print; "div." is the
        # reading, and the negative count still carries how many cells it applies to.
        v = ("div./" + str(r["neg"]) if r["mean"] != r["mean"]
             else f"{r['mean']:+.1f}/{r['neg']}")
        if not tex:
            return v + ("*" if r["ok"] else "")
        # A marker and a shade, NOT bold: bold in this project's tables means "best
        # performance" and nothing else (CLAUDE.md), and bolding every deployable cell would
        # put a whole bold row under our own method -- the exact pattern that rule forbids.
        # The shade is what makes the BAND a band: a run of deployable rates is the one thing
        # this table exists to show, and a superscript dot per cell leaves the reader to
        # assemble it. Grey rather than a colour so it survives a black-and-white printout,
        # and the marker is kept as the redundant encoding.
        return (r"\cellcolor[gray]{0.88}" + v + r"$^{\bullet}$") if r["ok"] else v

    ref_lo, ref_hi = D[order[0]]["ref_lo"], D[order[0]]["ref_hi"]
    print("\n| arm | state | " + " | ".join(f"{x:g}" for x in DEPLOY_TABLE_LRS)
          + " | band | dec |")
    print("|---" * (len(DEPLOY_TABLE_LRS) + 4) + "|")
    for k in order:
        d = D[k]
        dec = "--" if d["band"] < 2 else f"{d['band_decades']:.1f}"
        print(f"| {k} | {d['state']:.2f}x | "
              + " | ".join(cell(r, tex=False) for r in d["rows"])
              + f" | {d['band']} | {dec} |")

    # The one count the caption still quotes -- the cell total -- is computed, not typed: this
    # file is generated, and a literal "216 cells" goes quietly stale the first time the grid
    # moves. The no-harm-only count and the decade arithmetic left this caption on 2026-09-04:
    # both are stated in section III-B, and a caption that re-derives a definition the text
    # already gives is a third pass over the same material.
    tex = [r"% AUTO-GENERATED by stage0_figs.requirement_table() -- do not edit.",
           r"\begin{table*}[t]", r"\centering", r"\scriptsize",
           # THREE things keep this table inside the text block, and it needs all three: it
           # carries every one of the ten rates because the band is only checkable if the
           # reader can count the same rates off the row, so the columns are not negotiable.
           #   * 2pt of column padding instead of LaTeX's 6pt (13 columns x 2 x 4pt = 104pt)
           #   * one band column ("7 (3.0)") rather than two
           #   * \resizebox with \width in the target, the no-package idiom for "shrink to
           #     the text width, but only if you are wider than it" -- a plain \resizebox
           #     would ENLARGE a table that already fits and leave its font bigger than the
           #     caption's.
           r"\setlength{\tabcolsep}{2pt}",
           # The caption says how to READ the table. It used to re-derive the definitions of
           # R3, of the band and of the paired comparison, all of which section III-B states
           # first -- three passes over the same material, and a caption a reader has to
           # finish before reaching a number. Definitions live in the text now, and this
           # points at them.
           r"\caption{Untuned adaptation benefit (\%) and cells below the frozen model, read "
           r"at each of the " + f"{len(DEPLOY_TABLE_LRS)}" + r" candidate shipped rates without "
           r"any tuning, over the same " + f"{len(cells)}" + r" cells. A "
           r"\colorbox[gray]{0.88}{shaded} entry marked $\bullet$ is \emph{deployable} under "
           r"R3; the band column reports the longest unbroken run of such entries per row as a "
           r"count, with its width in decades in parentheses (Section~\ref{sec:reqs}). Only "
           r"class representatives are shown; every remaining rule in "
           r"Table~\ref{tab:optimizers} has a band of one rate or none. $\ddagger$ RelSign "
           r"takes no learning rate; its row measures sensitivity to $\tau$, which is the "
           r"quantity the ablation varies.}",
           r"\label{tab:requirements}",
           # The box opens HERE, after the caption and the label: \caption is not allowed
           # inside an LR box, and scaling it with the table would leave the caption in a
           # different size from every other caption in the paper.
           r"\resizebox{\ifdim\width>\textwidth \textwidth\else\width\fi}{!}{%",
           r"\begin{tabular}{lr" + "r" * len(DEPLOY_TABLE_LRS) + "r}", r"\toprule",
           r"& & \multicolumn{" + str(len(DEPLOY_TABLE_LRS))
           + r"}{c}{shipped default learning rate} & \\",
           r"\cmidrule(lr){3-" + str(2 + len(DEPLOY_TABLE_LRS)) + "}",
           r"optimizer & state & " + " & ".join(hdr)
           + r" & band: rates (dec) \\", r"\midrule"]
    labels = dict(DEPLOY_TABLE_ARMS)
    for k in order:
        d = D[k]
        band = (str(d["band"]) if d["band"] < 2
                else f"{d['band']} ({d['band_decades']:.1f})")
        tex.append(f"{labels[k]} & ${d['state']:.2f}" + r"\times$ & "
                   + " & ".join(cell(r) for r in d["rows"]) + f" & {band}" + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""]
    out = os.path.join(RES, "requirement_table.tex")
    open(out, "w").write("\n".join(tex))
    print(f"\nwrote {out}")


def _spread_labels(ys, gap):
    """Minimum vertical separation for labels that share an x, order preserved, block centred
    on where the markers actually are. Points are NOT moved -- only their names."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    lo_i, hi_i = order[0], order[-1]
    shift = ((ys[lo_i] + ys[hi_i]) - (out[lo_i] + out[hi_i])) / 2
    return [y + shift for y in out]


# Arms that earn a POINT on the memory figure. Every arm in the study has a reason to be in
# the study, but a figure carries a claim rather than an inventory, and this one answers "at
# each adaptation-memory budget, what is the most benefit available, and where does ObSign
# sit?". An arm is drawn only if it changes what the reader can conclude about THAT:
#   (i)   a reference the deployment recipe names            -> sgd, sgdm, adam
#   (ii)  an endpoint of ObSign's tau dial, or the mechanism
#         ObSign was derived from                            -> signsgd, obgd, adafactor
#   (iii) the best arm in its state class (these define the
#         achievable frontier)                               -> obsign_t3e3 0x, adafactor 0.54x,
#                                                               lion 1x, adam 2x, autostep 3x
#   (iv)  ours                                               -> obsign_t3e3, obsign_t1e3
# Dropped by this rule: UPGD, disc. ONS, Ada-ObGD (1x, all below Lion) and IDBD (2x, below
# Adam). Each is dominated inside its own state class, so on this axis it adds a point without
# adding a conclusion. All four remain in optimizer_table.tex and in the robustness
# figure, where they do carry one (the non-stationary family's flat LR response).
MEMORY_FIG_ARMS = ["sgd", "sgdm", "adam", "lion", "adafactor", "signsgd", "obgd", "autostep",
                   "obsign_t3e3", "obsign_t1e3"]


# ---------------------------------------------------------------------------------------
# SECTION IV: the requirement space, EXISTING ARMS ONLY.
#
# The proposition this figure carries is ONE CROSS-SECTION of section IV's claim: at the rate a
# deployment would actually ship, how much benefit is reachable at each adaptation-memory
# budget, and how far short of it the zero-state column falls. That is R1 x R3(b).
#
# It is NOT "the region R1 & R2 & R3 is empty" (the wording this comment carried until
# 2026-09-03). Two reasons that phrasing had to go. R1 is a continuous axis, not a pass/fail
# test -- nothing in this paper shows a device that cannot afford 0.54x -- so "the region is
# empty" smuggles a threshold back in. And R2 is about not depending on WHICH rate ships,
# which a single-rate figure cannot show by construction; that is requirement_table()'s job,
# and this figure's caption should point at it. Three consequences for the design:
#
#   * THE READING IS THE UNTUNED ONE. R2 says no per-site rate tuning, so each rule is read
#     the way it would actually be deployed with no tuning pass: a tunable rule at the
#     shared default 1e-3, a learning-rate-free rule at its own default, which is its entire
#     pitch. The three figures in section VI use the tuned reading and therefore
#     cannot be reused here -- they already assume R2 has been paid for.
#   * NO ObSign, NO RelSign. The proposed method first appears in section V.
#   * ARMS ARE CLASS REPRESENTATIVES. Ada-ObGD folds into ObGD and IDBD into Autostep (same
#     mechanism class, and below it). At the shared default IDBD is within 0.004 pt of plain
#     SGD in every cell: its meta-learned step sizes barely move over a stream this short,
#     which is the honest reason it adds nothing here. That degeneracy is understood and was
#     recorded when the arm was added: meta-learned step sizes driven by g^2 shrink as the
#     model fits, so the meta update stalls at |g| ~ 1e-3; Autostep avoids it by using the true
#     diagonal Gauss-Newton curvature.
#
# PLAIN SGD IS DRAWN, and labelled as the 0-state reference point. The conference paper
# retired it (sgdm-migration), but the whole claim of this figure is about what is reachable
# at zero optimizer state, and at the untuned default plain SGD is the best existing arm there
# (+10.50, zero negative cells). Leaving it out would inflate the gap ObSign closes from
# 3.5 pt to 22 pt against signSGD, which would be dishonest.
REQ_SPEC = [  # key, label, family, marker, reading
    ("sgd",         "SGD",        "base", "o", "fixed"),
    ("sgdm",        "SGD+mom.",   "base", "P", "fixed"),
    ("adam",        "Adam",       "base", "s", "fixed"),
    ("lion",        "Lion",       "gp",   "v", "fixed"),
    ("adafactor",   "AdaFactor",  "gp",   "D", "fixed"),
    ("signsgd",     "signSGD",    "gp",   "^", "fixed"),
    ("obgd",        "ObGD",       "ns",   "*", "fixed"),
    ("upgd",        "UPGD",       "ns",   ">", "fixed"),
    ("dons",        "disc. ONS",  "ns",   "<", "fixed"),
    ("autostep",    "Autostep",   "ns",   "p", "fixed"),
    ("prodigy",     "Prodigy",    "lrf",  "X", "lrfree"),
    ("dog",         "DoG",        "lrf",  "8", "lrfree"),
    ("dadapt_sgd",  "D-Adapt SGD",  "lrf", "h", "lrfree"),
    ("dadapt_adam", "D-Adapt Adam", "lrf", "H", "lrfree"),
]


def _req_stats():
    import stage0_pool as P
    cells, ref = P.load_cells()
    tun = [k for k, _, _, _, rd in REQ_SPEC if rd == "fixed"]
    lrf = [k for k, _, _, _, rd in REQ_SPEC if rd == "lrfree"]
    A = P.fixed(tun, 1e-3, cells, ref)
    B = P.lrfree(lrf, cells, ref)
    out = {}
    for k, lab, fam, mk, rd in REQ_SPEC:
        s = dict((A if rd == "fixed" else B).get(k, {}))
        if not s:
            continue
        s.update(label=lab, fam=fam, mk=mk, reading=rd,
                 mem=P._measured_mem(k, cells))
        out[k] = s
    return out


def requirement_gap_paper():
    """Section IV: what an untuned deployment gets, against what it costs in memory.

    BROKEN Y AXIS, on purpose. Ten of the fourteen arms live inside a 7 pt band while Lion
    sits at -20, so a single axis spends two thirds of its height on four points and squeezes
    the comparison that carries the claim. The panel is split -- the arms that stay useful
    without tuning above, the ones that fall through the static baseline below -- with the
    break drawn. Both panels share the x axis and the scale of neither is distorted.
    """
    S = _req_stats()
    keys = [k for k, *_ in REQ_SPEC if k in S]

    zero_x = S["sgd"]["mem"]                       # the 0-state column
    best_zero = max(S[k]["mean"] for k in keys if round(S[k]["mem"]) == round(zero_x))
    best_any = max(S[k]["mean"] for k in keys)     # AdaFactor, at 0.54x state
    xlo, xhi = 1.9e5, 3.1e6
    CUT = 4.0                                      # panel split, chosen between the two groups

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COLWIDTH, 4.6), sharex=True,
                                   gridspec_kw=dict(height_ratios=[2.5, 1], hspace=0.07))
    ax1.set_ylim(4.6, 16.9)
    ax2.set_ylim(-23.5, 3.2)

    # The gap the section is about, stated as a measured quantity rather than a threshold we
    # chose: inside the 0-state column nothing existing reaches what 0.54x of state buys.
    ax1.axhspan(best_zero, best_any, xmin=0,
                xmax=(np.log10(zero_x * 1.06) - np.log10(xlo)) / (np.log10(xhi) - np.log10(xlo)),
                color="#d62728", alpha=0.13, zorder=0)
    ax1.axhline(best_any, ls="--", color="0.55", lw=1.0, zorder=1)
    ax1.annotate("", xy=(2.30e5, best_any), xytext=(2.30e5, best_zero),
                 arrowprops=dict(arrowstyle="<->", color="#a01f1f", lw=1.0, shrinkA=0,
                                 shrinkB=0), zorder=7)
    ax1.text(2.42e5, (best_zero + best_any) / 2, f"{best_any - best_zero:.1f} pt",
             fontsize=5.4, color="#a01f1f", ha="left", va="center", zorder=7)
    ax1.text(xhi * 0.97, best_any + 0.35, "best untuned benefit at any budget",
             fontsize=5.0, color="0.4", ha="right", va="bottom")
    # THE ZERO LINE. The panel is split, so the reader meets the upper panel first and can take
    # its floor for the baseline; every point in the lower panel is a rule that made the
    # forecast WORSE than not adapting, and that has to be legible without reading the caption.
    # Hence: the zero line is drawn heavier than the grid, named with its value, and everything
    # under it is shaded.
    ax2.axhspan(-23.5, 0, color="#d62728", alpha=0.07, zorder=0)
    ax2.axhline(0, color="0.15", lw=1.4, zorder=3)
    # BELOW the line, not above it: the axis break is drawn as a white bar across the top of
    # this panel, and a label sitting just above zero is half-eaten by it.
    ax2.text(xlo * 1.04, -1.4, "static model (0%)", fontsize=5.4, color="0.2", ha="left",
             va="top")
    # bottom RIGHT: the left of this panel is where signSGD and Lion sit, and the note ran
    # straight through Lion's name at column width.
    ax2.text(xhi * 0.97, -22.6, "below the line: adapting is worse\nthan leaving the model alone",
             fontsize=5.2, color="#a01f1f", ha="right", va="bottom")

    for k in keys:
        s_ = S[k]
        ax = ax1 if s_["mean"] >= CUT else ax2
        col = FAM[s_["fam"]][0]
        ax.errorbar(s_["mem"], s_["mean"],
                    yerr=[[s_["mean"] - s_["lo"]], [s_["hi"] - s_["mean"]]], fmt="none",
                    ecolor=col, elinewidth=0.8, capsize=1.6, alpha=0.5, zorder=2)
        ax.scatter(s_["mem"], s_["mean"], marker=s_["mk"], s=115 if s_["mk"] == "*" else 66,
                   color=col, edgecolor="k", lw=0.7, zorder=4)

    # Labels carry the negative-cell count, because R3 is a binary property the y position does
    # not show: an arm can average well and still have put cells below the static baseline.
    # Six columns inside 1.2 decades, so each column SPLITS its names left and right by height
    # and spreads each side on its own. Points are never moved.
    lab_of = {k: f"{S[k]['label']}  ({S[k]['neg']})" for k in keys}
    for ax, span in ((ax1, 12.3), (ax2, 26.7)):
        cols = {}
        for k in keys:
            if (S[k]["mean"] >= CUT) == (ax is ax1):
                cols.setdefault(round(S[k]["mem"]), []).append(k)
        for x, members in cols.items():
            members = sorted(members, key=lambda k: S[k]["mean"])
            for side, mem in (("L", members[0::2]), ("R", members[1::2])):
                if not mem:
                    continue
                ys = _spread_labels([S[k]["mean"] for k in mem], 0.072 * span)
                left = side == "L"
                for k, ly in zip(mem, ys):
                    ax.annotate(lab_of[k], xy=(x, S[k]["mean"]),
                                xytext=(x * (0.80 if left else 1.22), ly), fontsize=5.0,
                                color="0.12", ha="right" if left else "left", va="center",
                                arrowprops=dict(arrowstyle="-", lw=0.5, color="0.6", shrinkA=0,
                                                shrinkB=3), zorder=6,
                                bbox=dict(boxstyle="round,pad=0.16", fc=ax.get_facecolor(),
                                          ec="none", alpha=0.9))

    # the break
    ax1.spines["bottom"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.tick_params(bottom=False)
    for ax, y in ((ax1, 0.0), (ax2, 1.0)):
        ax.plot([0, 1], [y, y], transform=ax.transAxes, clip_on=False, color="w", lw=2.4,
                zorder=8)
        for xx in (0, 1):
            ax.plot([xx - 0.006, xx + 0.006], [y - 0.022, y + 0.022], transform=ax.transAxes,
                    clip_on=False, color="0.2", lw=0.9, zorder=9)

    ax2.set_xscale("log")
    ax2.set_xlim(xlo, xhi)
    ticks = sorted({round(S[k]["mem"]) for k in keys})
    ax2.set_xticks(ticks, minor=False)
    # six columns inside 1.2 decades on a 3.36in canvas: horizontal tick labels collide, so
    # they are rotated rather than thinned (each one is a memory class the text refers to).
    ax2.set_xticklabels([(f"{t / 1e6:.2f}M" if t >= 1e6 else f"{t / 1e3:.0f}k") for t in ticks],
                        fontsize=5.4, rotation=40, ha="right")
    ax2.set_xticks([], minor=True)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3, which="both")
    ax2.set_xlabel("adaptation memory: grads $+$ opt. state\n"
                   "(B, log scale, $\\downarrow$ better; PatchTST, $L$=96, $H$=24)",
                   fontsize=6.0)
    fig.text(0.005, 0.56, "adaptation benefit % with NO rate tuning\n"
             "(linear scale; mean of 216 cells, 95% CI)", rotation=90, va="center", ha="left",
             fontsize=6.2)
    fams = [f for f in FAM if any(S[k]["fam"] == f for k in keys)]
    handles = [Line2D([], [], marker="o", color="w", markerfacecolor=FAM[f][0],
                      markeredgecolor="k", markersize=5, label=FAM[f][1]) for f in fams]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.035),
               frameon=False, fontsize=5.4, handletextpad=0.4, columnspacing=1.0)
    # The reading note goes UNDER THE LEGEND, not under the x label: with bbox_inches="tight"
    # The reading convention used to be a fig.text banner across the top.  At final size it
    # rendered as one 6.2pt line of running prose above the axes -- unreadable, and duplicating
    # what the caption has to say anyway.  It lives in the LaTeX caption now; the figure keeps
    # only the one thing a caption cannot carry, which is the "( ) = cells below static" key.
    fig.text(0.5, 0.905, "( ) after each name: cells below the static model",
             ha="center", fontsize=5.4, color="0.35")
    fig.tight_layout(rect=(0.045, 0, 1, 0.885))
    save(fig, "requirement_gap_paper")


# ---------------------------------------------------------------------------------------
# SECTION V: what the guard does, and why the knee has to be PLACED.
#
# Panel A is a schematic of alpha(lr) = min(lr, tau*RMS(p)); panel B is the measurement that
# turns it into a claim. They belong on one figure because the whole argument of section V is
# that a position in panel A predicts a number in panel B.
#
# RMS_REF is a real measurement, not a drawing convenience: the median RMS over trainable
# tensors at initialisation is 0.053 (compact PatchTST, L96/H24 and L192/H96) and 0.059
# (DLinear L96/H24). This module deliberately does not import torch, so the value is quoted
# here with the command that reproduces it:
#     .venv/bin/python -c "import sys;sys.path.insert(0,'experiments/tsf_edge');
#     from online_eval import build_model;import numpy as np,torch;
#     m=build_model('patchtst',96,24,7);
#     print(np.median([float(p.pow(2).mean().sqrt()) for p in m.parameters() if p.numel()>1]))"
RMS_REF = 0.053
DEPLOYED_LR = 1e-3
# Panel A stays at three curves: it is a SCHEMATIC of alpha(lr), and six lines of the same
# shape would hide the shape.  Panel B is the measurement and takes every tau that ran.
KNEE_TAUS = [(1e-2,  "obsign",      tau_label("obsign"),      "#E69F00"),
             (3e-3,  "obsign_t3e3", tau_label("obsign_t3e3"), "#009E73"),
             (1e-3,  "obsign_t1e3", tau_label("obsign_t1e3"), "#333333")]


def knee_paper():
    """Section V: the guard's knee, and the price of putting it in the wrong place."""
    import stage0_pool as P
    cells, ref = P.load_cells()
    # EVERY tau that ran, not just panel A's three: panel B plots the whole margin sweep, and
    # reading it off a dict built from panel A's list is how an arm added to the study turns
    # into a KeyError at figure time (it did, the first run after Stage 0d landed).
    arms = P.covered(list(TAU_OF), cells) + ["signsgd"]
    F = P.fixed(arms, DEPLOYED_LR, cells, ref)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.9))

    # ---- A: the mechanism ---------------------------------------------------------------
    lrs = np.logspace(-6, -1, 400)
    axA.plot(lrs, lrs, ls="--", color="0.45", lw=1.2, label="signSGD: $\\alpha=$ lr")
    for tau, _, lab, col in KNEE_TAUS:
        cap = tau * RMS_REF
        axA.plot(lrs, np.minimum(lrs, cap), color=col, lw=1.6, label=f"ObSign {lab}")
        # The knee is the whole point of the panel, so it is a ringed marker rather than a dot
        # the eye has to find on a line, and it is named "knee" once per curve.
        axA.plot([cap], [cap], marker="o", ms=6.0, mfc="w", mec=col, mew=1.6, zorder=6)
        axA.annotate(f"knee {cap:.0e}".replace("e-0", "e-"), xy=(cap, cap),
                     xytext=(cap * 0.55, cap * 2.6), fontsize=6.2, color=col, ha="right")

    # The two REGIMES, named on the plot. The identity that carries section V -- above the knee
    # the rate has left the update -- is visible here as a flat line, and a flat line is easy to
    # read as "the plot stopped" unless it is labelled. Drawn on the recommended arm.
    _cap = 1e-3 * RMS_REF
    axA.annotate("", xy=(6e-2, _cap), xytext=(_cap * 1.7, _cap),
                 arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0,
                                 shrinkA=0, shrinkB=0), zorder=7)
    axA.text(6.0e-3, _cap * 0.62, "above the knee: lr cancels,\n$\\alpha$ fixed at "
             "$\\tau\\,$RMS$(\\mathbf{p})$", fontsize=6.4, color="#333333", ha="center", va="top")
    # Not rotated along the line: the up-right diagonal is where the knee labels live, so the
    # name goes in the empty wedge under the plateau and points at the segment it names.
    axA.annotate("below the knee:\nObSign $=$ signSGD", xy=(2.2e-5, 2.2e-5),
                 xytext=(9.3e-4, 3.2e-6), fontsize=6.4, color="0.35", ha="right", va="bottom",
                 arrowprops=dict(arrowstyle="-", lw=0.5, color="0.6", shrinkA=2, shrinkB=2))
    axA.axvline(DEPLOYED_LR, color="#d62728", lw=1.0, ls=":", zorder=1)
    axA.text(DEPLOYED_LR * 1.15, 1.3e-6, "rate the deployment\ncannot tune ($10^{-3}$)",
             fontsize=6.2, color="#a01f1f", ha="left", va="bottom")
    axA.set_xscale("log"); axA.set_yscale("log")
    axA.set_xlim(1e-6, 1e-1); axA.set_ylim(1e-6, 1e-1)
    axA.grid(alpha=0.3, which="both")
    axA.set_xlabel("online learning rate")
    axA.set_ylabel("effective step $\\alpha$ per parameter")
    axA.set_title("(A) the effective step $\\alpha=\\min($lr$,\\ \\tau\\,$RMS$(\\mathbf{p}))$\n"
                  "has two regimes, and the knee is where they meet", fontsize=7.4)
    axA.legend(loc="upper left", fontsize=6.2, frameon=False, handlelength=1.8)

    # ---- B: the measurement -------------------------------------------------------------
    # EVERY tau that ran, ordered by knee position, with deployability as the fill: a filled
    # marker satisfies the two-part R3 at the shipped rate, an open one does not. Colour is not
    # doing that work -- all of these are the same rule at different guard levels, so they keep
    # ObSign's colour and the fill is the encoding (the palette rule: never colour alone).
    # THE X AXIS IS THE MARGIN, not tau and not the knee position. The design rule section V
    # states is a rule about a distance -- keep the knee at least one decade under the rate you
    # ship -- so the pass/fail boundary is a vertical line at 1.0 on this axis, and the reader
    # can see which side each tau falls on. On a knee-position axis the same points crowd into
    # one decade and the rule has to be described in words instead of drawn.
    import stage0_pool as _P
    order = sorted(_P.covered(list(TAU_OF), cells), key=lambda a: TAU_OF[a])
    dep = _P.deployable(order, cells, ref, grid=[DEPLOYED_LR])
    ok_of = {a: dep[a]["rows"][0]["ok"] for a in order}
    margin = lambda a: float(np.log10(DEPLOYED_LR / (TAU_OF[a] * RMS_REF)))
    xs = [margin(a) for a in order]
    ys = [F[a]["mean"] for a in order]
    XLO, XHI = 1.45, 0.02                       # decades of margin, DECREASING to the right
    XSEP, XINF = 0.22, 0.12                     # the strip where the guard is gone entirely

    # The crossing is MEASURED, not asserted: the largest margin that still puts a cell below
    # the frozen model, and the smallest that does not. The strip between them is what the
    # sweep actually establishes, and drawing it as a strip rather than as a line is the
    # honest form -- a single line at a round number is what Stage 0d falsified.
    ok_m = [margin(a) for a in order if ok_of[a]]
    bad_m = [margin(a) for a in order if not ok_of[a]]
    if ok_m and bad_m:
        lo_ok, hi_bad = min(ok_m), max(bad_m)
        axB.axvspan(XLO, lo_ok, color="#0072B2", alpha=0.09, zorder=0)
        axB.axvspan(lo_ok, hi_bad, color="#d62728", alpha=0.07, zorder=0)
        axB.text(XLO - 0.03, -13.8, "no cell below\nthe frozen model", fontsize=6.4,
                 color="#1b5e86", ha="left", va="bottom")
        axB.annotate(f"crossing\nmeasured to\n{lo_ok - hi_bad:.2f} dec",
                     xy=((lo_ok + hi_bad) / 2, -6.5), fontsize=6.2, color="#a01f1f",
                     ha="center", va="center")
        axB.axvline(lo_ok, color="#0072B2", lw=1.0, ls="--", zorder=1)
    axB.axvspan(XSEP, XHI, color="0.85", alpha=0.55, zorder=0)
    axB.text(XINF, 20.6, "guard\nremoved", fontsize=6.2, color="0.35", ha="center", va="top")
    axB.plot(xs, ys, "-", color="0.6", lw=1.0, zorder=1)
    for i, (a, x, y) in enumerate(zip(order, xs, ys)):
        axB.scatter(x, y, s=70, color="#333333" if ok_of[a] else "w", edgecolor="#333333",
                    lw=1.1, zorder=4)
        # Six points, and the three largest margins sit within 0.3 decades of each other at
        # almost the same height, so same-side names touch. Alternate the side instead.
        up = i % 2 == 1
        # a white bed under the name: the leftmost point sits exactly on the dashed crossing
        # line, so its label was struck through by it.
        axB.annotate(f"{tau_label(a)}\n({F[a]['neg']})", xy=(x, y),
                     xytext=(0, 9 if up else -11), textcoords="offset points", fontsize=6.2,
                     ha="center", va="bottom" if up else "top", color="0.12",
                     bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    axB.scatter(XINF, F["signsgd"]["mean"], marker="^", s=70, color="#E69F00", edgecolor="k",
                lw=0.7, zorder=4)
    axB.annotate(f"signSGD\nno knee  ({F['signsgd']['neg']})", xy=(XINF, F["signsgd"]["mean"]),
                 xytext=(0, 10), textcoords="offset points", fontsize=6.4, ha="center",
                 va="bottom", color="0.12")
    axB.axhline(0, color="0.15", lw=1.4, zorder=3)
    axB.text(XLO - 0.03, 0.6, "static model (0%)", fontsize=6.2, color="0.2", ha="left",
             va="bottom")
    axB.set_xlim(XLO, XHI); axB.set_ylim(-15, 21.5)
    axB.grid(alpha=0.3, which="both")
    axB.set_xlabel("margin: decades the knee sits below the\n"
                   "shipped rate   ($\\rightarrow$ less guard)")
    axB.set_ylabel("adaptation benefit % at lr $=10^{-3}$\n(mean of 216 cells)")
    axB.set_title("(B) filled $=$ deployable at the shipped rate;\n"
                  "( ) $=$ cells below static", fontsize=7.4)

    fig.tight_layout()
    save(fig, "knee_paper")
