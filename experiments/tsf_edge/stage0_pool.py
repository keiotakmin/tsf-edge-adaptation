"""Pooled 216-cell aggregation for the extension study -- ONE implementation, three consumers.

Until now the extension's headline numbers came from three places: `stage0_figs._tradeoff_stats`
(the rehearsal-selected table and the trade-off figures), `stage0_optimizers.summarize` (per
(L,H) slice only, never pooled), and a throwaway script used to re-verify EXTENSION_DIFF.md on
2026-08-15. Three implementations of the same statistic is exactly how a paper ends up with a
figure and a table that disagree (see the macro-rename post-mortem), so the pooled math lives
here and everything else calls it.

Three readings, all on the 216 cells (6 datasets x 2 backbones x 6 (L,H) x seeds 0-2):

  rehearsed()  Table A -- each arm's online rate is TUNED on the held-out pre-drift slice,
               the deployable protocol the paper argues for. Every LR-derived quantity is
               computed on SHARED_LR so a contender with extra top rates gets no wider search
               than the sgd/adam references (the C3 unfairness in reverse).
  fixed()      Table B -- one shared rate, no rehearsal at all. This is the honest reading for
               a device that cannot afford a rehearsal pass, and it is where AdaFactor and
               ObSign separate from Adam/Lion/signSGD.
  lrfree()     Table C -- the learning-rate-free rules at their own default, which is their
               entire pitch: they are stored as a single reading, not a sweep.

Conventions, shared by all three and matching the paper:
  * a cell with benefit < -100% is DIVERGED: excluded from mean/median/worst, still counted
    in `neg` (dropping it there would hide the failure the column exists to report)
  * `neg` counts benefit < -0.05, so float noise around zero is not a negative cell
  * state multiplier is MEASURED (opt_state_bytes / 4 / n_adapt_params, PatchTST cells), not
    the declared STATE_MULT, because some implementations park state on param_groups
  * the bootstrap CI reseeds default_rng(0) FOR EACH ARM. The previous implementation drew
    every arm from one stream in spec order, so adding an arm to the figure's list silently
    moved the published CI of every arm after it; the point estimates never moved, which is
    exactly what makes that class of drift hard to notice. Reseeding per arm cost <=0.2 pp on
    the CI endpoints when the table was regenerated (2026-08-31) and nothing else.

Self-check:  .venv/bin/python experiments/tsf_edge/stage0_pool.py
prints all three tables as markdown.
"""
from __future__ import annotations
import os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stage0_optimizers import (load_jsonl, OUT, OUT_B, OUT_C, OUT_D, LRFAIR, STATE_MULT,
                               SHARED_LR, _restrict, sel_oracle, lr_miss_cost)

# TERMINOLOGY (2026-09-03, see ieee_access/GLOSSARY.md).  The paper says TUNED rate; this
# module's function is still called rehearsed(), and the JSONL keys and \ExtReh* macros still
# say "reh".  "Rehearsal" means REPLAY in continual learning -- a field this paper imports from
# and compares against (NatSR ships a replay buffer) -- so the word had to leave the captions.
# It did NOT have to leave the code: renaming a canonical result key is how a number keeps its
# name while changing its value.  The two vocabularies are bridged here, not by renaming.
CORE = {"appliances", "bdg2", "ETTm2", "ETTh2", "ETTm1", "ETTh1"}
REF_ARMS = ("sgd", "adam", "sgdm")          # these live in lr_fairness.jsonl, not the stage files
LRFREE_ARMS = ("prodigy", "dog", "dadapt_adam", "dadapt_sgd")
DIVERGED = -100.0


SEED_EXT = os.path.join(os.path.dirname(OUT), "stage0_seeds34.jsonl")
HEADLINE = ("adam", "sgdm", "lion", "adafactor", "signsgd",
            "obsign", "obsign_t3e3", "obsign_t1e3", "relsign")


def load_cells(seed_ext=False):
    """[(key, merged row)] for the cells all three stage files and lr_fairness share.

    seed_ext=True also folds in stage0_seeds34.jsonl, the seeds-3/4 extension of the HEADLINE
    arms. Those cells carry only those arms, so an arm that is not in them simply contributes
    fewer cells -- every statistic here already reports its own n, and seed_robustness() below
    is the intended way to read the difference. The screen's 216 cells are never rewritten.
    """
    rows = dict(load_jsonl(OUT))
    for src in (OUT_B, OUT_C, OUT_D):
        for k, r in load_jsonl(src).items():
            rows.setdefault(k, {}).update(r)
    if seed_ext and os.path.exists(SEED_EXT):
        for k, r in load_jsonl(SEED_EXT).items():
            rows.setdefault(k, {}).update(r)
    ref = load_jsonl(LRFAIR)
    cells = [(k, r) for k, r in sorted(rows.items()) if k in ref and k[0] in CORE]
    return cells, ref


def covered(arms, cells, frac=1.0):
    """The arms that have a reading in EVERY cell (or in `frac` of them).

    Stage 0d writes its file WHILE it runs, and every table in this paper is a pooled mean over
    the same cell population -- so an arm that is half finished must not enter one. It would
    not be flagged either: the pooled statistic reports its own n and a 108-cell mean next to a
    216-cell mean looks like a number, not like a bug. Every consumer that reads an arm added
    after the original screen filters it through here.
    """
    n = frac * len(cells)
    return [a for a in arms
            if sum(1 for _, r in cells if isinstance(r.get(a), dict) and r[a]) >= n]


def _sweep(arm, row, ref_row):
    """The arm's LR surface for this cell, restricted to the shared grid. The sgd/adam/sgdm
    references are read from lr_fairness.jsonl -- the stage files hold at most the two-point
    top extension for them, which is a bracketing check and not a search."""
    src = ref_row if arm in REF_ARMS else row
    sw = src.get(arm)
    return _restrict(sw) if isinstance(sw, dict) and sw else {}


# The configuration the BYTE axis is quoted at. Adaptation memory is 4 B/param of gradient
# plus the optimizer's own state, so it scales with the trainable-parameter count -- which
# here runs from 4,656 (DLinear L96/H24) to 210,964 (PatchTST L192/H96). A byte figure pooled
# over all 216 cells would therefore describe no deployment that exists. frontier_paper solves
# this by quoting the axis at one configuration; this module does the same, and names it.
# Only the SCALE depends on the choice: memory is 4*N*(1 + state) with N shared by every arm,
# so the arms' order and their spacing in decades are the same at any configuration.
REF_CONFIG = ("patchtst", 96, 24)


def _measured_mem(arm, cells, config=REF_CONFIG):
    """Adaptation-state memory in BYTES (gradients + optimizer state) at REF_CONFIG, measured
    rather than derived from the declared multiplier -- the same quantity, and the same
    preference for the measured value, as frontier.adapt_mem_bytes."""
    bb, L, H = config
    got = [4 * r[f"res_{arm}"]["n_adapt_params"] + r[f"res_{arm}"]["opt_state_bytes"]
           for _, r in cells
           if f"res_{arm}" in r and (r["backbone"], r["L"], r["H"]) == (bb, L, H)]
    return float(np.median(got)) if got else float("nan")


def _measured_state(arm, cells):
    st = [r[f"res_{arm}"]["opt_state_bytes"] / max(r[f"res_{arm}"]["n_adapt_params"], 1) / 4
          for _, r in cells if f"res_{arm}" in r and r["backbone"] == "patchtst"]
    return float(np.mean(st)) if st else float(STATE_MULT.get(arm, float("nan")))


def _stats(vals, extra=None):
    v = np.array([x for x in vals if x == x], float)         # drop NaN (a diverged stream)
    fin = v[v >= DIVERGED]
    d = dict(n=int(fin.size), n_all=int(v.size), mean=float(fin.mean()) if fin.size else np.nan,
             median=float(np.median(fin)) if fin.size else np.nan,
             worst=float(fin.min()) if fin.size else np.nan,
             neg=int((v < -0.05).sum()), diverged=int((v < DIVERGED).sum()))
    if fin.size:
        rng = np.random.default_rng(0)
        boot = np.array([rng.choice(fin, fin.size).mean() for _ in range(2000)])
        d["lo"], d["hi"] = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    else:
        d["lo"] = d["hi"] = np.nan
    d.update(extra or {})
    return d


def rehearsed(arms, cells=None, ref=None):
    """Table A: per-arm stats with the online rate selected on the validation slice."""
    if cells is None:
        cells, ref = load_cells()
    out = {}
    for arm in arms:
        vals, miss = [], []
        for k, r in cells:
            sw = _sweep(arm, r, ref[k])
            if not sw:
                continue
            vals.append(sel_oracle(sw)[1])
            miss.append(lr_miss_cost(sw))
        if not vals:
            continue
        out[arm] = _stats(vals, dict(
            arm=arm, state=_measured_state(arm, cells), mem_bytes=_measured_mem(arm, cells),
            mis1x=float(np.median([m for m in miss if m == m])) if miss else np.nan))
    return out


def fixed(arms, lr=1e-3, cells=None, ref=None):
    """Table B: per-arm stats at ONE shared rate, no rehearsal."""
    if cells is None:
        cells, ref = load_cells()
    key = f"{lr:g}"
    out = {}
    for arm in arms:
        vals = [_sweep(arm, r, ref[k])[key]["benefit"] for k, r in cells
                if key in _sweep(arm, r, ref[k])]
        if not vals:
            continue
        out[arm] = _stats(vals, dict(arm=arm, state=_measured_state(arm, cells),
                                     mem_bytes=_measured_mem(arm, cells), lr=lr))
    return out


def lrfree(arms=LRFREE_ARMS, cells=None, ref=None):
    """Table C: the LR-free rules at their own default -- a single reading per cell."""
    if cells is None:
        cells, ref = load_cells()
    out = {}
    for arm in arms:
        vals = [r[arm]["benefit"] for _, r in cells
                if isinstance(r.get(arm), dict) and "benefit" in r[arm]]
        if not vals:
            continue
        out[arm] = _stats(vals, dict(arm=arm, state=_measured_state(arm, cells)))
    return out


def seed_robustness(arms=HEADLINE, reading="rehearsed"):
    """P0-2: does adding seeds 3 and 4 change what the headline arms say?

    The extension screen runs seeds 0-2 while the BigData grid runs 0-4. Rather than re-running
    all twenty arms, the nine the claims rest on were extended, and this compares the two
    readings ON THE SAME arms: the 216-cell screen against the 360-cell version. What matters
    is not that the means shift a little -- more cells, different cells -- but whether the
    ORDERING of the arms survives, because every claim in the paper is an ordering claim.
    """
    c3, ref3 = load_cells(seed_ext=False)
    c5, ref5 = load_cells(seed_ext=True)
    fn = {"rehearsed": rehearsed, "fixed": fixed}[reading]
    a3 = fn(list(arms), cells=c3, ref=ref3) if reading == "rehearsed" else fn(list(arms), 1e-3, c3, ref3)
    a5 = fn(list(arms), cells=c5, ref=ref5) if reading == "rehearsed" else fn(list(arms), 1e-3, c5, ref5)
    rank3 = [a for a in sorted(a3, key=lambda x: -a3[x]["mean"])]
    rank5 = [a for a in sorted(a5, key=lambda x: -a5[x]["mean"])]

    # A raw rank list flips whenever two arms swap, INCLUDING arms that are tied -- and the
    # three arms at the top of this table sit inside 0.06 pt of each other with almost fully
    # overlapping CIs. Reporting that as "the ordering changed" would overstate what moved.
    # The claim the paper can actually make is about pairs the 3-seed reading SEPARATES: for
    # every pair whose 95% CIs are disjoint at 3 seeds, does the 5-seed reading agree?
    resolved, flipped = 0, []
    for i, a in enumerate(rank3):
        for b in rank3[i + 1:]:
            if a3[a]["lo"] > a3[b]["hi"]:                  # separated at 3 seeds
                resolved += 1
                if a5[a]["mean"] <= a5[b]["mean"]:
                    flipped.append((a, b))
    out = {}
    for a in arms:
        if a not in a3 or a not in a5:
            continue
        out[a] = dict(arm=a, n3=a3[a]["n_all"], n5=a5[a]["n_all"],
                      mean3=a3[a]["mean"], mean5=a5[a]["mean"],
                      d_mean=a5[a]["mean"] - a3[a]["mean"],
                      neg3=a3[a]["neg"], neg5=a5[a]["neg"],
                      rank3=rank3.index(a) + 1, rank5=rank5.index(a) + 1)
    return out, rank3, rank5, resolved, flipped


def _md(title, stats, cols):
    print(f"\n### {title}  (n cells = {max(s['n_all'] for s in stats.values())})")
    print("| arm | " + " | ".join(c[0] for c in cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for arm, s in sorted(stats.items(), key=lambda kv: -kv[1]["mean"]):
        print(f"| {arm} | " + " | ".join(c[1](s) for c in cols) + " |")


if __name__ == "__main__":
    cells, ref = load_cells()
    print(f"{len(cells)} cells")
    arms_lrful = ["adam", "sgdm", "sgd", "lion", "adafactor", "signsgd", "obgd",
                  "adaptive_obgd", "dons", "upgd", "idbd", "autostep",
                  "obsign", "obsign_t3e3", "obsign_t1e3", "relsign"]
    f2 = lambda k, sgn="+": (lambda s: "nan" if s[k] != s[k] else f"{s[k]:{sgn}.2f}")
    f1 = lambda k, sgn="+": (lambda s: "nan" if s[k] != s[k] else f"{s[k]:{sgn}.1f}")
    _md("Table A -- rehearsal-selected", rehearsed(arms_lrful, cells, ref),
        [("state", lambda s: f"{s['state']:.2f}x"), ("mean %", f2("mean")),
         ("95% CI", lambda s: f"[{s['lo']:+.2f}, {s['hi']:+.2f}]"),
         ("neg", lambda s: str(s["neg"])), ("worst %", f1("worst")), ("mis1x", f2("mis1x", ""))])
    _md("Table B -- fixed lr=1e-3, no rehearsal", fixed(arms_lrful, 1e-3, cells, ref),
        [("state", lambda s: f"{s['state']:.2f}x"), ("mean %", f2("mean")),
         ("median %", f2("median")), ("neg", lambda s: str(s["neg"])),
         ("worst %", f1("worst")), ("div", lambda s: str(s["diverged"]))])
    _md("Table C -- LR-free at their own default", lrfree(cells=cells, ref=ref),
        [("state", lambda s: f"{s['state']:.2f}x"), ("mean %", f2("mean")),
         ("median %", f2("median")), ("neg", lambda s: str(s["neg"])),
         ("worst %", f1("worst")), ("div", lambda s: str(s["diverged"]))])

    if os.path.exists(SEED_EXT):
        for reading in ("rehearsed", "fixed"):
            rb, r3, r5, resolved, flipped = seed_robustness(reading=reading)
            print(f"\n### P0-2 seed robustness ({reading}): 3 seeds vs 5 seeds")
            print("| arm | n(3s) | n(5s) | mean 3s | mean 5s | delta | neg 3s | neg 5s | rank 3s -> 5s |")
            print("|---" * 9 + "|")
            for a, e in sorted(rb.items(), key=lambda kv: -kv[1]["mean5"]):
                mv = "" if e["rank3"] == e["rank5"] else "  <-- moved"
                print(f"| {a} | {e['n3']} | {e['n5']} | {e['mean3']:+.2f} | {e['mean5']:+.2f} | "
                      f"{e['d_mean']:+.2f} | {e['neg3']} | {e['neg5']} | "
                      f"{e['rank3']} -> {e['rank5']}{mv} |")
            dmax = max(abs(e["d_mean"]) for e in rb.values())
            print(f"largest mean shift: {dmax:+.2f} pt | raw rank list identical: {r3 == r5}")
            print(f"pairs SEPARATED at 3 seeds (disjoint 95% CI): {resolved}; "
                  f"of these, reversed at 5 seeds: {len(flipped)}"
                  + (" -- " + ", ".join(f"{a}/{b}" for a, b in flipped) if flipped else ""))
    else:
        print(f"\n(no {os.path.basename(SEED_EXT)} yet -- P0-2 seed robustness not computed)")


# ---------------------------------------------------------------------------------------
# G2: the deployment-configuration reading. A DIFFERENT population from the three above -- 18
# cells (6 datasets x PatchTST x seeds 0-2, L96/H24) with the online phase restricted to a PEFT
# slice -- so it gets its own loader rather than a flag on load_cells(). Everything else is
# shared: same _stats, same conventions, same bootstrap seeding.
PEFT_OUT = os.path.join(os.path.dirname(OUT), "stage0_optimizers_{}.jsonl")
PEFT_ARMS = ("obsign_t1e3", "adafactor", "adam", "sgdm")


def peft_cells(which):
    rows = load_jsonl(PEFT_OUT.format(which))
    return [(k, r) for k, r in sorted(rows.items()) if k[0] in CORE]


def peft(which="calib", arms=PEFT_ARMS, lr=1e-3):
    """Per-arm stats under a PEFT strategy: rehearsal-selected AND at the shared default.

    Both readings matter for section VIII. The recipe tells a deployment to adapt few
    parameters; whether that changes WHICH optimizer to ship is the question, and it can only
    be answered on the same two readings the full-model study uses.
    """
    cells = peft_cells(which)
    out = {}
    for arm in arms:
        sel, fx, miss = [], [], []
        for _, r in cells:
            sw = _restrict(r.get(arm) or {})
            if not sw:
                continue
            sel.append(sel_oracle(sw)[1])
            miss.append(lr_miss_cost(sw))
            if f"{lr:g}" in sw:
                fx.append(sw[f"{lr:g}"]["benefit"])
        if not sel:
            continue
        d = _stats(sel, dict(arm=arm, which=which,
                             mis1x=float(np.median([m for m in miss if m == m]))
                             if miss else np.nan))
        f = _stats(fx) if fx else {}
        d["fixed_mean"], d["fixed_neg"] = f.get("mean", np.nan), f.get("neg", -1)
        got = [r[f"res_{arm}"]["n_adapt_params"] for _, r in cells if f"res_{arm}" in r]
        st = [r[f"res_{arm}"]["opt_state_bytes"] for _, r in cells if f"res_{arm}" in r]
        d["n_params"] = float(np.median(got)) if got else np.nan
        d["mem_bytes"] = (4 * d["n_params"] + float(np.median(st))) if st else np.nan
        out[arm] = d
    return out


def print_peft(which="calib"):
    S = peft(which)
    if not S:
        print(f"no rows yet for which={which}")
        return
    print(f"\n### PEFT reading: which={which}  ({len(peft_cells(which))} cells, PatchTST)")
    print(f"{'arm':13s} {'params':>8s} {'mem B':>9s} | {'rehearsed%':>11s} {'neg':>4s} "
          f"{'mis1x':>6s} | {'@1e-3%':>8s} {'neg':>4s}")
    for a, s in sorted(S.items(), key=lambda kv: -kv[1]["mean"]):
        print(f"{a:13s} {s['n_params']:8.0f} {s['mem_bytes']:9.0f} | {s['mean']:+11.2f} "
              f"{s['neg']:4d} {s['mis1x']:6.2f} | {s['fixed_mean']:+8.2f} {s['fixed_neg']:4d}")


def mis_split(arm, cells=None, ref=None, peft_which=None):
    """mis1x, separated by DIRECTION: (up, down) medians in benefit points.

    mis1x is the worse of the two neighbours one decade from the oracle, which hides the fact
    that for a relatively guarded rule the two sides are not the same kind of risk. Above the
    knee the rate cancels out of ObSign's update, so a rate one decade too HIGH costs exactly
    nothing; a rate one decade too LOW is below the knee, where the rule IS signSGD, and it
    simply forgoes benefit without pushing a cell below the static baseline. The paper makes
    that argument, so the two halves are computed here rather than in the prose.
    """
    from stage0_optimizers import _surface
    if peft_which is not None:
        rows = [(None, r) for _, r in peft_cells(peft_which)]
        sweeps = [_restrict(r.get(arm) or {}) for _, r in rows]
    else:
        if cells is None:
            cells, ref = load_cells()
        sweeps = [_sweep(arm, r, ref[k]) for k, r in cells]
    up, dn = [], []
    for sw in sweeps:
        if not sw:
            continue
        grid, ben = _surface(sw)
        i = int(np.argmax(ben))
        for tgt, acc in ((grid[i] * 10, up), (grid[i] / 10, dn)):
            j = min(range(len(grid)), key=lambda j: abs(np.log10(grid[j] / tgt)))
            if j != i:
                acc.append(ben[i] - ben[j])
    med = lambda v: float(np.median(v)) if v else float("nan")
    return med(up), med(dn)


def tau_selected(tau_arms=None, cells=None, ref=None):
    """What a deployment gets if it SELECTS tau instead of following the design rule for it.

    Section V argues tau from a rule that needs no data from the site: put the knee at least a
    decade below the rate you ship. The obvious objection is that a deployment which has a
    held-out pre-drift slice -- and this protocol's own burn-in selection needs one -- could
    just select tau on it, the way the tuned reading selects the rate. So measure that: pick
    (tau, lr) jointly by lowest validation MSE over every tau arm and every shared rate, and
    read the test benefit of the pick. Same estimator as rehearsed(), one dimension wider.

    It matters that this is reported rather than left as an option, because the answer is not
    the flattering one: the selection buys a little mean and gives up the property the section
    is selling, which is that no cell ends below the frozen model.
    """
    if cells is None:
        cells, ref = load_cells()
    tau_arms = tau_arms or [a for a in ("obsign", "obsign_t5e3", "obsign_t3e3", "obsign_t2e3",
                                        "obsign_t1p5e3", "obsign_t1e3")
                            if a in covered([a], cells)]
    vals, picks = [], []
    for k, r in cells:
        best = None
        for a in tau_arms:
            for key, v in _sweep(a, r, ref[k]).items():
                mse = v.get("val")
                if mse is None or mse != mse:
                    continue
                if best is None or mse < best[0]:
                    best = (mse, v["benefit"], a, float(key))
        if best:
            vals.append(best[1])
            picks.append((best[2], best[3]))
    if not vals:
        return {}
    d = _stats(vals, dict(arm="tau_val_selected", n_taus=len(tau_arms), n_picks=len(picks)))
    # The largest tau on offer is the one a validation slice tends to reach for -- it is the
    # least guarded, so it wins on the slice before the drift arrives.
    big = max(tau_arms, key=lambda a: _TAU_ORDER.index(a))
    d["picked_largest"] = sum(1 for a, _ in picks if a == big)
    d["largest_arm"] = big
    return d


# tau, ascending. Kept here rather than imported from stage0_figs so this module stays free of
# matplotlib; the two lists are asserted equal by gen_macros_stage0 when it emits the macros.
_TAU_ORDER = ["obsign_t1e3", "obsign_t1p5e3", "obsign_t2e3", "obsign_t3e3", "obsign_t5e3",
              "obsign"]


def paired(other, base="obsign_t1e3", reading="rehearsed", lr=1e-3, cells=None, ref=None,
           n_boot=4000):
    """Per-cell paired difference (other - base) on the 216 cells, with a bootstrap CI.

    The cells are the same population for every arm, so the paired difference is the estimator
    the design calls for; `_stats` draws each arm independently and its intervals are therefore
    much wider than the uncertainty in a COMPARISON. That matters in both directions and the
    paper used to rely on it in both: an untuned arm passed "competitive" because two wide
    intervals overlapped, and no pair of arms separated under the tuned reading. Neither
    survives the paired reading. peft_paired() below has done this correctly on the 18-cell
    slices since G2; this is the same estimator on the full grid, which is where the headline
    claims live.
    """
    if cells is None:
        cells, ref = load_cells()
    key = f"{lr:g}"

    def series(arm, how):
        """Per-cell benefit for `arm` under one of the three readings the paper uses."""
        out = []
        for k, r in cells:
            if how == "lrfree":                       # a single reading at the rule's own default
                v = r.get(arm)
                out.append(v["benefit"] if isinstance(v, dict) and "benefit" in v else np.nan)
                continue
            sw = _sweep(arm, r, ref[k])
            if not sw:
                out.append(np.nan)
            elif how == "rehearsed":
                out.append(sel_oracle(sw)[1])
            else:
                out.append(sw.get(key, {}).get("benefit", np.nan))
        return np.array(out, float)

    # `reading` describes how the BASE arm is read.  The reference arm it is compared against is
    # always read on the shared grid at its tuned rate, because that is the frontier R3(b) asks
    # about -- including when the base is a learning-rate-free rule with no grid of its own.
    a = series(other, "rehearsed" if reading in ("fixed", "lrfree") else reading)
    b = series(base, reading)

    # Same DIVERGED convention as _stats: a cell where either arm blew past -100% has no
    # meaningful difference to average (SGD has one, and it alone moves the mean by 1e12).
    # Dropping it here and counting it in `diverged` mirrors what the unpaired tables do.
    ok = np.isfinite(a) & np.isfinite(b) & (a >= DIVERGED) & (b >= DIVERGED)
    d = a - b
    n_div = int((np.isfinite(a) & np.isfinite(b) & ~ok).sum())
    fin = d[ok]
    if not fin.size:
        return dict(n=0)
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(fin, fin.size).mean() for _ in range(n_boot)])
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    return dict(n=int(fin.size), mean=float(fin.mean()), lo=lo, hi=hi, diverged=n_div,
                wins=int((fin > 0).sum()), separated=bool(lo * hi > 0),
                other=other, base=base, reading=reading)


def peft_paired(which, other, base="obsign_t1e3", lr=1e-3, n_boot=4000):
    """Per-cell paired difference (other - base) at the shared default, with a bootstrap CI.

    Paired, not two independent means: the 18 cells are the same datasets and seeds for both
    arms, and the between-cell spread (appliances +40 vs bdg2 +3) is an order of magnitude
    larger than the between-arm difference, so an unpaired interval would drown the effect.
    """
    cells = peft_cells(which)
    key = f"{lr:g}"
    ser = lambda a: np.array([_restrict(r.get(a) or {}).get(key, {}).get("benefit", np.nan)
                              for _, r in cells])
    d = ser(other) - ser(base)
    fin = d[np.isfinite(d)]
    if not fin.size:
        return dict(n=0)
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(fin, fin.size).mean() for _ in range(n_boot)])
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    return dict(n=int(fin.size), mean=float(fin.mean()), lo=lo, hi=hi,
                wins=int((fin > 0).sum()), separated=bool(lo * hi > 0))


# ---------------------------------------------------------------------------------------
# The requirement reading (added 2026-09-03 for the IEEE Access section IV).
#
# Sections III/IV originally read R2 off ONE shipped default (1e-3) and R3 off the negative-cell
# count alone. Both are too weak to carry the section's proposition, and the raw data says so:
#
#   * with R3 = "no cell below the static baseline", TEN arms pass at 1e-3, plain SGD among them
#     (+10.50, neg 0, 0x state). R1^R2^R3 is then satisfied by an existing method and section IV
#     is false as written. What separates the arms is not harm, it is whether the untuned reading
#     stays with the rehearsed frontier -- so R3 is TWO-part here: no-harm AND competitive.
#   * 1e-3 has no standing as "the" default (Lion ships 1e-4), and judging a class at a rate
#     borrowed from another optimizer is the conference version's own C3 confound in reverse.
#     R2 is therefore read over the WHOLE shared grid: for how many decades of shipped default
#     does the arm stay deployable? That turns "AdaFactor passes R2 by accident" from an
#     assertion into a measurement -- its band is one decade wide, ObSign's is three.
#
# R1 stays a continuous axis (the measured state multiplier); nothing here justifies a 0x
# threshold, and section IV's proposition is about reaching the (0x, untuned, competitive)
# POINT, not about disqualifying 0.54x.
COMPETITIVE_REF = "lion"          # the best tuned arm; the frontier R3(b) compares against
# R3(b) is a NON-INFERIORITY test on the paired per-cell difference against that arm, not an
# overlap of two independently drawn intervals.  Two reasons the overlap test had to go:
#   * it is the wrong estimator.  All arms are read on the same cells, so the uncertainty in a
#     COMPARISON is much smaller than in either mean; overlapping intervals are evidence of
#     nothing.  The same module already does this correctly on the PEFT slices (peft_paired).
#   * it was convenient in both directions -- it let a weak arm pass "competitive" and let the
#     paper claim no pair of tuned arms separated.  Neither survives the paired reading.
# COMPETITIVE_MARGIN is the largest gap to the reference that still counts as competitive.  The
# measured gaps fall into two groups with nothing between 1.2 and 2.0 points, so every value in
# that interval gives the same partition; 2.0 is the round number inside it.  The threshold is
# reported rather than tuned, and the gap that makes it insensitive is quoted in the paper.
COMPETITIVE_MARGIN = 2.0


def deployable(arms, cells=None, ref=None, ref_arm=COMPETITIVE_REF, grid=None):
    """Per (arm, shipped default lr): is the untuned reading deployable?

    R3(a) no-harm   -- neg == 0 over the cells
    R3(b) competitive -- the untuned 95% CI overlaps the CI of `ref_arm` REHEARSED, i.e. the
                       arm gives up nothing measurable against the best rate anyone could have
                       picked with a rehearsal pass.
    R2 is then the width, in decades, of the contiguous band of defaults where both hold.
    """
    if cells is None:
        cells, ref = load_cells()
    from online_eval import LR_GRID
    grid = list(grid or LR_GRID)
    grid = [x for x in grid if SHARED_LR[0] <= x <= SHARED_LR[1]]
    base = rehearsed([ref_arm], cells, ref)[ref_arm]
    out = {}
    for arm in arms:
        rows = []
        for lr in grid:
            s = fixed([arm], lr, cells, ref).get(arm)
            if not s or s["n_all"] == 0:
                continue
            ok_harm = s["neg"] == 0
            # R3(b): the paired shortfall against the reference arm's TUNED reading is smaller
            # than the margin, with 95% confidence.
            gap = paired(ref_arm, arm, reading="fixed", lr=lr, cells=cells, ref=ref,
                         n_boot=2000)
            ok_comp = bool(gap.get("n")) and gap["hi"] < COMPETITIVE_MARGIN
            rows.append(dict(lr=lr, mean=s["mean"], neg=s["neg"], lo=s["lo"], hi=s["hi"],
                             gap=gap.get("mean", float("nan")),
                             gap_hi=gap.get("hi", float("nan")),
                             diverged=s["diverged"], harm=ok_harm, comp=ok_comp,
                             ok=ok_harm and ok_comp))
        if not rows:
            continue
        # Widest CONTIGUOUS run of deployable defaults, reported BOTH ways.
        #   band          how many of the grid's rates, which is what a reader can count off
        #                 the table row and check
        #   band_decades  how wide that run actually is, which is the quantity that means
        #                 something to a deployment
        # These are NOT interchangeable and conflating them was a real bug: the shared grid is
        # spaced at half a decade, so eight consecutive rates span 3.5 decades, not eight, and
        # the whole grid is only log10(hi/lo) wide.  A single rate is a run of one and a width
        # of zero.
        best = run = 0
        best_lo = best_hi = None
        for i, r in enumerate(rows):
            if r["ok"]:
                run += 1
                if run > best:
                    best, best_lo, best_hi = run, rows[i - run + 1]["lr"], r["lr"]
            else:
                run = 0
        band_dec = float(np.log10(best_hi / best_lo)) if best else float("nan")
        out[arm] = dict(arm=arm, state=_measured_state(arm, cells), rows=rows,
                        band=best, band_decades=band_dec, band_lo=best_lo, band_hi=best_hi,
                        n_ok=sum(r["ok"] for r in rows),
                        ref_arm=ref_arm, ref_lo=base["lo"], ref_hi=base["hi"])
    return out


def print_deployable(arms=None, cells=None, ref=None):
    if cells is None:
        cells, ref = load_cells()
    arms = arms or ["adafactor", "obsign_t1e3", "obsign_t3e3", "relsign", "signsgd", "sgd",
                    "sgdm", "adam", "lion", "obgd", "upgd", "idbd", "autostep", "dons"]
    D = deployable(arms, cells, ref)
    any_d = next(iter(D.values()))
    lrs = [r["lr"] for r in any_d["rows"]]
    print(f"\n### Table G -- untuned at EVERY shipped default (n cells = {len(cells)}). "
          f"deployable = neg 0 AND paired shortfall vs tuned {any_d['ref_arm']} below "
          f"{COMPETITIVE_MARGIN:g} pt (95% CI). * = deployable.")
    print("| arm | state | " + " | ".join(f"{x:g}" for x in lrs) + " | band (decades) |")
    print("|---" * (len(lrs) + 3) + "|")
    for a, d in sorted(D.items(), key=lambda kv: (-kv[1]["band"], kv[1]["state"])):
        cell = lambda r: (f"{r['mean']:+.2f}/{r['neg']}" + ("*" if r["ok"] else ""))
        dec = "--" if d["band"] == 0 else f"{d['band_decades']:.1f}"
        print(f"| {a} | {d['state']:.2f}x | " + " | ".join(cell(r) for r in d["rows"])
              + f" | **{d['band']}** ({dec} dec) |")


def requirement_check(cells=None, ref=None, lr=1e-3):
    """Section IV's proposition, evaluated mechanically at the shipped default `lr`.

    Prints which arms pass R3 under the WEAK reading (no-harm only) and under the TWO-PART
    reading. The weak reading is kept in the output on purpose: it is what makes plain SGD a
    counterexample, and the paper has to show it knows that.
    """
    if cells is None:
        cells, ref = load_cells()
    arms = ["adam", "sgdm", "sgd", "lion", "adafactor", "signsgd", "obgd", "adaptive_obgd",
            "dons", "upgd", "idbd", "autostep", "obsign", "obsign_t3e3", "obsign_t1e3",
            "relsign"]
    S = fixed(arms, lr, cells, ref)
    base = rehearsed([COMPETITIVE_REF], cells, ref)[COMPETITIVE_REF]
    weak = [a for a in S if S[a]["neg"] == 0]
    gaps = {a: paired(COMPETITIVE_REF, a, reading="fixed", lr=lr, cells=cells, ref=ref)
            for a in weak}
    strong = [a for a in weak if gaps[a].get("n") and gaps[a]["hi"] < COMPETITIVE_MARGIN]
    print(f"\n### Requirement check at the shipped default lr={lr:g}")
    print(f"R3 weak  (neg == 0 only): {len(weak):2d} arms -- "
          + ", ".join(f"{a} ({S[a]['mean']:+.2f}, {S[a]['state']:.2f}x)"
                      for a in sorted(weak, key=lambda a: -S[a]["mean"])))
    print(f"R3 two-part (+ paired shortfall vs tuned {COMPETITIVE_REF} under "
          f"{COMPETITIVE_MARGIN:g} pt): {len(strong):2d} arms -- "
          + ", ".join(f"{a} ({S[a]['mean']:+.2f}, {S[a]['state']:.2f}x)"
                      for a in sorted(strong, key=lambda a: -S[a]["mean"])))
    lrfree_arms = lrfree(cells=cells, ref=ref)
    lw = [a for a, s in lrfree_arms.items() if s["neg"] == 0]
    _lg = {a: paired(COMPETITIVE_REF, a, reading="lrfree", cells=cells, ref=ref)
           for a in lrfree_arms}
    ls = [a for a in lw if _lg[a].get("n") and _lg[a]["hi"] < COMPETITIVE_MARGIN]
    print(f"LR-free at their own default: no-harm {len(lw)}/{len(lrfree_arms)} ("
          + ", ".join(f"{a} {lrfree_arms[a]['mean']:+.2f}" for a in lw)
          + f"); competitive {len(ls)}/{len(lrfree_arms)}"
          + (" -- " + ", ".join(ls) if ls else ""))
    return dict(weak=weak, strong=strong, stats=S, ref=base)


def guard_split(cells=None, ref=None, guarded="obsign_t1e3", unguarded="relsign"):
    """Where does the min(lr, tau*RMS) cap actually DO anything?

    At the 1e-3 default the guarded and unguarded rules are identical to two decimals, which
    reads as "the cap does nothing" if that is the only column shown. It does nothing there by
    construction: the knee sits at 5.3e-5, so 1e-3 is above it and the cap is inactive. The cap
    is the whole rule BELOW the knee, where it degrades to signSGD instead of to a vanishing
    relative step -- so the ablation has to be read on the low side of the grid.
    """
    if cells is None:
        cells, ref = load_cells()
    from online_eval import LR_GRID
    print(f"\n### Guard ablation: {guarded} vs {unguarded} at every shipped default")
    print(f"| lr | {guarded} | {unguarded} | delta |")
    print("|---|---|---|---|")
    for lr in LR_GRID:
        if not (SHARED_LR[0] <= lr <= SHARED_LR[1]):
            continue
        g = fixed([guarded], lr, cells, ref).get(guarded)
        u = fixed([unguarded], lr, cells, ref).get(unguarded)
        if not g or not u:
            continue
        print(f"| {lr:g} | {g['mean']:+.2f} / {g['neg']} | {u['mean']:+.2f} / {u['neg']} "
              f"| {g['mean'] - u['mean']:+.2f} |")


if __name__ == "__main__":
    # The requirement reading runs from a second guard because its functions are defined below
    # the first one; the alternative is reordering the module, which would churn the diff for
    # every reader who knows where rehearsed()/fixed() live.
    _cells, _ref = load_cells()
    print_deployable(cells=_cells, ref=_ref)
    requirement_check(_cells, _ref)
    guard_split(_cells, _ref)
