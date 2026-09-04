"""Extension-paper macros: results/tsf_edge/macros_ext.tex, from the Stage-0/0b/0c data.

Same rule as gen_macros.py -- no number in the extension paper is ever hand-typed (CLAUDE.md)
-- but a SEPARATE generator writing a SEPARATE file, for two reasons:

  1. results/tsf_edge/macros.tex must stay byte-identical to the public repo's copy, which is
     the property the conference submission state is tracked by. A generator that wrote both
     files could not give that guarantee.
  2. the two papers are read and revised on different clocks, so their numbers are regenerated
     on different clocks too; one generator per paper keeps a rerun of one from touching the
     other.

The extension paper does \\input{macros.tex} AND \\input{macros_ext.tex}; names here are
prefixed Ext... so the two namespaces can never collide.

All numbers come from stage0_pool.py -- the single pooled implementation -- so the table, the
figures and the macros cannot drift apart.

Regenerate: .venv/bin/python experiments/tsf_edge/gen_macros_stage0.py

TERMINOLOGY: the paper says TUNED rate where these macros say Reh (\\ExtReh*).  "Rehearsal"
means replay in continual learning, a field this paper compares against, so the word left the
captions on 2026-09-03 -- but the 112 macro names did NOT change.  A macro rename that leaves
a value behind is the failure this project has already had once; the project glossary records it.
"""
from __future__ import annotations
import datetime, os, re, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, "results", "tsf_edge")
OUTFILE = os.path.join(RES, "macros_ext.tex")
sys.path.insert(0, HERE)
import stage0_pool as pool
from stage0_optimizers import LR_GRID_BY_OPT, _surface, sel_oracle
# The top of the shared grid BEFORE the P0-1 fill-in of 2026-08-31 (it ran 3e-6..1e-2;
# 3e-2 and 1e-1 were added by that pass; the fill-in runner and its merge script are
# run_stage0_fillin.sh and merge_stage0_fillin.py.
PREFILL_LR_HI = 1e-2
from online_eval import LR_GRID

# Readable macro stems. The three ObSign arms differ only in tau, which has no published
# default and is therefore SWEPT, so the stem spells tau out rather than numbering the arms:
# a reader of \ExtRehObsignTauThreeEThreeMean cannot mistake which guard level it refers to.
ARM_TEX = {
    "adam": "Adam", "sgdm": "Sgdm", "sgd": "Sgd",
    "lion": "Lion", "adafactor": "Adafactor", "signsgd": "Signsgd",
    "obgd": "Obgd", "adaptive_obgd": "AdaObgd", "dons": "Dons", "upgd": "Upgd",
    "idbd": "Idbd", "autostep": "Autostep",
    "obsign": "ObsignTauOneETwo", "obsign_t5e3": "ObsignTauFiveEThree",
    "obsign_t3e3": "ObsignTauThreeEThree", "obsign_t2e3": "ObsignTauTwoEThree",
    "obsign_t1p5e3": "ObsignTauOnePointFiveEThree",
    "obsign_t1e3": "ObsignTauOneEThree", "relsign": "Relsign",
    "prodigy": "Prodigy", "dog": "Dog", "dadapt_adam": "DadaptAdam", "dadapt_sgd": "DadaptSgd",
}
# Human-readable names for macros whose VALUE is printed as prose.  ARM_TEX above builds
# macro NAMES; putting an identifier like "ObsignTauOneEThree" into a sentence is a bug that
# LaTeX cannot catch, so any macro that names an arm in running text uses this map instead.
ARM_LABEL = {
    "adam": "Adam", "sgdm": "SGD with momentum", "sgd": "SGD", "lion": "Lion",
    "adafactor": "AdaFactor", "signsgd": "signSGD", "obgd": "ObGD",
    "adaptive_obgd": "Ada-ObGD", "dons": "discounted ONS", "upgd": "UPGD", "idbd": "IDBD",
    "autostep": "Autostep", "relsign": "RelSign",
    "obsign": r"ObSign $\tau{=}10^{-2}$", "obsign_t5e3": r"ObSign $\tau{=}5{\cdot}10^{-3}$",
    "obsign_t3e3": r"ObSign $\tau{=}3{\cdot}10^{-3}$",
    "obsign_t2e3": r"ObSign $\tau{=}2{\cdot}10^{-3}$",
    "obsign_t1p5e3": r"ObSign $\tau{=}1.5{\cdot}10^{-3}$",
    "obsign_t1e3": r"ObSign $\tau{=}10^{-3}$",
    "prodigy": "Prodigy", "dog": "DoG", "dadapt_adam": "D-Adaptation over Adam",
    "dadapt_sgd": "D-Adaptation over SGD",
}
LRFUL = ["adam", "sgdm", "sgd", "lion", "adafactor", "signsgd", "obgd", "adaptive_obgd",
         "dons", "upgd", "idbd", "autostep",
         "obsign", "obsign_t3e3", "obsign_t1e3", "relsign"]
BRACKET_ARMS = ["lion", "adafactor", "signsgd", "obsign", "obsign_t3e3", "obsign_t1e3",
                "relsign"]

lines, seen, warnings = [], set(), []


def emit(name, val):
    assert re.fullmatch(r"[A-Za-z]+", name), f"bad macro name: {name}"
    assert name not in seen, f"duplicate macro: {name}"
    seen.add(name)
    lines.append(f"\\newcommand{{\\{name}}}{{{val}}}")


def section(title):
    lines.append(f"\n% ---- {title} ----")


def lrtex(x):
    """A learning rate as LaTeX math.  "%g" gives 3e-06 and 0.001, which is how a rate ends up
    typeset as a float in running prose; every rate macro goes through here instead.  Returns
    self-contained math ($...$), so these macros are used OUTSIDE $ $ -- see the assertion in
    check_paper.py's hand-typed-number check, which strips math before it looks for decimals."""
    import math
    e = math.floor(math.log10(x) + 1e-9)
    m = x / 10 ** e
    mant = "" if abs(m - 1) < 1e-6 else (f"{m:.0f}" if abs(m - round(m)) < 1e-6
                                         else f"{m:.2g}") + r"{\cdot}"
    return f"${mant}10^{{{e}}}$"


def thousands(x):
    """A parameter count with LaTeX thousands separators.  optimizer_table.tex has always
    printed $N=85{,}670$ in its caption; the body printed the same number as 85670 from a
    macro, so the same quantity appeared two ways on facing pages."""
    return f"{int(round(x)):,}".replace(",", r"{,}")


def s2(x): return f"{round(x, 2) + 0.0:+.2f}"          # signed, avoids "-0.00"
def s1(x): return f"{round(x, 1) + 0.0:+.1f}"
def f2(x): return f"{x:.2f}"


def emit_group(prefix, stats, fields):
    for arm in [a for a in ARM_TEX if a in stats]:
        s = stats[arm]
        for suffix, fmt, key in fields:
            v = s[key]
            if v != v:                                       # NaN: emit nothing, warn once
                warnings.append(f"{prefix}{ARM_TEX[arm]}{suffix} is NaN -- macro omitted")
                continue
            emit(prefix + ARM_TEX[arm] + suffix, fmt(v))


cells, ref = pool.load_cells()

# Stage 0d (2026-09-04) added three values of tau. Its file is written WHILE the run proceeds,
# so an arm enters the paper's arm list only once it has a reading in every cell -- a pooled
# mean over half the grid would look like a number rather than like a half-finished run.
TAU_EXTRA = ["obsign_t1p5e3", "obsign_t2e3", "obsign_t5e3"]
TAU_READY = pool.covered(TAU_EXTRA, cells)
if TAU_READY != TAU_EXTRA:
    warnings.append("Stage-0d arms not yet complete over all cells, so they are excluded from "
                    "every table: " + ", ".join(a for a in TAU_EXTRA if a not in TAU_READY))
LRFUL += TAU_READY
BRACKET_ARMS += TAU_READY

section("Extension study: grid size and protocol")
emit("ExtCells", str(len(cells)))
emit("ExtSeeds", str(len({k[4] for k in (c[0] for c in cells)})))
emit("ExtDatasets", str(len({k[0] for k in (c[0] for c in cells)})))
emit("ExtShapes", str(len({(k[2], k[3]) for k in (c[0] for c in cells)})))
emit("ExtBackbones", str(len({k[1] for k in (c[0] for c in cells)})))
emit("ExtArms", str(len(LRFUL) + len(pool.LRFREE_ARMS)))
# The population for any claim quantified over the shared LR grid.  The LR-free rules are
# read at their own defaults and were never swept, so \ExtArms is the wrong denominator
# for a statement like "at every one of the ten rates".
emit("ExtGridArms", str(len(LRFUL)))
emit("ExtSharedLrPoints", str(len(LR_GRID)))
emit("ExtSharedLrLo", lrtex(min(LR_GRID)))
emit("ExtSharedLrHi", lrtex(max(LR_GRID)))
emit("ExtFixedLr", lrtex(1e-3))
# The protocol constant section III quotes.  Inherited from the conference version, but the
# extension paper states it, so it is a macro here rather than a number typed into prose.
from online_eval import VAL_FRAC
emit("ExtValPct", f"{100 * VAL_FRAC:.0f}")

section("Table A: TUNED on the held-out pre-drift slice of the shared grid (benefit %, "
        "bootstrap 95% CI, "
        "negative cells at < -0.05%, worst non-diverged cell, and the cost of a one-decade "
        "rate error). Diverged cells (< -100%) are out of mean/worst, in neg.")
reh = pool.rehearsed(LRFUL, cells, ref)
emit_group("ExtReh", reh, [("Mean", s2, "mean"), ("Lo", s2, "lo"), ("Hi", s2, "hi"),
                           ("Neg", str, "neg"), ("Worst", s1, "worst"),
                           ("MisOne", f2, "mis1x"), ("State", f2, "state")])

section("Table B: ONE shared rate, no tuning at all -- the reading for a device that "
        "cannot afford a tuning pass")
fix = pool.fixed(LRFUL, 1e-3, cells, ref)
emit_group("ExtFix", fix, [("Mean", s2, "mean"), ("Median", s2, "median"),
                           ("Neg", str, "neg"), ("Worst", s1, "worst"),
                           ("Div", str, "diverged"), ("State", f2, "state")])

section("Table C: learning-rate-free rules at their own default (their entire pitch)")
free = pool.lrfree(cells=cells, ref=ref)
emit_group("ExtFree", free, [("Mean", s2, "mean"), ("Median", s2, "median"),
                             ("Neg", str, "neg"), ("Worst", s1, "worst"),
                             ("Div", str, "diverged"), ("State", f2, "state")])

section("Bracketing audit (CLAUDE.md rule: a shared grid is only fair if it BRACKETS every "
        "method's optimum). Cells whose per-cell TEST oracle sits on an EDGE of that arm's "
        "own grid -- a non-zero count means the arm's quality is under-estimated and its "
        "mis1x optimistic, because the rate one decade past the edge was never run.")
for arm in BRACKET_ARMS:
    grid = sorted(LR_GRID_BY_OPT.get(arm, LR_GRID))
    on_edge = top = 0
    n = 0
    for k, r in cells:
        sw = r.get(arm)
        if not isinstance(sw, dict) or not sw:
            continue
        n += 1
        rates = sorted(float(x) for x in sw)
        orc = r.get(f"oracle_lr_{arm}")
        if orc is None:
            continue
        if abs(orc - rates[0]) < 1e-12 or abs(orc - rates[-1]) < 1e-12:
            on_edge += 1
        if abs(orc - rates[-1]) < 1e-12:
            top += 1
    if not n:
        continue
    emit("ExtBracket" + ARM_TEX[arm] + "Cells", str(n))
    emit("ExtBracket" + ARM_TEX[arm] + "OnEdge", str(on_edge))
    # The same audit as it stood BEFORE the P0-1 fill-in, obtained by restricting each cell's
    # sweep to the eight rates that existed then.  Without this the text can only report the
    # post-fill-in count, which is 0 for AdaFactor and therefore says nothing about what the
    # fill-in achieved.  This is recomputed from the canonical sweeps, not remembered.
    before = 0
    for _k, _r in cells:
        _sw = _r.get(arm)
        if not isinstance(_sw, dict) or not _sw:
            continue
        _restricted = {x: v for x, v in _sw.items() if float(x) <= PREFILL_LR_HI * (1 + 1e-9)}
        if len(_restricted) < 2:
            continue
        _g, _b = _surface(_restricted)
        _i = int(np.argmax(_b))
        before += (_i == 0 or _i == len(_g) - 1)
    emit("ExtBracket" + ARM_TEX[arm] + "OnEdgeBefore", str(before))
    emit("ExtBracket" + ARM_TEX[arm] + "AtTop", str(top))
    emit("ExtBracket" + ARM_TEX[arm] + "Rates", str(len({f"{x:g}" for x in grid})))

section("P0-2 seed robustness: the nine headline arms re-read at 5 seeds (360 cells) against "
        "the 3-seed screen (216). The claim is not that the means are identical -- more cells "
        "move them slightly -- but that no pair the 3-seed reading could SEPARATE (disjoint "
        "95% CI) comes back reversed.")
if os.path.exists(pool.SEED_EXT):
    for reading in ("rehearsed", "fixed"):
        rb, r3, r5, resolved, flipped = pool.seed_robustness(reading=reading)
        tag = "Reh" if reading == "rehearsed" else "Fix"
        emit("ExtSeed" + tag + "Arms", str(len(rb)))
        emit("ExtSeed" + tag + "CellsThree", str(max(e["n3"] for e in rb.values())))
        emit("ExtSeed" + tag + "CellsFive", str(max(e["n5"] for e in rb.values())))
        emit("ExtSeed" + tag + "MaxShift", s2(max(abs(e["d_mean"]) for e in rb.values())))
        emit("ExtSeed" + tag + "SeparatedPairs", str(resolved))
        emit("ExtSeed" + tag + "ReversedPairs", str(len(flipped)))
        for arm, e in rb.items():
            emit("ExtSeed" + tag + ARM_TEX[arm] + "MeanFive", s2(e["mean5"]))
            emit("ExtSeed" + tag + ARM_TEX[arm] + "NegFive", str(e["neg5"]))
else:
    warnings.append("stage0_seeds34.jsonl missing -- P0-2 seed macros omitted")

section("mis1x split by DIRECTION. The headline mis1x is the worse neighbour one decade from "
        "the oracle; splitting it is what lets the paper say that the relative guard removes "
        "the UPWARD risk entirely and leaves only a downward one, which forgoes benefit "
        "without pushing a cell below the static baseline.")
for arm in ["obsign_t1e3", "obsign_t3e3", "obsign", "signsgd", "adafactor", "sgdm", "adam",
            "obgd"]:
    up, dn = pool.mis_split(arm, cells, ref)
    if up != up or dn != dn:
        warnings.append(f"mis_split({arm}) is NaN -- macros omitted")
        continue
    emit("ExtMisUp" + ARM_TEX[arm], f2(up))
    emit("ExtMisDown" + ARM_TEX[arm], f2(dn))

section("G2: the deployment configuration. The same contenders with the ONLINE phase "
        "restricted to a PEFT slice (PatchTST, L96/H24, 6 datasets x seeds 0-2). Warmup is "
        "unchanged (always full-model); only what the stream adapts is restricted.")
PEFT_ARMS = list(pool.PEFT_ARMS)
peft_ok = True
for which, tag in (("calib", "Calib"), ("head", "Head")):
    st = pool.peft(which, PEFT_ARMS)
    if not st:
        warnings.append(f"stage0_optimizers_{which}.jsonl missing -- G2 macros omitted")
        peft_ok = False
        continue
    emit("ExtPeft" + tag + "Cells", str(len(pool.peft_cells(which))))
    emit_group("ExtPeft" + tag, st,
               [("Mean", s2, "mean"), ("Lo", s2, "lo"), ("Hi", s2, "hi"), ("Neg", str, "neg"),
                ("MisOne", f2, "mis1x"), ("FixMean", s2, "fixed_mean"),
                ("FixNeg", str, "fixed_neg"), ("Params", thousands, "n_params"),
                ("MemBytes", lambda v: f"{v:.0f}", "mem_bytes"),
                # kB as well as bytes: section VI-B compares these against on-chip SRAM
                # budgets, which are quoted in kB, and "203224 bytes against 262144" is a
                # sentence a reader has to do arithmetic on. 1 kB = 1024 B.
                ("MemKB", lambda v: f"{v / 1024:.0f}", "mem_bytes")])
    for other in ("sgdm", "adafactor", "adam"):
        d = pool.peft_paired(which, other)
        if not d.get("n"):
            continue
        stem = "ExtPeft" + tag + "Paired" + ARM_TEX[other]
        emit(stem + "Mean", s2(d["mean"]))
        emit(stem + "Lo", s2(d["lo"]))
        emit(stem + "Hi", s2(d["hi"]))
        emit(stem + "Wins", str(d["wins"]))
        emit(stem + "Separated", "yes" if d["separated"] else "no")
    for arm in ("obsign_t1e3",):
        up, dn = pool.mis_split(arm, peft_which=which)
        emit("ExtPeft" + tag + "MisUp" + ARM_TEX[arm], f2(up))
        emit("ExtPeft" + tag + "MisDown" + ARM_TEX[arm], f2(dn))

section("Adaptation memory in kB at the reference configuration, for section VI-B (what the "
        "state multiplier means against a real memory budget). The multiplier is "
        "dimensionless and every device budget is in bytes, so the section needs both. "
        "REF_CONFIG is one configuration by necessity -- the trainable-parameter count runs "
        "from 4,656 to 210,964 over the grid, so a byte figure pooled over all cells would "
        "describe no deployment that exists (stage0_pool.REF_CONFIG). Only the SCALE depends "
        "on that choice: memory is 4N(1+s) with N shared by every arm.")
_bb, _L, _H = pool.REF_CONFIG
emit("ExtRefConfigL", str(_L))
emit("ExtRefConfigH", str(_H))
_np = [r["res_obsign_t1e3"]["n_adapt_params"] for _, r in cells
       if "res_obsign_t1e3" in r and (r["backbone"], r["L"], r["H"]) == (_bb, _L, _H)]
emit("ExtRefConfigParams", thousands(np.median(_np)))
for _a in ("obsign_t1e3", "adafactor", "sgdm", "adam"):
    _m = pool._measured_mem(_a, cells)
    emit("ExtFullMemKB" + ARM_TEX[_a], f"{_m / 1024:.0f}")

section("What a tuning pass actually buys (section VI-E). The tuned reading selects each "
        "rule's rate per cell on a held-out pre-drift slice, and it beats the untuned reading "
        "-- but not because per-site selection finds something a fixed rate cannot. For every "
        "rule below there is a SINGLE rate on the shared grid whose untuned reading matches or "
        "beats that rule's tuned reading. That rate is chosen with hindsight over the whole "
        "216-cell population, which is not a selection a deployment can make, and it differs "
        "from rule to rule -- so what the pass buys is knowing WHICH rate, which is exactly "
        "what R2 says a deployment does not know. The spread of the per-cell selections is "
        "the same point from the other side: no single rate is the selected one in most cells.")
_TUNE_ARMS = ["lion", "adam", "signsgd", "adafactor", "sgdm", "obsign_t1e3"]
for _a in _TUNE_ARMS:
    _sels = []
    for _k, _r in cells:
        _sw = pool._sweep(_a, _r, ref[_k])
        if _sw:
            _sels.append(sel_oracle(_sw)[0])
    if not _sels:
        continue
    _sels = np.array(_sels, float)
    _u, _c = np.unique(_sels, return_counts=True)
    _best = max(((_lr, pool.fixed([_a], _lr, cells, ref).get(_a, {}).get("mean", np.nan))
                 for _lr in LR_GRID), key=lambda t: (-1e9 if t[1] != t[1] else t[1]))
    # WHICH WAY the selection errs. A tuned column below its own untuned one looks like a bug
    # unless the reader is told that validation selects on the PRE-DRIFT slice and picks low:
    # for every arm here the selected rate sits below the cell's own test optimum about twice
    # as often as above it. For a rule whose high rates are harmless that costs benefit; for
    # signSGD or Adam the same conservatism is what saves them, which is why only the guarded
    # rules show the inversion.
    _lo = sum(1 for _k, _r in cells
              if (_sw := pool._sweep(_a, _r, ref[_k])) and sel_oracle(_sw)[0] < sel_oracle(_sw)[2])
    _hi = sum(1 for _k, _r in cells
              if (_sw := pool._sweep(_a, _r, ref[_k])) and sel_oracle(_sw)[0] > sel_oracle(_sw)[2])
    _n = sum(1 for _k, _r in cells if pool._sweep(_a, _r, ref[_k]))
    _stem = "ExtTunePass" + ARM_TEX[_a]
    emit(_stem + "SelBelowOracle", f"{100 * _lo / _n:.0f}")
    emit(_stem + "SelAboveOracle", f"{100 * _hi / _n:.0f}")
    emit(_stem + "BestSingleLr", lrtex(_best[0]))
    emit(_stem + "BestSingleMean", s2(_best[1]))
    emit(_stem + "TunedMinusBestSingle", s2(reh[_a]["mean"] - _best[1]))
    emit(_stem + "SelDistinct", str(len(_u)))
    emit(_stem + "SelSpreadDecades", f"{np.log10(_sels.max() / _sels.min()):.1f}")
    emit(_stem + "SelModeLr", lrtex(float(_u[_c.argmax()])))
    emit(_stem + "SelModeShare", f"{100 * _c.max() / _c.sum():.0f}")
# The count that carries the sentence: how many of these rules have a single fixed rate at
# least as good as their own tuned reading.  Computed, not asserted -- if a rerun moves one,
# the number moves with it.
_n_matched = sum(1 for _a in _TUNE_ARMS
                 if reh[_a]["mean"] <= max((pool.fixed([_a], _lr, cells, ref).get(_a, {})
                                            .get("mean", -1e9) for _lr in LR_GRID)))
emit("ExtTunePassArms", str(len(_TUNE_ARMS)))
emit("ExtTunePassMatchedByFixedRate", str(_n_matched))

section("G1: our simplified calibration point against the official PETSA parameterisation, "
        "both inside THIS paper's protocol (same warmup, same shared grid, same held-out "
        "selection). Only PETSA's modules and loss are ported; its TAFAS-derived online "
        "schedule is not -- see petsa_calib.py.")
try:
    import petsa_compare
    tab = petsa_compare.pairs()
except Exception as e:                                   # missing file, not a code error
    tab, warnings_msg = [], f"PETSA comparison unavailable ({e}) -- G1 macros omitted"
    warnings.append(warnings_msg)
if tab:
    P = np.array([t[2] for t in tab]); O = np.array([t[4] for t in tab])
    n_par, o_par = petsa_compare.n_params()
    emit("ExtPetsaCells", str(len(tab)))
    emit("ExtPetsaMean", s2(float(P.mean())))
    emit("ExtPetsaOursMean", s2(float(O.mean())))
    emit("ExtPetsaDiff", s2(float(P.mean() - O.mean())))
    emit("ExtPetsaWins", str(int((P > O).sum())))          # cells where PETSA wins
    emit("ExtPetsaOursWins", str(int((O > P).sum())))      # cells where ours wins
    emit("ExtPetsaParams", f"{n_par:.0f}")
    emit("ExtPetsaOursParams", thousands(o_par))
    emit("ExtPetsaBenefitRatio", f"{100 * P.mean() / O.mean():.0f}")

section("The requirement reading (section IV). R3 is TWO-part -- no-harm (neg 0) AND "
        "competitive (untuned 95% CI overlapping the best TUNED arm) -- because the "
        "no-harm half alone is passed by ten arms including plain SGD, which would make "
        "section IV's proposition false. R2 is read over the WHOLE shared grid rather than at "
        "one borrowed default: the band is how many decades of shipped default keep the arm "
        "deployable. NOTE on the relsign row: RelSign has no lr (its lr IS tau), so its band "
        "is a sensitivity to its OWN hyper-parameter, not to a shipped rate -- which is the "
        "point of the ablation, not a category error.")
# EVERY arm on the shared grid, not a chosen subset. \ExtHarmFreeEverywhereCount is quoted in
# the text as "one of the \ExtGridArms rules measured on the shared grid", and that sentence is
# only true if the count ranges over the same population the denominator names -- with a
# hand-written subset here, an arm left out of the list could be harm-free everywhere and the
# macro would still say 1.  (It happened the other way round on 2026-09-04: the subset was
# missing adaptive_obgd, and the tau arms added that day would have been missing too.)
DEPLOY_ARMS = list(LRFUL)
dep = pool.deployable(DEPLOY_ARMS, cells, ref)
emit("ExtDeployRefArm", ARM_LABEL[pool.COMPETITIVE_REF])
emit("ExtDeployRefLo", s2(next(iter(dep.values()))["ref_lo"]))
emit("ExtDeployRefHi", s2(next(iter(dep.values()))["ref_hi"]))
emit("ExtDeployRates", str(len(next(iter(dep.values()))["rows"])))
# The grid's own width, so a reader can see that no band can exceed it.  The bug this guards
# against: the grid is spaced at HALF a decade, so a count of consecutive rates is not a count
# of decades, and "8 decades" on a grid 4.5 decades wide is self-evidently impossible.
emit("ExtSharedLrDecades", f"{np.log10(max(LR_GRID) / min(LR_GRID)):.1f}")
emit("ExtSharedLrStepDecades", f"{np.log10(max(LR_GRID) / min(LR_GRID)) / (len(LR_GRID) - 1):.1f}")
for arm, d in dep.items():
    emit("ExtDeploy" + ARM_TEX[arm] + "Band", str(d["band"]))
    # A run of one rate is a width of zero; emit it as 0.0 rather than omitting it, so a
    # sentence that says "a band of X decades" is never silently left with no macro.
    emit("ExtDeploy" + ARM_TEX[arm] + "BandDecades",
         "0.0" if d["band"] < 2 else f"{d['band_decades']:.1f}")
    emit("ExtDeploy" + ARM_TEX[arm] + "Ok", str(d["n_ok"]))
    emit("ExtDeploy" + ARM_TEX[arm] + "HarmFree", str(sum(r["harm"] for r in d["rows"])))
# The arms that never put a cell below the static baseline at ANY rate on the shared grid.
# TWO qualify once the population is every grid arm rather than a hand-written subset, and the
# second one matters: Ada-ObGD is harmless at all ten rates and competitive at none of them,
# which is the non-stationary class's whole pattern. Harmlessness alone is therefore not the
# distinguishing property even here, so the claim the paper makes is the CONJUNCTION -- harmless
# everywhere AND deployable somewhere -- and both counts are emitted so the text can say so.
_all_harm_free = [a for a, d in dep.items() if all(r["harm"] for r in d["rows"])]
emit("ExtHarmFreeEverywhereCount", str(len(_all_harm_free)))
emit("ExtHarmFreeEverywhereArms", ", ".join(ARM_LABEL[a] for a in _all_harm_free) or "none")
_hf_dep = [a for a in _all_harm_free if dep[a]["n_ok"] > 0]
_hf_not = [a for a in _all_harm_free if dep[a]["n_ok"] == 0]
emit("ExtHarmFreeAndDeployableCount", str(len(_hf_dep)))
emit("ExtHarmFreeAndDeployableArms", ", ".join(ARM_LABEL[a] for a in _hf_dep) or "none")
emit("ExtHarmFreeNeverDeployableArms", ", ".join(ARM_LABEL[a] for a in _hf_not) or "none")

emit("ExtCompetitiveMargin", f"{pool.COMPETITIVE_MARGIN:g}")

# Why the margin can be quoted without looking reverse-engineered: at the shipped default the
# measured shortfalls fall into two groups with a wide gap between them, so every threshold
# inside that gap gives the same partition.  These two macros ARE that gap.
_gaps = {a: pool.paired(pool.COMPETITIVE_REF, a, reading="fixed", lr=1e-3, cells=cells, ref=ref)
         for a in LRFUL}
_pass = [g for g in _gaps.values() if g.get("n") and g["hi"] < pool.COMPETITIVE_MARGIN]
_fail = [g for g in _gaps.values() if g.get("n") and g["hi"] >= pool.COMPETITIVE_MARGIN]
emit("ExtGapPassMaxHi", f"{max(g['hi'] for g in _pass):.2f}")
emit("ExtGapFailMinHi", f"{min(g['hi'] for g in _fail):.2f}")
for _a in ("adafactor", "obsign_t1e3", "autostep", "sgdm", "sgd", "obgd", "lion", "adam"):
    _g = _gaps[_a]
    emit("ExtGap" + ARM_TEX[_a] + "Mean", s2(_g["mean"]))
    emit("ExtGap" + ARM_TEX[_a] + "Lo", s2(_g["lo"]))
    emit("ExtGap" + ARM_TEX[_a] + "Hi", s2(_g["hi"]))

section("Paired per-cell differences against the proposed rule, TUNED reading. The unpaired "
        "intervals of Table A cannot answer a comparison -- every arm is read on the same "
        "cells, so the uncertainty in a difference is much smaller than in either mean. The "
        "paper used to claim no pair separated under this reading; on the paired estimator "
        "several do, and the ones that do are all ABOVE the proposed rule, which is why the "
        "paper claims robustness rather than quality.")
for _a in ("lion", "adam", "signsgd", "obsign_t3e3", "adafactor", "relsign", "sgdm", "obgd"):
    _d = pool.paired(_a, "obsign_t1e3", reading="rehearsed", cells=cells, ref=ref)
    _stem = "ExtTunedVsOurs" + ARM_TEX[_a]
    emit(_stem + "Mean", s2(_d["mean"]))
    emit(_stem + "Lo", s2(_d["lo"]))
    emit(_stem + "Hi", s2(_d["hi"]))
    emit(_stem + "Separated", "yes" if _d["separated"] else "no")
    emit(_stem + "Wins", str(_d["wins"]))

req = pool.requirement_check(cells, ref)
emit("ExtReqWeakCount", str(len(req["weak"])))
emit("ExtReqStrongCount", str(len(req["strong"])))
emit("ExtReqStrongArms", ", ".join(ARM_LABEL[a] for a in req["strong"]))
# OURS, excluded by construction rather than by a list of names: every ObSign arm (whatever
# tau it was run at) plus its ablation. A hand-written list here would have silently counted
# the three tau arms added on 2026-09-04 as EXISTING methods -- they all fail R3(a) at the
# shared default, so the count is 0 either way, but only by luck.
from stage0_figs import TAU_OF as _TAU_ARMS     # one definition of "which arm is which tau"
_ours = set(_TAU_ARMS) | {"relsign"}
_zero_strong = [a for a in req["strong"]
                if req["stats"][a]["state"] < 0.005 and a not in _ours]
emit("ExtReqStrongZeroStateExistingCount", str(len(_zero_strong)))
_free = pool.lrfree(cells=cells, ref=ref)
_fb = req["ref"]
emit("ExtFreeNoHarmCount", str(sum(1 for s in _free.values() if s["neg"] == 0)))
# BOTH parts of R3.  Kept under its original name because the paper's sentence is about the
# conjunction; the R3(b)-only count is emitted separately because writing "the number
# satisfying R3(b)" next to a conjunction count is how a macro's name stops matching what it
# measures (Prodigy fails R3(a) but its interval does overlap the reference).
emit("ExtFreeCompetitiveCount",
     str(sum(1 for s in _free.values()
             if s["neg"] == 0 and s["hi"] >= _fb["lo"] and _fb["hi"] >= s["lo"])))
# The learning-rate-free rules are read at their own defaults, so their shortfall against the
# tuned frontier needs the same treatment as the grid arms; section IV-B quotes it.
for _a in pool.LRFREE_ARMS:
    _g = pool.paired(pool.COMPETITIVE_REF, _a, reading="lrfree", cells=cells, ref=ref)
    if _g.get("n"):
        emit("ExtGap" + ARM_TEX[_a] + "Mean", s2(_g["mean"]))
        emit("ExtGap" + ARM_TEX[_a] + "Lo", s2(_g["lo"]))
        emit("ExtGap" + ARM_TEX[_a] + "Hi", s2(_g["hi"]))
emit("ExtFreeCompetitiveOnlyCount",
     str(sum(1 for s in _free.values() if s["hi"] >= _fb["lo"] and _fb["hi"] >= s["lo"])))
emit("ExtFreeArmCount", str(len(_free)))

section("Per-rate readings for the two arms the argument compares directly. Section IV says "
        "AdaFactor's untuned success is a property of WHERE its default sits, and section V "
        "says ObSign's reading does not move above the knee; both are claims about specific "
        "cells of the band table, so the prose needs those cells as macros. Without these the "
        "text has to borrow a macro that happens to have the right value -- the exact way a "
        "number keeps its name while changing its meaning.")
RATE_TEX = {3e-6: "ThreeESix", 1e-5: "OneEFive", 3e-5: "ThreeEFive", 1e-4: "OneEFour",
            3e-4: "ThreeEFour", 1e-3: "OneEThree", 3e-3: "ThreeEThree", 1e-2: "OneETwo",
            3e-2: "ThreeETwo", 1e-1: "OneEOne"}
for _arm in ("adafactor", "obsign_t1e3"):
    for _lr, _tag in RATE_TEX.items():
        _s = pool.fixed([_arm], _lr, cells, ref).get(_arm)
        if not _s:
            continue
        emit("ExtRate" + ARM_TEX[_arm] + _tag + "Mean", s2(_s["mean"]))
        emit("ExtRate" + ARM_TEX[_arm] + _tag + "Neg", str(_s["neg"]))

section("Why ObSign's TUNED reading sits below signSGD's. A fixed tau caps the step, so the "
        "rule can reach signSGD's behaviour only at rates BELOW the knee: it is a subset of "
        "signSGD's reachable steps, not a superset. The count below is the mechanism -- the "
        "cells whose signSGD optimum lies above the knee are exactly the cells ObSign cannot "
        "reach.")
from stage0_figs import RMS_REF as _RMS       # also imported below for the knee section
_knee_ref = 1e-3 * _RMS                        # the knee of the recommended arm, tau = 1e-3
_above = _tot = 0
for _k, _r in cells:
    _sw = pool._sweep("signsgd", _r, ref[_k])
    if not _sw:
        continue
    _tot += 1
    _above += sel_oracle(_sw)[0] > _knee_ref
emit("ExtSignsgdOptAboveKneeCells", str(_above))
emit("ExtSignsgdOptCells", str(_tot))

section("Guard ablation across the grid. At the 1e-3 default the guarded and unguarded rules "
        "are identical to two decimals -- the knee sits below it, so the cap is inactive there "
        "and that single column cannot separate the components. The cap is what the rule does "
        "everywhere ELSE on the grid.")
GUARD_LR_TEX = {3e-6: "ThreeESix", 1e-5: "OneEFive", 3e-5: "ThreeEFive", 1e-4: "OneEFour",
                3e-4: "ThreeEFour", 1e-3: "OneEThree", 3e-3: "ThreeEThree", 1e-2: "OneETwo",
                3e-2: "ThreeETwo", 1e-1: "OneEOne"}
for _lr, _tag in GUARD_LR_TEX.items():
    g = pool.fixed(["obsign_t1e3"], _lr, cells, ref).get("obsign_t1e3")
    u = pool.fixed(["relsign"], _lr, cells, ref).get("relsign")
    if not g or not u:
        continue
    emit("ExtGuardOn" + _tag, s2(g["mean"]))
    emit("ExtGuardOff" + _tag, s2(u["mean"]))
    emit("ExtGuardOffNeg" + _tag, str(u["neg"]))
    emit("ExtGuardDelta" + _tag, s2(g["mean"] - u["mean"]))

section("Section V needs the knee POSITION, not just tau. The knee is tau*RMS(p), so it is "
        "a measured quantity: RMS_REF is the median RMS over trainable tensors at "
        "initialisation (stage0_figs.RMS_REF, with the reproducing command in that module). "
        "Quoting tau alone would leave the reader unable to compare the knee against the "
        "shipped rate, which is the whole argument of the section.")
from stage0_figs import RMS_REF, DEPLOYED_LR, TAU_OF
_KNEE = {a: (t, ARM_TEX[a]) for a, t in TAU_OF.items() if a in fix or a == "obsign"}
emit("ExtRmsRef", f"{RMS_REF:g}")
emit("ExtDeployedLr", lrtex(DEPLOYED_LR))
for _arm, (_tau, _tex) in _KNEE.items():
    # tau is quoted INSIDE math in the prose ($\tau = \ExtTau...$), so this one stays a bare
    # number; every other rate-valued macro is self-contained math.
    emit("ExtTau" + _tex, f"{_tau:g}")
    emit("ExtKnee" + _tex, lrtex(round(_tau * RMS_REF, 12)))
    # decades between the knee and the rate the deployment ships -- the quantity the design
    # rule for tau is stated in, and the x axis of the margin reading below.
    emit("ExtKneeDecadesBelowDefault" + _tex, f"{np.log10(DEPLOYED_LR / (_tau * RMS_REF)):.2f}")

section("The design rule for tau, read as a MARGIN. The rule is 'put the knee at least a "
        "decade below the rate you ship', so the quantity that decides pass or fail is the "
        "distance in decades between tau*RMS(p) and the shipped rate -- not tau itself. Stage "
        "0c sampled that distance at three points (1.28 / 0.80 / 0.28 decades) and could only "
        "bound the crossing between the first two; Stage 0d added 1.10, 0.98 and 0.58. These "
        "macros are the sweep, ordered by margin, and the pass/fail column is deployable() at "
        "the shipped rate, so it cannot drift away from Table I.")
_tau_arms = sorted([a for a in TAU_OF if a in fix],
                   key=lambda a: -np.log10(DEPLOYED_LR / (TAU_OF[a] * RMS_REF)))
_tau_dep = pool.deployable(_tau_arms, cells, ref)
emit("ExtTauSweepCount", str(len(_tau_arms)))
for _a in _tau_arms:
    _m = np.log10(DEPLOYED_LR / (TAU_OF[_a] * RMS_REF))
    _row = [r for r in _tau_dep[_a]["rows"] if abs(r["lr"] - DEPLOYED_LR) < 1e-12][0]
    _stem = "ExtMargin" + ARM_TEX[_a]
    emit(_stem + "Decades", f"{_m:.2f}")
    emit(_stem + "Mean", s2(fix[_a]["mean"]))
    emit(_stem + "Neg", str(fix[_a]["neg"]))
    emit(_stem + "Deployable", "yes" if _row["ok"] else "no")
# The boundary the rule sits on: the smallest margin that is still deployable at the shipped
# rate, and the largest that is not.  Together they are what "at least one decade" is measured
# against, and they are computed rather than asserted -- if a rerun moves the crossing, the
# sentence in section V moves with it.
_ok = [np.log10(DEPLOYED_LR / (TAU_OF[a] * RMS_REF)) for a in _tau_arms
       if [r for r in _tau_dep[a]["rows"] if abs(r["lr"] - DEPLOYED_LR) < 1e-12][0]["ok"]]
_bad = [np.log10(DEPLOYED_LR / (TAU_OF[a] * RMS_REF)) for a in _tau_arms
        if not [r for r in _tau_dep[a]["rows"] if abs(r["lr"] - DEPLOYED_LR) < 1e-12][0]["ok"]]
if _ok and _bad:
    emit("ExtMarginPassMin", f"{min(_ok):.2f}")
    emit("ExtMarginFailMax", f"{max(_bad):.2f}")
    emit("ExtMarginBracketWidth", f"{min(_ok) - max(_bad):.2f}")
else:
    warnings.append("tau margin sweep has no pass/fail crossing -- margin macros omitted")

# HOW the margin rule fails when it is broken, not just that it fails. A reviewer reading
# "10 of 216 cells" cannot tell whether one pathological dataset carries it or whether the
# guard is coming off everywhere, and those two readings would justify different rules.
_NAMES = {"ETTh1": "ETTh1", "ETTh2": "ETTh2", "ETTm1": "ETTm1", "ETTm2": "ETTm2",
          "appliances": "the appliances series", "bdg2": "the building corpus"}
for _a in _tau_arms:
    _neg = [k for k, _r in cells
            if (pool._sweep(_a, _r, ref[k]).get(f"{DEPLOYED_LR:g}", {}).get("benefit", 0)
                or 0) < -0.05]
    _ds = sorted({k[0] for k in _neg})
    _stem = "ExtMargin" + ARM_TEX[_a]
    emit(_stem + "NegDatasets", str(len(_ds)))
    emit(_stem + "NegDatasetNames",
         ", ".join(_NAMES[d] for d in _ds[:-1]) + (" and " if len(_ds) > 1 else "")
         + (_NAMES[_ds[-1]] if _ds else "none"))
    emit(_stem + "Worst", s1(fix[_a]["worst"]))

section("Selecting tau on the held-out slice INSTEAD of following the design rule "
        "(stage0_pool.tau_selected). The objection the design rule invites is that a "
        "deployment with a validation slice could just select tau on it, so the paper measures "
        "that rather than leaving it as an option -- and the answer is not the flattering one: "
        "the selection reaches for the least-guarded tau, gains a little mean and loses the "
        "no-harm property that section V is selling.")
_ts = pool.tau_selected(cells=cells, ref=ref)
if _ts:
    emit("ExtTauValSelTaus", str(_ts["n_taus"]))
    emit("ExtTauValSelMean", s2(_ts["mean"]))
    emit("ExtTauValSelNeg", str(_ts["neg"]))
    emit("ExtTauValSelWorst", s1(_ts["worst"]))
    emit("ExtTauValSelPickedLargest", str(_ts["picked_largest"]))
    emit("ExtTauValSelLargestArm", ARM_LABEL[_ts["largest_arm"]])
else:
    warnings.append("tau_selected() returned nothing -- validation-selection macros omitted")
# What one decade of tau costs, at the shipped default -- the honest sensitivity number the
# limitations section has to carry.
emit("ExtTauOneDecadeCost",
     s2(fix["obsign_t1e3"]["mean"] - pool.fixed(["obsign"], 1e-3, cells, ref)["obsign"]["mean"]))

section("Derived claims the text makes (kept as macros so a rerun cannot leave prose stale)")
zero_state = [a for a in reh if reh[a]["state"] < 0.005]
best_zero = max(zero_state, key=lambda a: reh[a]["mean"])
emit("ExtBestZeroStateArm", ARM_LABEL[best_zero])
emit("ExtBestZeroStateMean", s2(reh[best_zero]["mean"]))
emit("ExtSignsgdMinusObsignMisOne", f2(reh["signsgd"]["mis1x"] - reh["obsign_t3e3"]["mis1x"]))
emit("ExtObsignMinusSignsgdMean", s2(reh["obsign_t3e3"]["mean"] - reh["signsgd"]["mean"]))
emit("ExtFixAdamNegPct", f"{100 * fix['adam']['neg'] / fix['adam']['n_all']:.0f}")
emit("ExtFixSgdmNegPct", f"{100 * fix['sgdm']['neg'] / fix['sgdm']['n_all']:.0f}")
emit("ExtMacrosDate", datetime.date.today().isoformat())

header = ["% AUTO-GENERATED by experiments/tsf_edge/gen_macros_stage0.py -- DO NOT EDIT.",
          "% Regenerate: .venv/bin/python experiments/tsf_edge/gen_macros_stage0.py",
          "% Extension-paper macros only; the BigData macros live in macros.tex and the two",
          "% namespaces are disjoint (everything here starts with \\Ext).",
          "% Percents carry an explicit +/- sign; values are bare numbers (append \\% in prose)."]
header += [f"% WARNING: {w}" for w in warnings]
with open(OUTFILE, "w") as f:
    f.write("\n".join(header) + "\n" + "\n".join(lines) + "\n")
print(f"wrote {OUTFILE}: {len(seen)} macros")
for w in warnings:
    print(f"WARNING: {w}")
