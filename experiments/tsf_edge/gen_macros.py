"""Single-source-of-truth numbers: generate results/tsf_edge/macros.tex from the RESULT DATA
FILES so no number in the paper is ever hand-copied. Rerun after any experiment rerun; the
paper does `\\input{macros.tex}` and cites only macros.

Sources (missing optional files are skipped with a warning):
  grid.jsonl                 C3 grid (required)  — benefit% sign: >0 = adaptation BETTER
                                                   (fixed default online LR 1e-3: the confound)
  lr_fairness.jsonl          M1 LR-fairness      — benefit% sign: >0 = adaptation BETTER;
                                                   readings Fixed(@1e-3)/Sel(val-rehearsed)/Orc
  frontier_data.json         C2 frontier         — benefit% sign: >0 = adaptation BETTER
                                                   (fair LR; BenefitFixed = old fixed-1e-3)
  *_sgdm.json                all C1/staleness/M6/timing artifacts: the SGD-family arm is
                             SGD+MOMENTUM (online_eval.SGD_STRAT); the momentum-free files are
                             kept next to them but are no longer read
  staleness_patchtst_full_sgdm.json  staleness    — win% sign: >0 = drift-trigger BETTER
  staleness_patchtst_full_adam.json               — full-Adam variant (StalAdam* macros)
  leakage_check.json         C1b (optional)      — benefit% sign: >0 = adaptation BETTER;
                                                   inflation pt = leaky - clean
  warmup_confound.json       C1a (optional)      — benefit% sign: <0 = adaptation BETTER
                                                   ((adapted-static)/static, as in FINDINGS Table 1)
  validation_protocol.json   C1c (optional)      — improvement% sign: >0 = adaptation BETTER

Macro names are letters-only (digits spelled out): \\GridSgdFloor, \\FroEttmTwoPatchtstCalibSgdBenefit.
Values are bare numbers (append \\% etc. in prose); percents carry an explicit +/- sign.
"""
from __future__ import annotations
import datetime, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, "results", "tsf_edge")
OUT = os.path.join(RES, "macros.tex")
sys.path.insert(0, HERE)
import combined_grid as cg                                  # expected cell count stays in sync
from online_eval import P_EDGE_W, WARM_GRID                 # single source for energy proxy / cap
from frontier_timing import load_timing

TIMING = load_timing()

# R3: the paper's SGD-family arm is SGD WITH MOMENTUM (see online_eval.SGD_STRAT). SGDF is the
# key it is stored under in the result files; SGDN is how it is spelled inside macro names, so
# a reader of the paper can never mistake a \Lr...Sgdm... value for momentum-free SGD. The
# momentum-free surfaces stay in the files and can be re-emitted by flipping these two.
SGDF, SGDN = "sgdm", "Sgdm"

DIG = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
       "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}


def texname(*parts):
    words = [w for p in parts for w in re.split(r"[^0-9A-Za-z]+", str(p)) if w]
    s = "".join(w[:1].upper() + w[1:].lower() for w in words)
    return "".join(DIG.get(ch, ch) for ch in s)


def s1(x, nd=1):                       # signed, nd decimals; round-then-add-0.0 avoids "-0.0"
    return f"{round(x, nd) + 0.0:+.{nd}f}"
def ki(x): return f"{int(x):,}"        # thousands separator (IEEE style: 20,000 not 20000)
def f1(x): return f"{x:.1f}"
def f2(x): return f"{x:.2f}"
def f3(x): return f"{x:.3f}"
def f4(x): return f"{x:.4f}"


lines, seen, warnings = [], set(), []


def emit(name, val):
    assert re.fullmatch(r"[A-Za-z]+", name), f"bad macro name: {name}"
    assert name not in seen, f"duplicate macro: {name}"
    seen.add(name)
    lines.append(f"\\newcommand{{\\{name}}}{{{val}}}")


def section(title):
    lines.append(f"\n% ---- {title} ----")


def load_optional(fname):
    p = os.path.join(RES, fname)
    if not os.path.exists(p):
        warnings.append(f"missing {fname} (rerun its script to include these macros)")
        return None
    return json.load(open(p))


# ---------- C3 grid ----------
rows = []
for line in open(os.path.join(RES, "grid.jsonl")):
    try:
        rows.append(json.loads(line))
    except json.JSONDecodeError:                            # partial last line of a running grid
        pass
expected = len(cg.DATASETS) * len(cg.BACKBONES) * len(cg.HS) * len(cg.LS) * len(cg.SEEDS)
if len(rows) != expected:
    warnings.append(f"grid.jsonl has {len(rows)}/{expected} cells — PARTIAL, regenerate when done")

# R3: grid.jsonl's SGD column is momentum-FREE SGD, which the paper no longer reports. Its
# Adam column and its P1/P2/P3 probes (computed on the frozen model, optimizer-independent) are
# unaffected, so rather than re-running combined_grid.py we substitute the SGD-family benefit
# from the fair-LR grid at the same fixed 1e-3 rate. That substitution is exact by construction:
# lr_fairness@1e-3 was verified to reproduce grid.jsonl cell-for-cell to floating point.
_lrf_at_default = {}
for _line in open(os.path.join(RES, "lr_fairness.jsonl")):
    try:
        _r = json.loads(_line)
    except json.JSONDecodeError:
        continue
    if SGDF in _r and "0.001" in _r[SGDF]:
        _lrf_at_default[(_r["dataset"], _r["backbone"], _r["L"], _r["H"], _r["seed"])] = \
            _r[SGDF]["0.001"]["benefit"]
_missing = 0
for _r in rows:
    _k = (_r["dataset"], _r["backbone"], _r["L"], _r["H"], _r["seed"])
    if _k in _lrf_at_default:
        _r["benefit_sgd"] = _lrf_at_default[_k]
    else:
        _missing += 1
if _missing:
    warnings.append(f"C3 grid: {_missing} cells have no {SGDF}@1e-3 reading — still momentum-free")

def winner(r): return "SGD" if r["benefit_sgd"] >= r["benefit_adam"] else "Adam"
                            # "SGD" here = the SGD-family arm, i.e. SGD+momentum

section("C3 grid (grid.jsonl); benefit% >0 = adaptation better than static")
bs = [r["benefit_sgd"] for r in rows]
ba = [r["benefit_adam"] for r in rows]
emit("GridCells", len(rows))
emit("GridSeeds", len({r["seed"] for r in rows}))
emit("GridDatasets", len({r["dataset"] for r in rows}))
emit("Grid" + SGDN + "Floor", s1(min(bs)))
emit("Grid" + SGDN + "FloorExact", s1(min(bs), nd=2))         # +0.03 before the 1-dp rounding (minor 4)
emit("Grid" + SGDN + "NearZeroCells", sum(abs(b) < 0.05 for b in bs))
emit("Grid" + SGDN + "NegCells", sum(b < 0 for b in bs))
emit("GridAdamWorst", s1(min(ba)))
emit("GridAdamNegCells", sum(b < 0 for b in ba))
emit("GridAdamNegPct", round(100 * sum(b < 0 for b in ba) / len(ba)))
emit("Grid" + SGDN + "Wins", sum(winner(r) == "SGD" for r in rows))
emit("GridAdamWins", sum(winner(r) == "Adam" for r in rows))

configs = {}
for r in rows:
    configs.setdefault((r["dataset"], r["backbone"], r["L"], r["H"]), []).append(r)
uni = [v for v in configs.values() if len({winner(r) for r in v}) == 1]
flip = [v for v in configs.values() if len({winner(r) for r in v}) > 1]
emit("GridConfigs", len(configs))
emit("GridUnanimousConfigs", len(uni))
emit("GridFlipConfigs", len(flip))


def _mean_gap(cfgs):                        # per-config mean |SGD-Adam| margin, averaged
    gs = [sum(abs(r["benefit_sgd"] - r["benefit_adam"]) for r in v) / len(v) for v in cfgs]
    return sum(gs) / len(gs)


if flip and uni:
    emit("GridFlipMeanGapPt", f1(_mean_gap(flip)))
    emit("GridUnanimousMeanGapPt", f1(_mean_gap(uni)))
emit("GridWarmCapStep", ki(max(WARM_GRID)))
emit("GridWarmCapCells", sum(r["warmup"] == max(WARM_GRID) for r in rows))

for probe, fmt in [("p3_drift", f2), ("p2_gradcos", f2), ("p1_noise", f3)]:
    win_s = [r[probe] for r in rows if winner(r) == "SGD"]
    win_a = [r[probe] for r in rows if winner(r) == "Adam"]
    base = texname("Grid", probe.split("_")[0])
    emit(base + SGDN + "WinMean", fmt(sum(win_s) / len(win_s)))
    emit(base + "AdamWinMean", fmt(sum(win_a) / len(win_a)))
    emit(base + "Gap", s1(sum(win_a) / len(win_a) - sum(win_s) / len(win_s),
                          nd=1 if probe == "p3_drift" else 2))

for ds in sorted({r["dataset"] for r in rows}):
    sub = [r for r in rows if r["dataset"] == ds]
    b = texname("Grid", ds)
    emit(b + "Cells", len(sub))
    emit(b + "PThree", f2(sum(r["p3_drift"] for r in sub) / len(sub)))
    emit(b + "POne", f3(sum(r["p1_noise"] for r in sub) / len(sub)))
    emit(b + SGDN + "Wins", sum(winner(r) == "SGD" for r in sub))
    emit(b + "AdamWins", sum(winner(r) == "Adam" for r in sub))
    emit(b + "AdamNegCells", sum(r["benefit_adam"] < 0 for r in sub))

for L in sorted({r["L"] for r in rows}):                    # lookback robustness (L=192 shrinks
    sub = [r for r in rows if r["L"] == L]                  # the Adam-favourable corner)
    b = texname("Grid", "L", L)
    emit(b + "Cells", len(sub))
    emit(b + SGDN + "Wins", sum(winner(r) == "SGD" for r in sub))
    emit(b + "AdamWins", sum(winner(r) == "Adam" for r in sub))
    emit(b + "AdamNegCells", sum(r["benefit_adam"] < 0 for r in sub))
    emit(b + "AdamNegPct", round(100 * sum(r["benefit_adam"] < 0 for r in sub) / len(sub)))
    emit(b + SGDN + "Floor", s1(min(r["benefit_sgd"] for r in sub)))
    emit(b + "AdamWorst", s1(min(r["benefit_adam"] for r in sub)))

for ds in sorted({r["dataset"] for r in rows}):             # dataset x L cells (cited: Appliances)
    for L in sorted({r["L"] for r in rows}):
        sub = [r for r in rows if r["dataset"] == ds and r["L"] == L]
        b = texname("Grid", ds, "L", L)
        emit(b + "Cells", len(sub))
        emit(b + SGDN + "Wins", sum(winner(r) == "SGD" for r in sub))
        emit(b + "AdamWins", sum(winner(r) == "Adam" for r in sub))
        emit(b + "AdamNegCells", sum(r["benefit_adam"] < 0 for r in sub))

# ---------- M1 LR-fairness grid ----------
lrf_path = os.path.join(RES, "lr_fairness.jsonl")
if os.path.exists(lrf_path):
    lrf_all = [json.loads(l) for l in open(lrf_path)]
    core = set(cg.DATASETS)                 # exclude M5 extras (bdg2_fox etc.) from C3 stats
    lrf = [r for r in lrf_all if r["dataset"] in core]
    section("M1 LR-fairness (lr_fairness.jsonl); benefit% >0 = adaptation better; readings: "
            "Fixed = @1e-3 default, Sel = val-rehearsed LR, Orc = test-oracle LR; "
            "Lr* = the FULL fair-LR grid (6 ds x 2 bb x H in {24,48,96} x L in {96,192} x 5 seeds)")
    LRNAME = {3e-06: "ThreeEMinusSix", 1e-05: "OneEMinusFive", 3e-05: "ThreeEMinusFive",
              1e-04: "OneEMinusFour", 3e-04: "ThreeEMinusFour", 1e-03: "OneEMinusThree",
              3e-03: "ThreeEMinusThree", 1e-02: "OneEMinusTwo", 3e-02: "ThreeEMinusTwo",
              1e-01: "OneEMinusOne"}
    # rates present in EVERY cell (both optimizers) — during an in-flight grid extension some
    # cells have the new rates and some do not; pooled per-LR stats must only use the common set
    common_lrs = sorted(set.intersection(
        *[{float(k) for k in r[SGDF]} & {float(k) for k in r["adam"]} for r in lrf]))
    emit("LrCells", len(lrf))
    emit("LrGridPoints", len(common_lrs))
    emit("LrSeeds", len({r["seed"] for r in lrf}))
    readings = {"Fixed": lambda r, o: r[o]["0.001"]["benefit"],
                "Sel":   lambda r, o: r[f"sel_benefit_{o}"],
                "Orc":   lambda r, o: r[f"oracle_benefit_{o}"]}
    def _median(xs):
        xs = sorted(xs)
        n = len(xs)
        return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2

    DIV = -100.0                # a benefit below -100% (or nan/-inf) = a diverged stream;
                                # counted as negative, excluded from worst-value/mean stats
    def _negstats(vals):
        bounded = [v for v in vals if v == v and v >= DIV]
        div = len(vals) - len(bounded)
        return div + sum(v < 0 for v in bounded), div, min(bounded)

    def _cfg_stats(sub, get):
        """Seed-majority config-level wins (a config = dataset x backbone x L x H), guarding
        against the seeds-as-independent-samples reading of the cell counts (referee minor)."""
        cfgs = {}
        for r in sub:
            cfgs.setdefault((r["dataset"], r["backbone"], r["L"], r["H"]), []).append(r)
        aw = sum(sum(get(r, "adam") > get(r, SGDF) for r in v) > len(v) / 2
                 for v in cfgs.values())
        unan = sum(sum(get(r, "adam") > get(r, SGDF) for r in v) in (0, len(v))
                   for v in cfgs.values())
        return len(cfgs), aw, unan

    for Lv in sorted({r["L"] for r in lrf}):                # per-lookback three-reading stats
        sub = [r for r in lrf if r["L"] == Lv]
        base = texname("Lr", "L", Lv)
        emit(base + "Cells", len(sub))
        for rd, get in readings.items():
            b_s = [get(r, SGDF) for r in sub]
            b_a = [get(r, "adam") for r in sub]
            b = base + rd
            emit(b + SGDN + "Wins", sum(s >= a for s, a in zip(b_s, b_a)))
            emit(b + "AdamWins", sum(a > s for s, a in zip(b_s, b_a)))
            s_neg, s_div, s_worst = _negstats(b_s)
            a_neg, a_div, a_worst = _negstats(b_a)
            emit(b + SGDN + "NegCells", s_neg)
            emit(b + SGDN + "DivCells", s_div)
            emit(b + SGDN + "Min", s1(s_worst))
            emit(b + "AdamNegCells", a_neg)
            emit(b + "AdamDivCells", a_div)
            emit(b + "AdamNegPct", round(100 * a_neg / len(b_a)))
            emit(b + "AdamMin", s1(a_worst))
            gaps = [a - s for s, a in zip(b_s, b_a)]
            bounded_gaps = [g for g in gaps if abs(g) <= -DIV]   # excl. diverged-SGD cells
            emit(b + "MeanGapPt", s1(sum(bounded_gaps) / len(bounded_gaps)))
            emit(b + "MedianGapPt", s1(_median(gaps)))
            ncfg, aw, unan = _cfg_stats(sub, get)
            emit(b + "CfgAdamWins", aw)
            emit(b + "Cfg" + SGDN + "Wins", ncfg - aw)
            emit(b + "CfgUnanimous", unan)
    for lr in common_lrs:                                   # pooled per-LR plateau statistics
        for o in (SGDF, "adam"):
            vals = [r[o][f"{lr:g}"]["benefit"] for r in lrf]
            fin = [v for v in vals if v == v]                # a NaN mse (diverged stream at an
            b = texname("Lr", o) + "At" + LRNAME[lr]         # extreme rate) counts as negative;
            emit(b + "NegCells", sum(v < 0 for v in fin) + len(vals) - len(fin))
            emit(b + "Mean", s1(sum(fin) / len(fin)))        # mean/min are over finite cells
            emit(b + "Min", s1(min(fin)))
        # head-to-head at this SHARED fixed rate (cited: the fixed-1e-4 ranking reversal); a
        # NaN benefit (diverged stream) never wins, matching the NegCells convention above
        b_s = [r[SGDF][f"{lr:g}"]["benefit"] for r in lrf]
        b_a = [r["adam"][f"{lr:g}"]["benefit"] for r in lrf]
        emit("LrAdamWinsAt" + LRNAME[lr], sum(a > s for s, a in zip(b_s, b_a)))
    # selection behaviour (pooled over all cells)
    emit("LrAdamSelLeqThreeEMinusFourCells", sum(r["sel_lr_adam"] <= 3e-4 for r in lrf))
    emit("LrAdamSelGeqOneEMinusThreeCells", sum(r["sel_lr_adam"] >= 1e-3 for r in lrf))
    emit("Lr" + SGDN + "SelGeqOneEMinusThreeCells", sum(r[f"sel_lr_{SGDF}"] >= 1e-3 for r in lrf))
    s_neg, s_div, s_worst = _negstats([r[f"sel_benefit_{SGDF}"] for r in lrf])
    a_neg, a_div, a_worst = _negstats([r["sel_benefit_adam"] for r in lrf])
    emit("Lr" + SGDN + "SelNegCellsAll", s_neg)
    emit("Lr" + SGDN + "SelDivCellsAll", s_div)
    emit("Lr" + SGDN + "SelMinAll", s1(s_worst))
    emit("LrSelAdamNegCellsAll", a_neg)
    emit("LrSelAdamMinAll", s1(a_worst))
    # R1 grid-top extension: selection/bracketing behaviour at the two added rates (3e-2, 1e-1)
    top = max(common_lrs)
    below_top = common_lrs[-2]
    emit("Lr" + SGDN + "SelExtCells", sum(r[f"sel_lr_{SGDF}"] >= 3e-2 for r in lrf))
    emit("LrAdamSelExtCells", sum(r["sel_lr_adam"] >= 3e-2 for r in lrf))
    emit("Lr" + SGDN + "OrcAtTopCells", sum(r[f"oracle_lr_{SGDF}"] == top for r in lrf))
    _top_rise = [r[SGDF][f"{top:g}"]["benefit"] - r[SGDF][f"{below_top:g}"]["benefit"]
                 for r in lrf if r[f"oracle_lr_{SGDF}"] == top]
    if _top_rise:
        emit("Lr" + SGDN + "OrcTopRiseMaxPt", s1(max(_top_rise)))
    sel_neg = [r for r in lrf if not (r[f"sel_benefit_{SGDF}"] >= 0)]
    emit("Lr" + SGDN + "SelNegAppliancesDlinearCells",
         sum(r["dataset"] == "appliances" and r["backbone"] == "dlinear" for r in sel_neg))
    # by how much did the picks that go on to DIVERGE win their rehearsals? (the val slice
    # sees only a marginal advantage where the test stream later explodes)
    _div_margin = [100 * (1 - r[SGDF][f"{r[f'sel_lr_{SGDF}']:g}"]["val"] / r[SGDF]["0.001"]["val"])
                   for r in sel_neg if not (r[f"sel_benefit_{SGDF}"] >= DIV)]
    if _div_margin:
        emit("Lr" + SGDN + "SelDivValMarginMaxPct", f1(max(_div_margin)))
    # the no-free-fix check: a conservative rule (smallest rate within 2% of the best val
    # MSE) repairs the diverged picks but costs benefit almost everywhere else
    def _tol_sel(r, o, tol=0.02):
        grid = sorted(float(k) for k in r[o])
        best = min(r[o][f"{lr:g}"]["val"] for lr in grid
                   if r[o][f"{lr:g}"]["val"] == r[o][f"{lr:g}"]["val"])
        return next(lr for lr in grid if r[o][f"{lr:g}"]["val"] <= (1 + tol) * best)
    deltas = []
    for r in lrf:
        g = _tol_sel(r, SGDF)
        if g != r[f"sel_lr_{SGDF}"]:
            deltas.append(r[SGDF][f"{g:g}"]["benefit"] - r[f"sel_benefit_{SGDF}"])
    emit("LrTolGuardChangedCells", len(deltas))
    emit("LrTolGuardMedianCostPt", f1(-_median([d for d in deltas if abs(d) <= -DIV])))
    # skeptic reading: deployable (rehearsed) Adam vs SGD's unattainable per-cell test-oracle
    emit("LrSelAdamVsOrc" + SGDN + "Wins",
         sum(r["sel_benefit_adam"] > r[f"oracle_benefit_{SGDF}"] for r in lrf))
    emit("LrSelAdamVsOrc" + SGDN + "MedianPt", s1(_median(
        [r["sel_benefit_adam"] - r[f"oracle_benefit_{SGDF}"] for r in lrf]), nd=2))
    # pooled three-reading win counts (abstract/intro cite these)
    for rd, get in readings.items():
        b_s = [get(r, SGDF) for r in lrf]
        b_a = [get(r, "adam") for r in lrf]
        emit("Lr" + rd + SGDN + "WinsAll", sum(s >= a for s, a in zip(b_s, b_a)))
        emit("Lr" + rd + "AdamWinsAll", sum(a > s for s, a in zip(b_s, b_a)))
        gaps = [a - s for s, a in zip(b_s, b_a)]
        bounded_gaps = [g for g in gaps if abs(g) <= -DIV]     # excl. diverged-SGD cells
        emit("Lr" + rd + "MeanGapPtAll", s1(sum(bounded_gaps) / len(bounded_gaps)))
        emit("Lr" + rd + "MedianGapPtAll", s1(_median(gaps)))
        ncfg, aw, unan = _cfg_stats(lrf, get)
        emit("Lr" + rd + "CfgAdamWinsAll", aw)
        emit("Lr" + rd + "Cfg" + SGDN + "WinsAll", ncfg - aw)
        emit("Lr" + rd + "CfgUnanimousAll", unan)
        if rd == "Sel":                     # two-sided binomial sign test on the config-level
            from math import comb           # winners (guards the seeds-as-samples objection)
            k = max(aw, ncfg - aw)
            p = min(1.0, sum(comb(ncfg, i) for i in range(k, ncfg + 1)) / 2 ** (ncfg - 1))
            e = 0
            while p < 1:                    # 1-sig-fig scientific form for use in math mode
                p *= 10; e -= 1
            emit("LrSelCfgSignPAll", f"{round(p)}{{\\times}}10^{{{e}}}" if e else f1(p))
            # (R3) the 72 configs are NESTED -- 6 series x 2 backbones x 3 H x 2 L -- so a sign
            # test that treats them as independent overstates its evidence. Repeat it on the two
            # coarser partitions whose members ARE distinct series/models, and report those.
            for lvl, keyf in (("DsBb", lambda r: (r["dataset"], r["backbone"])),
                              ("Ds", lambda r: r["dataset"])):
                cl = {}
                for r in lrf:
                    cl.setdefault(keyf(r), []).append(get(r, "adam") > get(r, SGDF))
                won = sum(sum(v) * 2 > len(v) for v in cl.values())
                n = len(cl)
                emit("LrSelCluster" + lvl + "Adam", won)
                emit("LrSelCluster" + lvl + "Total", n)
                kk = max(won, n - won)
                pc = min(1.0, sum(comb(n, i) for i in range(kk, n + 1)) / 2 ** (n - 1))
                if pc >= 0.01:                       # plain decimal while it is readable
                    emit("LrSelCluster" + lvl + "P", f2(pc))
                else:                                # else 1-sig-fig scientific, math mode
                    ee = 0
                    while pc < 1:
                        pc *= 10; ee -= 1
                    emit("LrSelCluster" + lvl + "P", f"{round(pc)}{{\\times}}10^{{{ee}}}")
            # (R3) "unanimous" counted BOTH directions; split it so the text cannot be misread
            # as "53 of the Adam-winning configs".
            cfgs = {}
            for r in lrf:
                cfgs.setdefault((r["dataset"], r["backbone"], r["L"], r["H"]), []).append(
                    get(r, "adam") > get(r, SGDF))
            emit("LrSelCfgUnanimousAdam", sum(all(v) for v in cfgs.values()))
            emit("LrSelCfgUnanimous" + SGDN, sum(not any(v) for v in cfgs.values()))
    emit("LrConfigs", len({(r["dataset"], r["backbone"], r["L"], r["H"]) for r in lrf}))
    for Hv in sorted({r["H"] for r in lrf}):                # per-horizon robustness (compact)
        sub = [r for r in lrf if r["H"] == Hv]
        b = texname("Lr", "H", Hv)
        emit(b + "Cells", len(sub))
        emit(b + "SelAdamWins", sum(r["sel_benefit_adam"] > r[f"sel_benefit_{SGDF}"] for r in sub))
        emit(b + "SelAdamNegCells", sum(r["sel_benefit_adam"] < 0 for r in sub))
    # M5: BDG2 extension subsets (fair-LR H24/L96/3-seed cells; NOT part of the C3 stats)
    def _val_chosen(r):
        """Fully deployable per-seed reading (referee N1): the OPTIMIZER, like its rate, is
        chosen by validation online MSE (each optimizer at its own rehearsed rate); the test
        benefit of that choice is reported. No test data enters any selection."""
        v_s = r[SGDF][f"{r[f'sel_lr_{SGDF}']:g}"]["val"]
        v_a = r["adam"][f"{r['sel_lr_adam']:g}"]["val"]
        return r[f"sel_benefit_{SGDF}"] if v_s <= v_a else r["sel_benefit_adam"]

    extras = sorted({r["dataset"] for r in lrf_all} - core)
    if extras:
        section("M5 BDG2 extension subsets (lr_fairness.jsonl extras); SelBest = per-seed "
                "REHEARSAL-selected optimizer at its rehearsed rate (optimizer chosen by "
                "validation online MSE; fully deployable, no test data in any selection)")
        ref = [r for r in lrf if r["dataset"] == "bdg2" and r["H"] == 24 and r["L"] == 96
               and r["seed"] < 3]
        pools = [("bdg2", ref)] + [(ds, [r for r in lrf_all if r["dataset"] == ds])
                                   for ds in extras]
        for ds, sub in pools:
            for bb in sorted({r["backbone"] for r in sub}):
                cells = [_val_chosen(r) for r in sub if r["backbone"] == bb]
                b = texname("MFive", ds, bb)
                emit(b + "SelBestMean", s1(sum(cells) / len(cells)))
                emit(b + "SelBestMin", s1(min(cells)))
                emit(b + "SelBestMax", s1(max(cells)))
else:
    warnings.append("missing lr_fairness.jsonl (run lr_fairness.py to include these macros)")

# ---------- C3 frontier (5-seed, R2) ----------
fs_path = os.path.join(RES, "frontier_seeds.jsonl")
if os.path.exists(fs_path):
    fs = [json.loads(l) for l in open(fs_path)]
    section("C3 frontier (frontier_seeds.jsonl, 5 seeds; supersedes the seed-0 "
            "frontier_data.json); Benefit/BenefitFixed = seed means (rehearsed / fixed-1e-3), "
            "Std = population std over seeds; energy from P_EDGE_W=" + f1(P_EDGE_W) + "W")
    import statistics
    groups = {}
    for r in fs:
        groups.setdefault((r["dataset"], r["label"]), []).append(r)
    emit("FroSeeds", len({r["seed"] for r in fs}))
    energies, mss, params_by = [], [], {}
    for (ds, lab), rows_ in sorted(groups.items()):
        b = texname("Fro", ds, lab)
        bens = [r["benefit"] for r in rows_ if abs(r["benefit"]) <= 100]
        fixd = [r["benefit_fixed"] for r in rows_ if abs(r["benefit_fixed"]) <= 100]
        # per-update wall-clock comes from the CONTROLLED measurement (frontier_timing.py);
        # the in-stream mean carried in frontier_seeds.jsonl is warm-up- and contention-biased
        # to the point of inverting the SGD+m-vs-Adam ordering, so it is only the fallback
        ms = TIMING.get((ds, lab), sum(r["ms"] for r in rows_) / len(rows_))
        emit(b + "Benefit", s1(sum(bens) / len(bens)))
        emit(b + "BenefitStd", f1(statistics.pstdev(bens)))
        emit(b + "BenefitFixed", s1(sum(fixd) / len(fixd)))
        emit(b + "Params", f"{rows_[0]['params']:,}")
        emit(b + "Ms", f2(ms))
        emit(b + "EnergyMilliJoule", f1(ms * P_EDGE_W))
        # the frontier's x axis in kB, so the text can cite it instead of eyeballing Fig. 4;
        # adapt_mem_bytes prefers the MEASURED optimizer state when the run recorded it
        from frontier import adapt_mem_bytes as _amb
        emit(b + "MemKb", f"{_amb(rows_[0]) / 1000:,.0f}")
        # Pooled ms/energy ranges cover only the points the paper REPORTS, i.e. those with a
        # controlled timing. frontier_seeds.jsonl still carries the retired momentum-free SGD
        # points, whose timings come from the old in-stream estimator; letting them set the
        # range would quote a number no figure shows and no strategy in the paper uses.
        if (ds, lab) in TIMING:
            energies.append(ms * P_EDGE_W)
            mss.append(ms)
        params_by.setdefault(ds, {})[lab] = rows_[0]["params"]
    for ds, p in params_by.items():
        full, calib = p.get("PatchTST full·SGD"), p.get("PatchTST calib·SGD")
        if full and calib:
            emit(texname("Fro", ds) + "FullOverCalibParams", f1(full / calib))
    emit("FroEnergyMinMj", f1(min(energies)))
    emit("FroEnergyMaxMj", f1(max(energies)))
    # Duty cycle, not energy, is the honest way to say "compute is not binding": it needs no
    # power assumption (the 5 W proxy cancels), so the claim survives the caveat in the
    # Discussion. One update per revealed horizon at H=24 on an hourly meter = 365 updates/yr.
    emit("FroMsMin", f2(min(mss)))
    emit("FroMsMax", f2(max(mss)))
    emit("FroSecPerYearMin", f2(min(mss) * 365 / 1000))
    emit("FroSecPerYearMax", f2(max(mss) * 365 / 1000))
else:
    warnings.append("missing frontier_seeds.jsonl (run frontier_seeds.py)")

# ---------- staleness ----------
stal = load_optional("staleness_patchtst_full_sgdm.json")
if stal:
    section("staleness (staleness_patchtst_full_sgdm.json); win% >0 = drift-trigger beats periodic "
            "@budget (mean +/- std over seeds where present)")
    for ds, r in stal.items():
        b = texname("Stal", ds)
        if r["win_pct"] is not None:
            emit(b + "WinPct", s1(r["win_pct"]))
            if r.get("win_pct_std") is not None:            # multi-seed schema (referee W2)
                emit(b + "WinPctStd", f1(r["win_pct_std"]))
        if "seeds" in r:
            emit(b + "Seeds", r["seeds"])
        if not isinstance(r["warm"], list):                 # single-seed legacy schema
            emit(b + "Warm", r["warm"])
        emit(b + "StaticMse", f4(r["static"]))
        emit(b + "BestMse", f4(r["best"]))

stal_a = load_optional("staleness_patchtst_full_adam.json")
if stal_a:
    section("staleness, full-Adam variant (staleness_patchtst_full_adam.json)")
    for ds, r in stal_a.items():
        b = texname("Stal", "Adam", ds)
        if r["win_pct"] is not None:
            emit(b + "WinPct", s1(r["win_pct"]))
            if r.get("win_pct_std") is not None:
                emit(b + "WinPctStd", f1(r["win_pct_std"]))

# ---------- C1a warmup confound ----------
wc = load_optional("warmup_confound_sgdm.json")
if wc:
    section("C1a warmup confound (warmup_confound_sgdm.json); values NEGATED to the paper-wide "
            "positive-good convention (improvement% >0 = adaptation better; minor 1). "
            "InflPt = improvement minus sweet-spot improvement (>0 = benefit INFLATED)")
    n_under_infl = n_over_infl = 0
    u_infls, o_infls = [], []
    for key, r in wc.items():
        b = texname("Wc", *key.split("|"))
        emit(b + "Under", s1(-r["under"]))
        emit(b + "Sweet", s1(-r["sweet"]))
        emit(b + "Over", s1(-r["over"]))
        emit(b + "SweetStep", ki(r["sweet_step"]))
        j = r["sweet_idx"]
        emit(b + "UnderStd", f1(r["benefit_std"][0]))
        emit(b + "SweetStd", f1(r["benefit_std"][j]))
        emit(b + "OverStd", f1(r["benefit_std"][-1]))
        u_infl = -(r["under"] - r["sweet"])              # >0 = under-warming inflates
        o_infl = -(r["over"] - r["sweet"])               # >0 = over-warming inflates
        emit(b + "UnderInflPt", s1(u_infl))
        emit(b + "OverInflPt", s1(o_infl))
        n_under_infl += u_infl > 0
        n_over_infl += o_infl > 0
        u_infls.append(u_infl)
        o_infls.append(o_infl)
    emit("WcSettings", len(wc))
    emit("WcUnderInflatedCount", n_under_infl)           # 6/6; the smallest is a statistical tie
    emit("WcOverInflatedCount", n_over_infl)             # 6/6 on the seed-mean
    # The prose quotes the SPREAD of the inflation across settings. Emit its endpoints instead of
    # naming two settings by hand: which setting is extremal moves when the arms are re-run (the
    # SGD -> SGD+momentum migration moved BOTH ends), and a hand-picked pair then silently
    # misstates the range while every individual number in the table stays correct.
    emit("WcUnderInflMinPt", s1(min(u_infls)))
    emit("WcUnderInflMaxPt", s1(max(u_infls)))
    emit("WcOverInflMinPt", s1(min(o_infls)))
    emit("WcOverInflMaxPt", s1(max(o_infls)))
    # (R3) pre-empt the straw-man reading of the endpoints. 50 steps and 50000 steps are easy to
    # dismiss as budgets nobody would pick; the confound survives without them, so quantify the
    # spread of the reported benefit over the range a practitioner WOULD plausibly consider.
    LO, HI = 1000, 20000
    spreads = {}
    for key, r in wc.items():
        idx = [i for i, m in enumerate(r["milestones"]) if LO <= m <= HI]
        vals = [-r["benefit_mean"][i] for i in idx]
        spreads[key] = max(vals) - min(vals)
        emit(texname("Wc", *key.split("|")) + "PracticalSpreadPt", f1(spreads[key]))
    emit("WcPracticalLo", ki(LO))
    emit("WcPracticalHi", ki(HI))
    emit("WcPracticalSpreadMinPt", f1(min(spreads.values())))
    emit("WcPracticalSpreadMaxPt", f1(max(spreads.values())))

# ---------- W1 scalability timing ----------
sc = load_optional("scale_timing_sgdm.json")
if sc:
    section("W1 scalability (scale_timing_sgdm.json); per-update adaptation wall-clock, PatchTST, "
            "SGD@1e-3 / Adam@1e-4")
    for ds, r in sc.items():
        b = texname("Sc", ds)
        emit(b + "Channels", r["channels"])
        emit(b + SGDN + "Ms", f1(TIMING.get((ds, "scale full_sgdm"), r["sgd_ms"])))
        emit(b + "AdamMs", f1(TIMING.get((ds, "scale full_adam"), r["adam_ms"])))

# ---------- LR-transient guard (Fig 5A collapse = steady-state, not startup transient) ----------
lt = load_optional("lr_transient.json")
if lt:
    section("LR-transient guard (lr_transient.json); per-window online MSE by stream quarter")
    for key, e in lt.items():
        ds = key.split("|")[0]
        emit(texname("Lt", ds) + "NWindows", f"{e['arms']['adam_hi']['n_windows']:,}")
        for tag, a in e["arms"].items():
            b = texname("Lt", ds, tag)
            emit(b + "QOne", f2(a["quarters"][0]))
            emit(b + "QFour", f2(a["quarters"][3]))
            emit(b + "LastHundred", f2(a["last_hundred"]))

# ---------- M6: warmup confound across strategies ----------
m6 = load_optional("m6_strategies_sgdm.json")
if m6:
    section("M6 strategy-generality of the warmup confound (m6_strategies.json); "
            "improvement% >0 = adaptation better; InflPt >0 = benefit inflated vs sweet spot")
    for key, e in m6.items():
        ds = key.split("|")[0]
        emit(texname("MSix", ds) + "SweetStep", ki(e["sweet_step"]))
        for strat, s in e["strategies"].items():
            b = texname("MSix", ds, strat)
            emit(b + "Under", s1(s["under"]))
            emit(b + "Sweet", s1(s["sweet"]))
            emit(b + "Over", s1(s["over"]))
            emit(b + "UnderInflPt", s1(s["under_infl"]))
            emit(b + "OverInflPt", s1(s["over_infl"]))

# ---------- C1b leakage check ----------
lk = load_optional("leakage_check_sgdm.json")
if lk:
    section("C1b leakage check (leakage_check.json); benefit% >0 = adaptation better; "
            "leak pt = leaky - delayed (the leak proper, M3); evalset pt = delayed - clean; "
            "inflation pt = leaky - clean (their sum)")
    infl, leak, evs = [], [], []
    for key, r in lk.items():
        b = texname("Lk", *key.split("|"))
        emit(b + "Leaky", s1(r["leaky_benefit"]))
        emit(b + "Clean", s1(r["clean_benefit"]))
        emit(b + "InflationPt", s1(r["inflation_pt"]))
        infl.append(r["inflation_pt"])
        if "delayed_benefit" in r:                       # 3-arm decomposition (referee M3)
            emit(b + "Delayed", s1(r["delayed_benefit"]))
            emit(b + "LeakPt", s1(r["leak_pt"]))
            emit(b + "EvalsetPt", s1(r["evalset_pt"]))
            leak.append(r["leak_pt"]); evs.append(r["evalset_pt"])
    emit("LkInflationMinPt", s1(min(infl)))
    emit("LkInflationMaxPt", s1(max(infl)))
    if leak:
        emit("LkLeakMinPt", s1(min(leak)))
        emit("LkLeakMaxPt", s1(max(leak)))
        emit("LkEvalsetMinPt", s1(min(evs)))
        emit("LkEvalsetMaxPt", s1(max(evs)))

# ---------- C1c validation protocol ----------
vp = load_optional("validation_protocol_sgdm.json")
if vp:
    section("C1c deployable protocol (validation_protocol.json); improvement% >0 = adaptation better")
    for key, r in vp.items():
        b = texname("Vp", *key.split("|"))
        emit(b + "OracleStep", ki(r["oracle_step"]))
        emit(b + "ValStep", ki(r["val_step"]))
        emit(b + "ImpOracle", s1(r["imp_oracle"]))
        emit(b + "ImpVal", s1(r["imp_val"]))
        emit(b + "Delta", s1(r["delta"]))
    emit("VpDeltaMinPt", s1(min(r["delta"] for r in vp.values())))
    emit("VpDeltaMaxPt", s1(max(r["delta"] for r in vp.values())))

# ---------- dataset composition (R3: what is actually being forecast) ----------
# The Appliances stream is multivariate and the two energy channels are a small share of it, so
# the loss the paper reports is dominated by the indoor/outdoor sensor channels. Stated in III
# rather than left for a reader of the public code to discover.
try:
    import numpy as _np
    from online_eval import load_csv as _load_csv
    _d = _load_csv(os.path.join(HERE, "data", "appliances.csv"))
    _T = _d.shape[0]
    _n = int(_T * 0.5)                                    # prep()'s warmup_frac
    _z = (_d - _d[:_n].mean(0)) / (_d[:_n].std(0) + 1e-8)  # prep()'s z-normalisation
    _v = (_z[_n:] ** 2).mean(0)
    section("Dataset composition (appliances.csv through load_csv, i.e. after date/rv1/rv2 are "
            "dropped); share = fraction of the z-normalised test-half variance the paper's MSE "
            "is taken over")
    emit("ApplChannels", _d.shape[1])
    emit("ApplEnergyChannels", 2)                         # Appliances, lights (first two columns)
    emit("ApplSensorChannels", _d.shape[1] - 2)
    emit("ApplEnergySharePct", f1(100 * _v[:2].sum() / _v.sum()))
    emit("ApplSensorSharePct", f1(100 * _v[2:].sum() / _v.sum()))
except Exception as _e:                                   # never block macro generation on this
    warnings.append(f"dataset composition skipped: {type(_e).__name__}: {_e}")

# ---------- R3: two quantities the reviewer asked us to state rather than assert ----------
try:
    import numpy as _np2, csv as _csv
    _vp = load_optional("validation_protocol_sgdm.json")
    if _vp:
        section("C1c: rank correlation between the held-out validation curve and the static "
                "TEST curve (Fig. 2 claims the former 'tracks' the latter; this quantifies it)")
        def _sp(a, b):
            ra, rb = _np2.argsort(_np2.argsort(a)), _np2.argsort(_np2.argsort(b))
            return float(_np2.corrcoef(ra, rb)[0, 1])
        _rs = [_sp(_np2.array(v["val_mean"]), _np2.array(v["static_mean"])) for v in _vp.values()]
        emit("VpSpearmanMin", f2(min(_rs)))
        emit("VpSpearmanMax", f2(max(_rs)))
    section("BDG2 subset regularity: share of consecutive-equal samples, i.e. how much of a "
            "series is gap-fill rather than movement (the anti-selected subset is mostly fill)")
    for _f, _n in (("bdg2", "BdgTwo"), ("bdg2_rat_worst", "BdgTwoRatWorst")):
        _p = os.path.join(HERE, "data", _f + ".csv")
        if not os.path.exists(_p):
            continue
        _rows = list(_csv.reader(open(_p)))
        _d = _np2.array([[float(x) if x else _np2.nan for x in r[1:]] for r in _rows[1:]])
        _rep = [float((_np2.diff(_d[:, j]) == 0).mean()) * 100 for j in range(_d.shape[1])]
        emit("Data" + _n + "RepeatPct", f1(float(_np2.median(_rep))))
except Exception as _e:
    warnings.append(f"R3 extras skipped: {type(_e).__name__}: {_e}")

emit("MacrosDate", datetime.date.today().isoformat())

header = ["% AUTO-GENERATED by experiments/tsf_edge/gen_macros.py -- DO NOT EDIT.",
          "% Regenerate: .venv/bin/python experiments/tsf_edge/gen_macros.py",
          "% Usage: \\input{macros.tex}; values are bare numbers (append \\% in prose),",
          "% percents carry explicit +/- signs; sign conventions per section header below."]
header += [f"% WARNING: {w}" for w in warnings]
with open(OUT, "w") as f:
    f.write("\n".join(header) + "\n" + "\n".join(lines) + "\n")
print(f"wrote {OUT}: {len(seen)} macros")
for w in warnings:
    print(f"WARNING: {w}")
