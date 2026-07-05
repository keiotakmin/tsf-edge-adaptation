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
  staleness_patchtst.json    staleness (optional)— win% sign: >0 = drift-trigger BETTER
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

DIG = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
       "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}


def texname(*parts):
    words = [w for p in parts for w in re.split(r"[^0-9A-Za-z]+", str(p)) if w]
    s = "".join(w[:1].upper() + w[1:].lower() for w in words)
    return "".join(DIG.get(ch, ch) for ch in s)


def s1(x, nd=1):                       # signed, nd decimals; round-then-add-0.0 avoids "-0.0"
    return f"{round(x, nd) + 0.0:+.{nd}f}"
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

def winner(r): return "SGD" if r["benefit_sgd"] >= r["benefit_adam"] else "Adam"

section("C3 grid (grid.jsonl); benefit% >0 = adaptation better than static")
bs = [r["benefit_sgd"] for r in rows]
ba = [r["benefit_adam"] for r in rows]
emit("GridCells", len(rows))
emit("GridSeeds", len({r["seed"] for r in rows}))
emit("GridDatasets", len({r["dataset"] for r in rows}))
emit("GridSgdFloor", s1(min(bs)))
emit("GridSgdFloorExact", s1(min(bs), nd=2))         # +0.03 before the 1-dp rounding (minor 4)
emit("GridSgdNearZeroCells", sum(abs(b) < 0.05 for b in bs))
emit("GridSgdNegCells", sum(b < 0 for b in bs))
emit("GridAdamWorst", s1(min(ba)))
emit("GridAdamNegCells", sum(b < 0 for b in ba))
emit("GridAdamNegPct", round(100 * sum(b < 0 for b in ba) / len(ba)))
emit("GridSgdWins", sum(winner(r) == "SGD" for r in rows))
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
emit("GridWarmCapStep", max(WARM_GRID))
emit("GridWarmCapCells", sum(r["warmup"] == max(WARM_GRID) for r in rows))

for probe, fmt in [("p3_drift", f2), ("p2_gradcos", f2), ("p1_noise", f3)]:
    win_s = [r[probe] for r in rows if winner(r) == "SGD"]
    win_a = [r[probe] for r in rows if winner(r) == "Adam"]
    base = texname("Grid", probe.split("_")[0])
    emit(base + "SgdWinMean", fmt(sum(win_s) / len(win_s)))
    emit(base + "AdamWinMean", fmt(sum(win_a) / len(win_a)))
    emit(base + "Gap", s1(sum(win_a) / len(win_a) - sum(win_s) / len(win_s),
                          nd=1 if probe == "p3_drift" else 2))

for ds in sorted({r["dataset"] for r in rows}):
    sub = [r for r in rows if r["dataset"] == ds]
    b = texname("Grid", ds)
    emit(b + "Cells", len(sub))
    emit(b + "PThree", f2(sum(r["p3_drift"] for r in sub) / len(sub)))
    emit(b + "POne", f3(sum(r["p1_noise"] for r in sub) / len(sub)))
    emit(b + "SgdWins", sum(winner(r) == "SGD" for r in sub))
    emit(b + "AdamWins", sum(winner(r) == "Adam" for r in sub))
    emit(b + "AdamNegCells", sum(r["benefit_adam"] < 0 for r in sub))

for L in sorted({r["L"] for r in rows}):                    # lookback robustness (L=192 shrinks
    sub = [r for r in rows if r["L"] == L]                  # the Adam-favourable corner)
    b = texname("Grid", "L", L)
    emit(b + "Cells", len(sub))
    emit(b + "SgdWins", sum(winner(r) == "SGD" for r in sub))
    emit(b + "AdamWins", sum(winner(r) == "Adam" for r in sub))
    emit(b + "AdamNegCells", sum(r["benefit_adam"] < 0 for r in sub))
    emit(b + "AdamNegPct", round(100 * sum(r["benefit_adam"] < 0 for r in sub) / len(sub)))
    emit(b + "SgdFloor", s1(min(r["benefit_sgd"] for r in sub)))
    emit(b + "AdamWorst", s1(min(r["benefit_adam"] for r in sub)))

for ds in sorted({r["dataset"] for r in rows}):             # dataset x L cells (cited: Appliances)
    for L in sorted({r["L"] for r in rows}):
        sub = [r for r in rows if r["dataset"] == ds and r["L"] == L]
        b = texname("Grid", ds, "L", L)
        emit(b + "Cells", len(sub))
        emit(b + "SgdWins", sum(winner(r) == "SGD" for r in sub))
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
              3e-03: "ThreeEMinusThree", 1e-02: "OneEMinusTwo"}
    emit("LrCells", len(lrf))
    emit("LrGridPoints", len(lrf[0]["lrs"]))
    emit("LrSeeds", len({r["seed"] for r in lrf}))
    readings = {"Fixed": lambda r, o: r[o]["0.001"]["benefit"],
                "Sel":   lambda r, o: r[f"sel_benefit_{o}"],
                "Orc":   lambda r, o: r[f"oracle_benefit_{o}"]}
    def _median(xs):
        xs = sorted(xs)
        n = len(xs)
        return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2

    def _cfg_stats(sub, get):
        """Seed-majority config-level wins (a config = dataset x backbone x L x H), guarding
        against the seeds-as-independent-samples reading of the cell counts (referee minor)."""
        cfgs = {}
        for r in sub:
            cfgs.setdefault((r["dataset"], r["backbone"], r["L"], r["H"]), []).append(r)
        aw = sum(sum(get(r, "adam") > get(r, "sgd") for r in v) > len(v) / 2
                 for v in cfgs.values())
        unan = sum(sum(get(r, "adam") > get(r, "sgd") for r in v) in (0, len(v))
                   for v in cfgs.values())
        return len(cfgs), aw, unan

    for Lv in sorted({r["L"] for r in lrf}):                # per-lookback three-reading stats
        sub = [r for r in lrf if r["L"] == Lv]
        base = texname("Lr", "L", Lv)
        emit(base + "Cells", len(sub))
        for rd, get in readings.items():
            b_s = [get(r, "sgd") for r in sub]
            b_a = [get(r, "adam") for r in sub]
            b = base + rd
            emit(b + "SgdWins", sum(s >= a for s, a in zip(b_s, b_a)))
            emit(b + "AdamWins", sum(a > s for s, a in zip(b_s, b_a)))
            emit(b + "SgdNegCells", sum(x < 0 for x in b_s))
            emit(b + "SgdMin", s1(min(b_s)))
            emit(b + "AdamNegCells", sum(x < 0 for x in b_a))
            emit(b + "AdamNegPct", round(100 * sum(x < 0 for x in b_a) / len(b_a)))
            emit(b + "AdamMin", s1(min(b_a)))
            emit(b + "MeanGapPt", s1(sum(a - s for s, a in zip(b_s, b_a)) / len(sub)))
            emit(b + "MedianGapPt", s1(_median([a - s for s, a in zip(b_s, b_a)])))
            ncfg, aw, unan = _cfg_stats(sub, get)
            emit(b + "CfgAdamWins", aw)
            emit(b + "CfgSgdWins", ncfg - aw)
            emit(b + "CfgUnanimous", unan)
    for lr in sorted(lrf[0]["lrs"]):                        # pooled per-LR plateau statistics
        for o in ("sgd", "adam"):
            vals = [r[o][f"{lr:g}"]["benefit"] for r in lrf]
            b = texname("Lr", o) + "At" + LRNAME[lr]
            emit(b + "NegCells", sum(v < 0 for v in vals))
            emit(b + "Mean", s1(sum(vals) / len(vals)))
            emit(b + "Min", s1(min(vals)))
    # selection behaviour (pooled over all cells)
    emit("LrAdamSelLeqThreeEMinusFourCells", sum(r["sel_lr_adam"] <= 3e-4 for r in lrf))
    emit("LrAdamSelGeqOneEMinusThreeCells", sum(r["sel_lr_adam"] >= 1e-3 for r in lrf))
    emit("LrSgdSelGeqOneEMinusThreeCells", sum(r["sel_lr_sgd"] >= 1e-3 for r in lrf))
    emit("LrSgdSelNegCellsAll", sum(r["sel_benefit_sgd"] < 0 for r in lrf))
    emit("LrSgdSelMinAll", s1(min(r["sel_benefit_sgd"] for r in lrf)))
    emit("LrSelAdamNegCellsAll", sum(r["sel_benefit_adam"] < 0 for r in lrf))
    emit("LrSelAdamMinAll", s1(min(r["sel_benefit_adam"] for r in lrf)))
    # pooled three-reading win counts (abstract/intro cite these)
    for rd, get in readings.items():
        b_s = [get(r, "sgd") for r in lrf]
        b_a = [get(r, "adam") for r in lrf]
        emit("Lr" + rd + "SgdWinsAll", sum(s >= a for s, a in zip(b_s, b_a)))
        emit("Lr" + rd + "AdamWinsAll", sum(a > s for s, a in zip(b_s, b_a)))
        emit("Lr" + rd + "MeanGapPtAll", s1(sum(a - s for s, a in zip(b_s, b_a)) / len(lrf)))
        emit("Lr" + rd + "MedianGapPtAll", s1(_median([a - s for s, a in zip(b_s, b_a)])))
        ncfg, aw, unan = _cfg_stats(lrf, get)
        emit("Lr" + rd + "CfgAdamWinsAll", aw)
        emit("Lr" + rd + "CfgSgdWinsAll", ncfg - aw)
        emit("Lr" + rd + "CfgUnanimousAll", unan)
    emit("LrConfigs", len({(r["dataset"], r["backbone"], r["L"], r["H"]) for r in lrf}))
    for Hv in sorted({r["H"] for r in lrf}):                # per-horizon robustness (compact)
        sub = [r for r in lrf if r["H"] == Hv]
        b = texname("Lr", "H", Hv)
        emit(b + "Cells", len(sub))
        emit(b + "SelAdamWins", sum(r["sel_benefit_adam"] > r["sel_benefit_sgd"] for r in sub))
        emit(b + "SelAdamNegCells", sum(r["sel_benefit_adam"] < 0 for r in sub))
    # M5: BDG2 extension subsets (fair-LR H24/L96/3-seed cells; NOT part of the C3 stats)
    extras = sorted({r["dataset"] for r in lrf_all} - core)
    if extras:
        section("M5 BDG2 extension subsets (lr_fairness.jsonl extras); SelBest = per-seed "
                "max(sel_sgd, sel_adam), i.e. the fair benefit of the better online optimizer")
        ref = [r for r in lrf if r["dataset"] == "bdg2" and r["H"] == 24 and r["L"] == 96
               and r["seed"] < 3]
        pools = [("bdg2", ref)] + [(ds, [r for r in lrf_all if r["dataset"] == ds])
                                   for ds in extras]
        for ds, sub in pools:
            for bb in sorted({r["backbone"] for r in sub}):
                cells = [max(r["sel_benefit_sgd"], r["sel_benefit_adam"])
                         for r in sub if r["backbone"] == bb]
                b = texname("MFive", ds, bb)
                emit(b + "SelBestMean", s1(sum(cells) / len(cells)))
                emit(b + "SelBestMin", s1(min(cells)))
                emit(b + "SelBestMax", s1(max(cells)))
else:
    warnings.append("missing lr_fairness.jsonl (run lr_fairness.py to include these macros)")

# ---------- C2 frontier ----------
fro = load_optional("frontier_data.json")
if fro:
    section("C2 frontier (frontier_data.json); benefit% >0 = better; energy from P_EDGE_W="
            + f1(P_EDGE_W) + "W")
    energies = []
    for ds, frows in fro.items():
        params = {r["label"]: r["params"] for r in frows}
        for r in frows:
            b = texname("Fro", ds, r["label"])
            emit(b + "Benefit", s1(r["benefit"]))
            emit(b + "Params", f"{r['params']:,}")
            emit(b + "Ms", f2(r["ms"]))
            emit(b + "EnergyMilliJoule", f1(r["ms"] * P_EDGE_W))
            energies.append(r["ms"] * P_EDGE_W)
            if "warmup" in r:
                emit(b + "Warmup", r["warmup"])
            if "benefit_fixed" in r:                        # the old fixed-1e-3 reading, kept
                emit(b + "BenefitFixed", s1(r["benefit_fixed"]))   # for the confound narrative
        full, calib = params.get("PatchTST full·SGD"), params.get("PatchTST calib·SGD")
        if full and calib:
            emit(texname("Fro", ds) + "FullOverCalibParams", f1(full / calib))
    emit("FroEnergyMinMj", f1(min(energies)))
    emit("FroEnergyMaxMj", f1(max(energies)))

# ---------- staleness ----------
stal = load_optional("staleness_patchtst.json")
if stal:
    section("staleness (staleness_patchtst.json); win% >0 = drift-trigger beats periodic @budget")
    for ds, r in stal.items():
        b = texname("Stal", ds)
        if r["win_pct"] is not None:
            emit(b + "WinPct", s1(r["win_pct"]))
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

# ---------- C1a warmup confound ----------
wc = load_optional("warmup_confound.json")
if wc:
    section("C1a warmup confound (warmup_confound.json); values NEGATED to the paper-wide "
            "positive-good convention (improvement% >0 = adaptation better; minor 1). "
            "InflPt = improvement minus sweet-spot improvement (>0 = benefit INFLATED)")
    n_under_infl = n_over_infl = 0
    for key, r in wc.items():
        b = texname("Wc", *key.split("|"))
        emit(b + "Under", s1(-r["under"]))
        emit(b + "Sweet", s1(-r["sweet"]))
        emit(b + "Over", s1(-r["over"]))
        emit(b + "SweetStep", r["sweet_step"])
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
    emit("WcSettings", len(wc))
    emit("WcUnderInflatedCount", n_under_infl)           # 5/6: Appliances/PatchTST is a tie
    emit("WcOverInflatedCount", n_over_infl)             # 6/6 on the seed-mean

# ---------- M6: warmup confound across strategies ----------
m6 = load_optional("m6_strategies.json")
if m6:
    section("M6 strategy-generality of the warmup confound (m6_strategies.json); "
            "improvement% >0 = adaptation better; InflPt >0 = benefit inflated vs sweet spot")
    for key, e in m6.items():
        ds = key.split("|")[0]
        emit(texname("MSix", ds) + "SweetStep", e["sweet_step"])
        for strat, s in e["strategies"].items():
            b = texname("MSix", ds, strat)
            emit(b + "Under", s1(s["under"]))
            emit(b + "Sweet", s1(s["sweet"]))
            emit(b + "Over", s1(s["over"]))
            emit(b + "UnderInflPt", s1(s["under_infl"]))
            emit(b + "OverInflPt", s1(s["over_infl"]))

# ---------- C1b leakage check ----------
lk = load_optional("leakage_check.json")
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
vp = load_optional("validation_protocol.json")
if vp:
    section("C1c deployable protocol (validation_protocol.json); improvement% >0 = adaptation better")
    for key, r in vp.items():
        b = texname("Vp", *key.split("|"))
        emit(b + "OracleStep", r["oracle_step"])
        emit(b + "ValStep", r["val_step"])
        emit(b + "ImpOracle", s1(r["imp_oracle"]))
        emit(b + "ImpVal", s1(r["imp_val"]))
        emit(b + "Delta", s1(r["delta"]))
    emit("VpDeltaMinPt", s1(min(r["delta"] for r in vp.values())))
    emit("VpDeltaMaxPt", s1(max(r["delta"] for r in vp.values())))

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
