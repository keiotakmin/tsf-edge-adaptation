"""Single-source-of-truth numbers: generate results/tsf_edge/macros.tex from the RESULT DATA
FILES so no number in the paper is ever hand-copied. Rerun after any experiment rerun; the
paper does `\\input{macros.tex}` and cites only macros.

Sources (missing optional files are skipped with a warning):
  grid.jsonl                 C3 grid (required)  — benefit% sign: >0 = adaptation BETTER
  frontier_data.json         C2 frontier         — benefit% sign: >0 = adaptation BETTER
  staleness_patchtst.json    staleness (optional)— win% sign: >0 = drift-trigger BETTER
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

# ---------- C1a warmup confound ----------
wc = load_optional("warmup_confound.json")
if wc:
    section("C1a warmup confound (warmup_confound.json); benefit% <0 = adaptation better (Table 1 sign)")
    for key, r in wc.items():
        b = texname("Wc", *key.split("|"))
        emit(b + "Under", s1(r["under"]))
        emit(b + "Sweet", s1(r["sweet"]))
        emit(b + "Over", s1(r["over"]))
        emit(b + "SweetStep", r["sweet_step"])

# ---------- C1b leakage check ----------
lk = load_optional("leakage_check.json")
if lk:
    section("C1b leakage check (leakage_check.json); benefit% >0 = adaptation better; "
            "inflation pt = leaky benefit - clean benefit")
    infl = []
    for key, r in lk.items():
        b = texname("Lk", *key.split("|"))
        emit(b + "Leaky", s1(r["leaky_benefit"]))
        emit(b + "Clean", s1(r["clean_benefit"]))
        emit(b + "InflationPt", s1(r["inflation_pt"]))
        infl.append(r["inflation_pt"])
    emit("LkInflationMinPt", s1(min(infl)))
    emit("LkInflationMaxPt", s1(max(infl)))

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
