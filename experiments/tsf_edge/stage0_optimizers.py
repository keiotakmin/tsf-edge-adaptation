"""Stage 0 of the specialized-optimizer extension: do EXISTING memory-light and
learning-rate-free optimizers already occupy the niche the paper's recipe leaves open
("Adam-quality at SGD-memory with no rehearsal")?

Protocol = lr_fairness.py exactly (fair warmup via warm_and_select; per-cell, per-optimizer
online-LR REHEARSAL on the held-out pre-drift validation slice; full LR x {val,test} surface
logged; full-model adaptation, stride=H, leakage-free). Two optimizer families:

  LR-ful  (8-LR rehearsal grid, like sgd/adam in lr_fairness.py):
    lion        1x state (sign-momentum; lion-pytorch)
    adafactor   sublinear state (factored 2nd moment; relative_step=False so the grid applies)
    signsgd     0 state (sign of grad, momentum=0) -- zero-state like SGD but scale-free
  LR-free (ONE run at their lr=1.0 default, NO rehearsal -- that is their pitch):
    prodigy     Adam-flavored (~4x state: tests the tuning axis, not memory)
    dadapt_sgd  SGD-flavored D-Adaptation
    dadapt_adam Adam-flavored D-Adaptation
    dog         distance-over-gradients (official dog-optimizer package)

ScheduleFree / SM3 deliberately excluded: online rates are constant (no schedules) and
AdaFactor already covers sublinear state. sgd/adam themselves are NOT rerun -- their rehearsed
readings come from lr_fairness.jsonl (same cells, deterministic warmup; the static baseline is
cross-checked against the stored one and a loud warning fires on drift).

STAGE 0b (--stage 0b, writes stage0b_optimizers.jsonl) asks the follow-up question: Stage 0
showed no GENERAL-PURPOSE optimizer fills the gap, so does one of the optimizers actually
DESIGNED for non-stationary online prediction? None of them has ever been run on a deep TSF
backbone. All are LR-ful (each has a base step, so it gets the same rehearsal grid -- treating
them as "LR-free" would re-introduce the C3 unfairness in reverse):

    sgdm            1x state  the missing SGD+momentum rung of the frontier (0x/1x/2x)
    obgd            0x state  overshooting-bounded GD, streaming RL (Elsayed+ 2024)
    adaptive_obgd   1x state  its preconditioned variant
    dons            1x state  discounted diagonal Online Newton Step / RLS (Hazan+ 07, Anava+ 13)
    upgd            1x state  utility-based perturbed GD, continual learning (ICLR 2024)
    idbd            2x state  Sutton's (1992) step-size meta-learning -- Adam's footprint
    autostep        3x state  its scale-robust successor (Mahmood+ 2012); the variant that
                              actually answers the step-size-meta-learning question, since
                              idbd degenerates to plain SGD at this gradient scale

Implementations live in online_optimizers.py and are gated by test_online_optimizers.py
(IDBD reproduces Sutton's tracking testbed; UPGD is bit-compared with the authors' code;
ObGD's overshoot cap is verified to bind at 1/kappa) -- run that BEFORE spending grid time.

One JSONL row per cell -> results/tsf_edge/stage0[b]_optimizers.jsonl. Resumable and mergeable
like lr_fairness.py: only missing (optimizer, lr) points are computed on rerun. `--summarize`
merges both stages and prints the verdict table (contenders vs rehearsed-Adam / default-SGD),
including the two deployment metrics an edge operator actually needs: how many DECADES of LR
stay within 1pt of the optimizer's own best (`band`), and what a 10x LR mis-set costs (`mis1x`).
"""
from __future__ import annotations
import argparse, json, os, time

import numpy as np

from online_eval import LR_GRID, VAL_FRAC, load_csv, prep, stream_eval, warm_and_select

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "tsf_edge", "stage0_optimizers.jsonl")
OUT_B = os.path.join(ROOT, "results", "tsf_edge", "stage0b_optimizers.jsonl")
OUT_C = os.path.join(ROOT, "results", "tsf_edge", "stage0c_optimizers.jsonl")
OUT_D = os.path.join(ROOT, "results", "tsf_edge", "stage0d_optimizers.jsonl")
LRFAIR = os.path.join(ROOT, "results", "tsf_edge", "lr_fairness.jsonl")
DEFAULT_LRS = ",".join(f"{x:g}" for x in LR_GRID)
OPTS_LRFUL = ["lion", "adafactor", "signsgd"]
OPTS_LRFREE = ["prodigy", "dadapt_sgd", "dadapt_adam", "dog"]
OPTS_STAGE0B = ["sgdm", "obgd", "adaptive_obgd", "dons", "upgd", "idbd", "autostep"]
OPTS_STAGE0C = ["obsign", "obsign_t3e3", "obsign_t1e3", "relsign"]   # designed FROM the Stage-0b frontier (see
                                       # online_optimizers.ObSign) -- run with --stage 0c
# Stage 0d (2026-09-04, --stage 0d, writes stage0d_optimizers.jsonl): three more values of tau,
# for the ONE question Stage 0c could not answer. The paper's design rule for tau is a margin
# rule -- put the knee tau*RMS(p) at least a decade below the rate you ship -- and Stage 0c
# sampled that margin at 1.28 (tau=1e-3, passes), 0.80 (3e-3, fails) and 0.28 decades (1e-2,
# fails). Three points can say the boundary lies between 0.80 and 1.28 and nothing more, which
# is what the limitations section had to admit. These three put it at 1.10, 0.98 and 0.58
# decades, so the pass/fail crossing is bracketed inside 0.1-0.2 decades and the rule is a
# measurement rather than a round number. NEW FILE, per CLAUDE.md: the Stage-0c artifact is
# never rewritten.
OPTS_STAGE0D = ["obsign_t1p5e3", "obsign_t2e3", "obsign_t5e3"]
# Per-optimizer LR grids. A shared grid is only fair if it BRACKETS every method's optimum
# (CLAUDE.md bracketing rule). The ObGD family needs the shared grid PLUS extra reach upward:
# above the lr where its overshoot cap engages the update no longer depends on lr at all (a
# plateau, by design), while below it the rule reduces exactly to SGD at that lr -- so the
# whole curve, and the transition between the two regimes, has to be inside the grid.
# Round 1 (2026-08-06) started at 1e-3 and the live bracketing check caught the bottom edge
# winning by 0.6 pt within 23 cells: the capped plateau can be WORSE than small-lr SGD, so the
# grid was extended down to the shared bottom. Keeping the shared 10 points also makes the
# `band` metric comparable across optimizers (same range, same density).
# UPGD gates the update by (1 - sigmoid(scaled utility)), a factor in roughly [0.27, 0.73], so
# its effective step is about half the nominal one and its optimum sits ABOVE SGD's -- whose own
# grid top was already pushed to 1e-1 for the same reason. The live check caught it on that edge
# at L=192 (round 2, 2026-08-07); two more rates follow the same bracketing rule.
#
# OPEN, DELIBERATELY NOT PATCHED MID-RUN (2026-08-07): at H=96 the SGD-family rules (dons,
# idbd, autostep, and upgd before its extension) all select the shared grid's TOP rate. The
# mechanism is the paper's own: at stride=H a long horizon leaves 2-4x fewer online updates,
# so each one has to be larger -- \S IV-B already reports SGD picking the added top rates only
# at H in {48,96}. Extending now would (a) make this a third grid change inside one run and
# (b) give the contenders a WIDER search than the sgd/adam references, whose readings come from
# lr_fairness.jsonl on 3e-6..1e-1 -- exactly the C3 unfairness this paper is about, in reverse.
# Plan instead: finish the run, then ONE fill-in pass adding 3e-1 and 1.0 for sgdm/dons/idbd/
# autostep AND matched sgd/adam readings at those two rates. The headline table stays on the
# shared 3e-6..1e-1 grid (fair by construction, and what `band`/`mis1x` are computed on); the
# top extension is reported separately as the bracketing check it is.
# RESOLVED by the summarize side instead (2026-08-07): every headline column is now computed on
# SHARED_LR, so giving a contender extra top rates can no longer buy it an unfair search. The
# top rates are therefore added here for the SGD-family rules that hit the edge at H=96 -- plus
# MATCHED sgd/adam readings at exactly those two rates, so the bracketing question is answered
# for the references too. sgd/adam are not in OPTS_STAGE0B; pass them via --opts-lrful on the
# fill-in pass (run_stage0b_fillin.sh) and this table supplies their two-point grid.
LR_GRID_BY_OPT = {
    "obgd":          list(LR_GRID) + [1.0, 10.0, 100.0],
    "adaptive_obgd": list(LR_GRID) + [1.0, 10.0, 100.0],
    "upgd":          list(LR_GRID) + [3e-1, 1.0],
    "sgdm":          list(LR_GRID) + [3e-1, 1.0],
    "dons":          list(LR_GRID) + [3e-1, 1.0],
    "idbd":          list(LR_GRID) + [3e-1, 1.0],
    "autostep":      list(LR_GRID) + [3e-1, 1.0],
    "obsign":        list(LR_GRID) + [3e-1, 1.0],   # needs reach: above the knee the
                                                    # rate cancels and the curve plateaus
    # the smaller-tau arms put the knee at tau*RMS(p) ~ 5e-5 and 1.5e-4, so their whole plateau
    # already lies inside the shared grid -- no extension needed, and `band` stays comparable
    "sgd":           [3e-1, 1.0],          # matched top-extension for the references only
    "adam":          [3e-1, 1.0],          # (their 3e-6..1e-1 surface lives in lr_fairness.jsonl)
}
RES_KEYS = ("opt_state_bytes", "adapt_ms", "n_adapt_params", "peak_adapt_mem_kb")
STATE_MULT = {"sgd": 0, "obgd": 0, "signsgd": 0, "sgdm": 1, "lion": 1, "dons": 1, "upgd": 1,
              "adaptive_obgd": 1, "adam": 2, "idbd": 2, "autostep": 3,
              "obsign": 0, "obsign_t5e3": 0, "obsign_t3e3": 0, "obsign_t2e3": 0,
              "obsign_t1p5e3": 0, "obsign_t1e3": 0,
              "relsign": 0}   # documented; measured value wins
dev = "cuda"


def load_jsonl(path):
    rows = {}
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
                rows[(r["dataset"], r["backbone"], r["L"], r["H"], r["seed"])] = r
            except json.JSONDecodeError:
                pass                  # drop a partial last line from an interrupted run
    return rows


def run_cells(args):
    L, H = args.L, args.H
    datasets = args.datasets.split(",")
    backbones = args.backbones.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    lrs = [float(x) for x in args.lrs.split(",")]
    opts_lrful = [o for o in args.opts_lrful.split(",") if o]
    opts_lrfree = [o for o in args.opts_lrfree.split(",") if o]
    out_path = args.out
    # LR_GRID_BY_OPT overrides the shared grid, and for sgd/adam it overrides it with TWO
    # rates -- correct for the full-model screen, where their 3e-6..1e-1 surface is read from
    # lr_fairness.jsonl, and WRONG for any other strategy, where no such file exists and the
    # reference arms would be swept on {3e-1, 1.0} alone. Outside "full" the override is
    # therefore additive: the shared grid plus whatever top extension the arm declares.
    grid_for = (lambda o: LR_GRID_BY_OPT.get(o, lrs)) if args.which == "full" else \
        (lambda o: sorted(set(lrs) | set(LR_GRID_BY_OPT.get(o, []))))

    all_rows = load_jsonl(out_path)
    ref_rows = load_jsonl(LRFAIR)     # sgd/adam readings + static cross-check

    def flush():
        tmp = out_path + ".tmp"
        with open(tmp, "w") as f:
            for r in all_rows.values():
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, out_path)

    def need_of(old):
        need = {}
        for o in opts_lrful:
            miss = [lr for lr in grid_for(o) if old is None or f"{lr:g}" not in old.get(o, {})]
            if miss:
                need[o] = miss
        for o in opts_lrfree:
            if old is None or o not in old:
                need[o] = [1.0]
        return need

    todo = []
    for name in datasets:
        for bb in backbones:
            for seed in seeds:
                need = need_of(all_rows.get((name, bb, L, H, seed)))
                if need:
                    todo.append((name, bb, seed, need))
    npts = sum(len(v) for *_, n in todo for v in n.values())
    print(f"L={L} H={H}: {len(todo)} cells need work ({npts} missing (opt,lr) points)", flush=True)

    data_cache = {}
    for i, (name, bb, seed, need) in enumerate(todo, 1):
        if name not in data_cache:
            data_cache[name] = load_csv(os.path.join(ROOT, "experiments/tsf_edge/data", f"{name}.csv"))
        t0 = time.perf_counter()
        d, n_warm, C = prep(data_cache[name], device=dev)
        n_train = int(n_warm * (1 - VAL_FRAC))
        model, wstep, wval = warm_and_select(bb, L, H, C, d, n_train, n_warm, seed)
        st = stream_eval(model, d, bb, n_warm, L, H, "static", device=dev)["mse"]
        ref = ref_rows.get((name, bb, L, H, seed))
        if ref is not None and abs(st - ref["static"]) > 1e-3 * max(ref["static"], 1e-8):
            print(f"  WARNING: re-warmed static {st:.5f} != lr_fairness {ref['static']:.5f} "
                  f"({name}/{bb}/s{seed}) -- cross-file comparison may be inconsistent", flush=True)
        old = all_rows.get((name, bb, L, H, seed))
        st_use = (old or ref or {}).get("static", st)
        d_val = d[:n_warm]           # validation stream = the held-out pre-drift slice
        row = old if old is not None else dict(dataset=name, backbone=bb, L=L, H=H, seed=seed,
                                               warmup=wstep, static=st, val_static=wval)
        parts = []
        for okind, lrs_need in need.items():
            strat = f"{args.which}_{okind}"
            lrful = okind in opts_lrful
            sweep = dict(row.get(okind, {})) if lrful else {}
            try:
                tests = {}
                for lr in lrs_need:
                    v = stream_eval(model, d_val, bb, n_train, L, H, strat, lr=lr,
                                    device=dev)["mse"]
                    te = stream_eval(model, d, bb, n_warm, L, H, strat, lr=lr, device=dev)
                    tests[lr] = te
                    sweep[f"{lr:g}"] = dict(val=v, test=te["mse"],
                                            benefit=100 * (st_use - te["mse"]) / st_use)
                if lrful:
                    row[okind] = sweep
                    grid = sorted(float(k) for k in sweep)
                    key = lambda x, f: sweep[f"{x:g}"][f] if sweep[f"{x:g}"][f] == sweep[f"{x:g}"][f] else float("inf")
                    sel = min(grid, key=lambda x: key(x, "val"))
                    orc = min(grid, key=lambda x: key(x, "test"))
                    row[f"sel_lr_{okind}"] = sel
                    row[f"sel_benefit_{okind}"] = sweep[f"{sel:g}"]["benefit"]
                    row[f"oracle_lr_{okind}"] = orc
                    row[f"oracle_benefit_{okind}"] = sweep[f"{orc:g}"]["benefit"]
                    res_from = tests.get(sel) or next(iter(tests.values()))
                    parts.append(f"{okind} sel={sel:g} {row[f'sel_benefit_{okind}']:+6.1f}%")
                else:
                    row[okind] = sweep["1"]      # {val,test,benefit} at the lr=1.0 default
                    res_from = tests[1.0]
                    parts.append(f"{okind} {row[okind]['benefit']:+6.1f}%")
                row[f"res_{okind}"] = {k: res_from[k] for k in RES_KEYS}
            except Exception as e:                # a broken contender must not kill the run
                row[f"error_{okind}"] = f"{type(e).__name__}: {e}"
                parts.append(f"{okind} ERROR")
        all_rows[(name, bb, L, H, seed)] = row
        flush()
        for o in need:                            # CLAUDE.md bracketing rule, checked live
            orc, sw = row.get(f"oracle_lr_{o}"), row.get(o)
            g = grid_for(o)
            if orc is None or not isinstance(sw, dict) or orc not in (min(g), max(g)):
                continue
            interior = [sw[f"{x:g}"]["benefit"] for x in g if x not in (min(g), max(g))
                        and sw.get(f"{x:g}", {}).get("benefit") is not None]
            interior = [b for b in interior if b == b]
            # a plateau that merely REACHES the edge is fine (ObGD does this by design);
            # only a peak that is still RISING at the edge means the grid is too narrow.
            if interior and row[f"oracle_benefit_{o}"] - max(interior) > 0.5:
                print(f"  BRACKETING: {o} oracle lr={orc:g} is on the grid EDGE "
                      f"[{min(g):g},{max(g):g}] and beats every interior LR by "
                      f"{row[f'oracle_benefit_{o}'] - max(interior):.1f}pt -- extend the grid",
                      flush=True)
        reftxt = (f" | ref SGDr {ref['sel_benefit_sgd']:+.1f}% Adamr {ref['sel_benefit_adam']:+.1f}%"
                  if ref is not None else "")
        print(f"[{i:2d}/{len(todo)}] L{L} H{H} {name:11s} {bb:9s} s{seed} warm={wstep:5d} | "
              + " | ".join(parts) + reftxt + f" | {time.perf_counter() - t0:5.0f}s", flush=True)
    print(f"\nwrote {out_path} ({len(all_rows)} rows total)")   # NOT the module-level OUT:
                                    # with --out this line used to name the canonical
                                    # stage0_optimizers.jsonl while writing elsewhere



# The range every Stage-0/0b optimizer AND the sgd/adam references share. Rates outside it
# exist for some contenders (the ObGD plateau reaches 100), but every headline column -- sel,
# oracle, band, mis1x -- is computed ON THIS RANGE ONLY, or a contender with a wider grid would
# be getting a wider search than the baselines: the C3 unfairness in reverse.
SHARED_LR = (min(LR_GRID), max(LR_GRID))


def _restrict(sweep, rng=SHARED_LR):
    lo, hi = rng
    return {k: v for k, v in sweep.items() if lo * 0.999 <= float(k) <= hi * 1.001}


def sel_oracle(sweep):
    """(deployable pick by held-out val, its benefit, test-oracle pick, its benefit)."""
    grid = sorted(float(k) for k in sweep)
    if not grid:
        return None, float("nan"), None, float("nan")
    key = lambda x, f: (sweep[f"{x:g}"][f] if sweep[f"{x:g}"][f] == sweep[f"{x:g}"][f]
                        else float("inf"))
    sel = min(grid, key=lambda x: key(x, "val"))
    orc = min(grid, key=lambda x: key(x, "test"))
    return sel, sweep[f"{sel:g}"]["benefit"], orc, sweep[f"{orc:g}"]["benefit"]


def _surface(sweep):
    """(grid, benefit) with NaNs pushed to -inf so argmax is safe."""
    grid = sorted(float(k) for k in sweep)
    ben = [sweep[f"{x:g}"]["benefit"] for x in grid]
    return grid, [b if b == b else -1e9 for b in ben]


def lr_band(sweep, tol=1.0):
    """DECADES of online LR whose test benefit stays within `tol` points of this optimizer's
    own oracle, as the contiguous run around that oracle. Wide band = safe to deploy on a
    device that cannot afford a rehearsal sweep. Grid-density independent by construction."""
    grid, ben = _surface(sweep)
    i = int(np.argmax(ben))
    if ben[i] < -1e8:                       # every LR failed -> no band to speak of
        return float("nan")
    lo = hi = i
    while lo > 0 and ben[lo - 1] >= ben[i] - tol:
        lo -= 1
    while hi < len(grid) - 1 and ben[hi + 1] >= ben[i] - tol:
        hi += 1
    return float(np.log10(grid[hi] / grid[lo]))


def lr_miss_cost(sweep):
    """Benefit points LOST by mis-setting the online LR by one DECADE either side of the
    oracle (worse side). This is what an un-rehearsed deployment actually risks."""
    grid, ben = _surface(sweep)
    i = int(np.argmax(ben)); worst = ben[i]
    for tgt in (grid[i] * 10, grid[i] / 10):
        j = min(range(len(grid)), key=lambda j: abs(np.log10(grid[j] / tgt)))
        if j != i:
            worst = min(worst, ben[j])
    return ben[i] - worst


def summarize(args):
    L, H = args.L, args.H
    rows = load_jsonl(OUT)
    for src in (OUT_B, OUT_C, OUT_D):             # each stage has its own file; merge per cell
        for k, r in load_jsonl(src).items():
            rows.setdefault(k, {}).update(r)
    ref_rows = load_jsonl(LRFAIR)
    cells = [(k, r) for k, r in rows.items() if k[2] == L and k[3] == H
             and k[0] in args.datasets.split(",")]
    if not cells:
        print("no stage0 rows for this slice yet"); return
    opts = ([(o, "rehearsed") for o in OPTS_LRFUL] + [(o, "default") for o in OPTS_LRFREE]
            + [(o, "rehearsed") for o in OPTS_STAGE0B + OPTS_STAGE0C + OPTS_STAGE0D])

    def stats(vals, adam_ref, sgd_ref, states, bands=(), miss=()):
        vals = np.array(vals, dtype=float)
        ok = ~np.isnan(vals)
        # Paper convention (Table II): benefit < -100% means the stream DIVERGED. Such a cell
        # still counts as negative and is reported in its own column, but is excluded from mean
        # and worst -- one blow-up (-1e18 was observed for UPGD at L=192) otherwise owns both.
        div, fin = ok & (vals < -100), ok & ~(vals < -100)
        out = dict(n=int(ok.sum()), div=int(div.sum()), med=np.nanmedian(vals[ok]),
                   mean=np.nanmean(vals[fin]) if fin.any() else float("nan"),
                   neg=int((vals[ok] < -0.05).sum()),
                   worst=np.nanmin(vals[fin]) if fin.any() else float("nan"))
        out["win_adam"] = int(np.nansum(vals[ok] >= np.array(adam_ref)[ok]))
        out["win_sgdd"] = int(np.nansum(vals[ok] >= np.array(sgd_ref)[ok]))
        out["state"] = np.nanmean(states) if states else float("nan")
        # MEDIAN, not mean: a single diverged cell (benefit ~ -1e15) would otherwise own the
        # average -- the tail is already reported by the neg/worst columns.
        out["band"] = np.nanmedian(bands) if len(bands) else float("nan")
        out["miss"] = np.nanmedian(miss) if len(miss) else float("nan")
        return out

    # reference columns from lr_fairness: rehearsed adam / rehearsed sgd / sgd@default
    table = []
    adam_ref, sgd_def, missing_ref = [], [], 0
    for k, r in cells:
        ref = ref_rows.get(k)
        if ref is None:
            missing_ref += 1; adam_ref.append(np.nan); sgd_def.append(np.nan); continue
        adam_ref.append(ref["sel_benefit_adam"])
        sgd_def.append(ref["sgd"]["0.001"]["benefit"])
    if missing_ref:
        print(f"NOTE: {missing_ref}/{len(cells)} cells missing from lr_fairness.jsonl")
    for label, get, swkey in [
            ("sgd @default 1e-3", lambda ref: ref["sgd"]["0.001"]["benefit"], None),
            ("sgd rehearsed", lambda ref: ref["sel_benefit_sgd"], "sgd"),
            ("adam rehearsed", lambda ref: ref["sel_benefit_adam"], "adam")]:
        vals, bands, miss, wider = [], [], [], []
        for k, r in cells:
            ref = ref_rows.get(k)
            vals.append(get(ref) if ref else np.nan)
            if ref and swkey:
                bands.append(lr_band(ref[swkey])); miss.append(lr_miss_cost(ref[swkey]))
                # sgd/adam's 3e-6..1e-1 surface lives in lr_fairness.jsonl; the matched
                # top-extension rates (3e-1, 1.0) are in the Stage-0b file. Union them so the
                # bracketing question is answered for the references on the same footing as
                # for the contenders -- the headline columns above stay on SHARED_LR.
                ext = r.get(swkey) if isinstance(r.get(swkey), dict) else {}
                if ext:
                    wider.append(lr_band({**ref[swkey], **ext}))
        st = stats(vals, adam_ref, sgd_def, [], bands, miss)
        st["state"] = STATE_MULT.get(swkey or "sgd", float("nan"))
        note = f"[full-grid band {np.nanmedian(wider):.1f}]" if wider else ""
        table.append((label, st, note))
    for o, mode in opts:
        vals, states, sels, bands, miss, wider = [], [], [], [], [], []
        for k, r in cells:
            if mode == "rehearsed":
                # recomputed from the surface on SHARED_LR, not read from sel_benefit_* (which
                # the runner writes over whatever grid that optimizer happened to be given)
                sw = _restrict(r[o]) if isinstance(r.get(o), dict) else {}
                s_lr, s_ben, _, _ = sel_oracle(sw)
                vals.append(s_ben); sels.append(s_lr)
                if sw:
                    bands.append(lr_band(sw)); miss.append(lr_miss_cost(sw))
                    if len(sw) < len(r[o]):          # this optimizer has rates outside SHARED_LR
                        wider.append(lr_band(r[o]))
            else:
                vals.append(r.get(o, {}).get("benefit", np.nan) if isinstance(r.get(o), dict) else np.nan)
            if f"res_{o}" in r and r["backbone"] == "patchtst":
                states.append(r[f"res_{o}"]["opt_state_bytes"] / max(r[f"res_{o}"]["n_adapt_params"], 1) / 4)
        note = ""
        if mode == "rehearsed":
            picked = sorted({f"{s:g}" for s in sels if s is not None})
            note = "sel∈{" + ",".join(picked) + "}"
            if wider:
                note += f"  [full-grid band {np.nanmedian(wider):.1f}]"
        table.append((f"{o} ({mode})", stats(vals, adam_ref, sgd_def, states, bands, miss), note))
    n = len(cells)
    print(f"\n=== Stage 0/0b verdict slice L={L} H={H} ({n} cells) ===")
    print("band = decades of online LR within 1pt of that optimizer's own best (higher = safer "
          "to deploy un-rehearsed)\nmis1x = benefit points lost by a one-decade LR error "
          "(lower = safer)")
    print("mean%/worst% EXCLUDE diverged cells (benefit < -100%); div = how many diverged")
    print(f"\n{'optimizer':25s} {'n':>3s} {'mean%':>7s} {'med%':>6s} {'neg':>4s} {'div':>3s} "
          f"{'worst%':>7s} {'≥Adamr':>6s} {'≥SGDd':>6s} {'state/par':>9s} {'band':>5s} "
          f"{'mis1x':>6s}  note")
    for label, s, note in table:
        print(f"{label:25s} {s['n']:>3d} {s['mean']:>+7.1f} {s['med']:>+6.1f} {s['neg']:>4d} "
              f"{s['div']:>3d} {s['worst']:>+7.1f} {s['win_adam']:>6d} {s['win_sgdd']:>6d} "
              f"{s['state']:>8.2f}x {s['band']:>5.1f} {s['miss']:>6.1f}  {note}")
    errs = {}
    for k, r in cells:
        for key in r:
            if key.startswith("error_"):
                errs.setdefault(key[6:], []).append((k[0], k[1], k[4], r[key]))
    for o, es in errs.items():
        print(f"\nERRORS {o}: {len(es)} cells, e.g. {es[0]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="appliances,bdg2,ETTm2,ETTh2,ETTm1,ETTh1")
    ap.add_argument("--backbones", default="patchtst,dlinear")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--L", type=int, default=96)
    ap.add_argument("--H", type=int, default=24)
    ap.add_argument("--lrs", default=DEFAULT_LRS)
    ap.add_argument("--opts-lrful", default=None)
    ap.add_argument("--opts-lrfree", default=None)
    ap.add_argument("--stage", default="0", choices=["0", "0b", "0c", "0d"],
                    help="0 = memory-light / LR-free general-purpose contenders (stage0_"
                         "optimizers.jsonl); 0b = the non-stationarity-specialised optimizers, "
                         "written to a SEPARATE file so the Stage-0 artifact is never rewritten")
    # G2: which parameters the ONLINE phase adapts. The 216-cell screen is all "full", and
    # that stays the default so every existing artifact is reproduced bit for bit. "head" and
    # "calib" run the same contenders through the conference paper's PEFT strategies, which is
    # what the deployment recipe actually recommends -- the extension has to show its arms on
    # the configuration it tells people to ship, not only on full-model adaptation. Warmup is
    # unchanged (always full-model): only the online phase is restricted.
    ap.add_argument("--which", default="full", choices=["full", "head", "calib"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    b = args.stage in ("0b", "0c", "0d")
    args.opts_lrful = args.opts_lrful if args.opts_lrful is not None else \
        ",".join(OPTS_STAGE0D if args.stage == "0d" else
                 (OPTS_STAGE0C if args.stage == "0c" else
                  (OPTS_STAGE0B if b else OPTS_LRFUL)))
    args.opts_lrfree = args.opts_lrfree if args.opts_lrfree is not None else \
        ("" if b else ",".join(OPTS_LRFREE))
    args.out = args.out or ({"0b": OUT_B, "0c": OUT_C, "0d": OUT_D}.get(args.stage, OUT))
    if args.which != "full":
        # ALWAYS a separate file, even when --out was given explicitly: rows are keyed
        # (dataset, backbone, L, H, seed) with no room for the strategy, so writing a PEFT run
        # into a full-model artifact would silently overwrite the screen's cell.
        base, ext = os.path.splitext(args.out)
        args.out = f"{base}_{args.which}{ext}"
    if args.summarize:
        summarize(args)
    else:
        run_cells(args)


if __name__ == "__main__":
    main()
