"""G1 -- our simplified calibration point vs the official PETSA parameterisation.

The conference paper's Limitations promise this comparison. It is deliberately ONE-SIDED on
purpose: the "ours" side is not recomputed here, it is read from the G2 artifact
(stage0_optimizers_calib.jsonl), which ran `calib` + Adam on exactly these cells, with the same
warmup selection, the same shared LR grid and the same leak-free stream. Recomputing it in a
second script would be a second implementation of the same statistic, which is the failure mode
stage0_pool.py exists to prevent.

So this script runs one arm: PETSA's calibration modules and loss (petsa_calib.py) around the
same frozen forecaster, with Adam, swept on the same grid and selected on the same held-out
pre-drift slice.

  .venv/bin/python experiments/tsf_edge/petsa_compare.py                 # run (resumable)
  .venv/bin/python experiments/tsf_edge/petsa_compare.py --summarize     # read the table

adapt_ms from this file is NOT a timing measurement: it may have been produced while another
job shared the GPU. The extension makes no compute claim (see PAPER_STORY.md).
"""
from __future__ import annotations
import argparse, json, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE))

from online_eval import LR_GRID, VAL_FRAC, load_csv, prep, stream_eval, warm_and_select
from stage0_optimizers import load_jsonl, LRFAIR, RES_KEYS
import petsa_calib

OUT = os.path.join(ROOT, "results", "tsf_edge", "petsa_compare.jsonl")
OURS = os.path.join(ROOT, "results", "tsf_edge", "stage0_optimizers_calib.jsonl")
# matched to the grid stage0_optimizers gives `adam` outside --which full: the shared grid plus
# the declared top extension. Anything narrower would hand PETSA a smaller search than ours.
GRID = list(LR_GRID) + [3e-1, 1.0]


def run(args):
    datasets, seeds = args.datasets.split(","), [int(s) for s in args.seeds.split(",")]
    L, H, bb = args.L, args.H, args.backbone
    rows = load_jsonl(OUT)
    ref = load_jsonl(LRFAIR)
    wrap = petsa_calib.make_wrap()

    def flush():
        tmp = OUT + ".tmp"
        with open(tmp, "w") as f:
            for r in rows.values():
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, OUT)

    todo = [(n, s) for n in datasets for s in seeds
            if len(rows.get((n, bb, L, H, s), {}).get("petsa", {})) < len(GRID)]
    print(f"{len(todo)} cells need work", flush=True)
    cache = {}
    for i, (name, seed) in enumerate(todo, 1):
        if name not in cache:
            cache[name] = load_csv(os.path.join(HERE, "data", f"{name}.csv"))
        t0 = time.perf_counter()
        d, n_warm, C = prep(cache[name], device="cuda")
        n_train = int(n_warm * (1 - VAL_FRAC))
        model, wstep, wval = warm_and_select(bb, L, H, C, d, n_train, n_warm, seed)
        st = stream_eval(model, d, bb, n_warm, L, H, "static", device="cuda")["mse"]
        r = ref.get((name, bb, L, H, seed))
        if r is not None and abs(st - r["static"]) > 1e-3 * max(r["static"], 1e-8):
            print(f"  WARNING: static {st:.5f} != lr_fairness {r['static']:.5f}", flush=True)
        st_use = (r or {}).get("static", st)
        d_val = d[:n_warm]
        row = rows.get((name, bb, L, H, seed)) or dict(dataset=name, backbone=bb, L=L, H=H,
                                                       seed=seed, warmup=wstep, static=st,
                                                       val_static=wval)
        sweep = dict(row.get("petsa", {}))
        for lr in GRID:
            if f"{lr:g}" in sweep:
                continue
            v = stream_eval(model, d_val, bb, n_train, L, H, "wrap_adam", lr=lr, device="cuda",
                            wrap=wrap, loss_fn=petsa_calib.petsa_loss)["mse"]
            te = stream_eval(model, d, bb, n_warm, L, H, "wrap_adam", lr=lr, device="cuda",
                             wrap=wrap, loss_fn=petsa_calib.petsa_loss)
            sweep[f"{lr:g}"] = dict(val=v, test=te["mse"],
                                    benefit=100 * (st_use - te["mse"]) / st_use)
            row["res_petsa"] = {k: te[k] for k in RES_KEYS}
        row["petsa"] = sweep
        rows[(name, bb, L, H, seed)] = row
        flush()
        fin = lambda x: x if x == x else float("inf")
        sel = min(sweep, key=lambda k: fin(sweep[k]["val"]))
        print(f"[{i}/{len(todo)}] {name:11s} s{seed} sel={sel} "
              f"{sweep[sel]['benefit']:+6.1f}%  ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"wrote {OUT} ({len(rows)} rows)")


def pairs():
    """[(dataset, seed, PETSA benefit, its lr, ours benefit, its lr)] for the cells both files
    hold, each side selected on the held-out slice. One implementation, two consumers
    (summarize() and gen_macros_stage0)."""
    rows, ours = load_jsonl(OUT), load_jsonl(OURS)
    fin = lambda x: x if x == x else float("inf")
    tab = []
    for k, r in sorted(rows.items()):
        o = ours.get(k, {}).get("adam", {})
        if not o or "petsa" not in r:
            continue
        p = r["petsa"]
        sp = min(p, key=lambda x: fin(p[x]["val"]))
        so = min(o, key=lambda x: fin(o[x]["val"]))
        tab.append((k[0], k[4], p[sp]["benefit"], float(sp), o[so]["benefit"], float(so)))
    return tab


def n_params():
    """(PETSA, ours) adapted-parameter counts, measured -- MEDIAN over cells, matching
    stage0_pool.peft(). The count varies slightly with the meter count C, so taking the first
    cell here and the median there would print two different numbers for the same quantity."""
    rows, ours = load_jsonl(OUT), load_jsonl(OURS)
    med = lambda d, k: float(np.median([r[k]["n_adapt_params"] for r in d.values() if k in r]))
    return med(rows, "res_petsa"), med(ours, "res_adam")


def summarize(args):
    tab = pairs()
    if not tab:
        print("no comparable cells yet (needs both petsa_compare.jsonl and "
              "stage0_optimizers_calib.jsonl)")
        return
    print(f"{'dataset':12s} {'seed':>4s} {'PETSA%':>8s} {'lr':>8s} {'ours(calib)%':>13s} {'lr':>8s}")
    for name, sd, bp, lp, bo, lo in tab:
        print(f"{name:12s} {sd:4d} {bp:8.2f} {lp:8g} {bo:13.2f} {lo:8g}")
    P, O = np.array([t[2] for t in tab]), np.array([t[4] for t in tab])
    n_par, o_par = n_params()
    print(f"\nn={len(tab)} cells   PETSA mean {P.mean():+.2f}%   ours {O.mean():+.2f}%   "
          f"diff {P.mean() - O.mean():+.2f} pt   PETSA wins {int((P > O).sum())}/{len(tab)}")
    print(f"adapted parameters: PETSA {n_par:.0f}   ours(calib) {o_par:.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="appliances,bdg2,ETTm2,ETTh2,ETTm1,ETTh1")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--backbone", default="patchtst")
    ap.add_argument("--L", type=int, default=96)
    ap.add_argument("--H", type=int, default=24)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    summarize(a) if a.summarize else run(a)
