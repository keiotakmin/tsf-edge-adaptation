"""M1 (referee) rebuttal grid: does the SGD-vs-Adam asymmetry survive FAIR per-cell
online-LR selection?

The C3 grid ran BOTH online optimizers at a fixed lr=1e-3 (Adam's canonical default), so the
headline asymmetry ("SGD never below static, Adam negative in 48% of cells") could in principle
be an LR artifact (referee M1). Here we select each optimizer's online LR per cell the same way
C1 selects the warmup budget -- on the HELD-OUT PRE-DRIFT VALIDATION SLICE, never on test:
after fair warmup (`warm_and_select`), we REHEARSE online adaptation over the validation stream
(same score-then-adapt protocol, stride=H) at each candidate LR and pick, per optimizer, the LR
with the lowest validation ONLINE MSE. The full LR x {val,test} surface is logged, so the
analysis can report three readings per cell:
  fixed-1e-3    reproduces the corresponding C3 grid cell (sanity check),
  val-selected  the deployable-fair comparison (the M1 answer),
  test-oracle   upper bound: even an unfairly per-cell-oracle-tuned Adam.
Betas stay at Adam defaults (0.9, 0.999); LR is the first-order factor M1 contests.

RESULT (round 1, L=96): M1 confirmed -- see FINDINGS.md "Referee M1 response". Round 2 adds
(i) a downward grid extension to 3e-6 (10/36 Adam val-picks sat on the old 3e-5 bottom edge)
and (ii) the L=192 slice (the "Adam corner shrinks with lookback" claim, same-artifact suspect).

One JSONL row per cell -> results/tsf_edge/lr_fairness.jsonl. Resumable AND mergeable: a cell
already in the file is re-warmed (deterministic; verified to reproduce grid.jsonl) and only its
MISSING LRs are computed, then sel/oracle are re-derived over the merged surface. New-point
benefits reuse the stored static baseline (a loud warning fires if the re-warmed static drifts).
Scope: the H=24 / 3-seed slice of the C3 grid at --L 96 (default) or --L 192;
datasets/backbones/seeds/lrs overridable for smoke tests.
"""
from __future__ import annotations
import argparse, json, os, time

from online_eval import LR_GRID, VAL_FRAC, load_csv, prep, stream_eval, warm_and_select

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "tsf_edge", "lr_fairness.jsonl")
DEFAULT_LRS = ",".join(f"{x:g}" for x in LR_GRID)   # 3.5 decades; bottom extended past the
                                                    # round-1 3e-5 edge (censoring resolved)
dev = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="appliances,bdg2,ETTm2,ETTh2,ETTm1,ETTh1")
    ap.add_argument("--backbones", default="patchtst,dlinear")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--L", type=int, default=96)
    ap.add_argument("--H", type=int, default=24)
    ap.add_argument("--lrs", default=DEFAULT_LRS)
    args = ap.parse_args()
    L, H = args.L, args.H
    datasets = args.datasets.split(",")
    backbones = args.backbones.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    lrs = [float(x) for x in args.lrs.split(",")]

    all_rows = {}                 # (dataset, backbone, L, H, seed) -> row, across ALL slices
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                r = json.loads(line)
                all_rows[(r["dataset"], r["backbone"], r["L"], r["H"], r["seed"])] = r
            except json.JSONDecodeError:
                pass                  # drop a partial last line from an interrupted run

    def flush():
        tmp = OUT + ".tmp"
        with open(tmp, "w") as f:
            for r in all_rows.values():
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, OUT)

    todo = []
    for name in datasets:
        for bb in backbones:
            for seed in seeds:
                old = all_rows.get((name, bb, L, H, seed))
                need = [lr for lr in lrs if old is None or f"{lr:g}" not in old["sgd"]]
                if need:
                    todo.append((name, bb, seed, need))
    print(f"L={L} H={H}: {len(todo)} cells need work "
          f"({sum(len(n) for *_, n in todo)} missing LR points)", flush=True)

    data_cache = {}
    for i, (name, bb, seed, need) in enumerate(todo, 1):
        if name not in data_cache:
            data_cache[name] = load_csv(os.path.join(ROOT, "experiments/tsf_edge/data", f"{name}.csv"))
        t0 = time.perf_counter()
        d, n_warm, C = prep(data_cache[name], device=dev)
        n_train = int(n_warm * (1 - VAL_FRAC))
        model, wstep, wval = warm_and_select(bb, L, H, C, d, n_train, n_warm, seed)
        st = stream_eval(model, d, bb, n_warm, L, H, "static", device=dev)["mse"]
        old = all_rows.get((name, bb, L, H, seed))
        if old is not None and abs(st - old["static"]) > 1e-3 * max(old["static"], 1e-8):
            print(f"  WARNING: re-warmed static {st:.5f} != stored {old['static']:.5f} "
                  f"({name}/{bb}/s{seed}) -- merged surface may be inconsistent", flush=True)
        st_use = old["static"] if old is not None else st
        d_val = d[:n_warm]           # validation stream = the held-out pre-drift slice
        row = old if old is not None else dict(dataset=name, backbone=bb, L=L, H=H, seed=seed,
                                               warmup=wstep, static=st, val_static=wval)
        for okind in ("sgd", "adam"):
            strat = f"full_{okind}"
            sweep = dict(row.get(okind, {}))
            for lr in need:          # rehearse on val, then measure on test, per missing LR
                v = stream_eval(model, d_val, bb, n_train, L, H, strat, lr=lr,
                                device=dev)["mse"]
                te = stream_eval(model, d, bb, n_warm, L, H, strat, lr=lr,
                                 device=dev)["mse"]
                sweep[f"{lr:g}"] = dict(val=v, test=te, benefit=100 * (st_use - te) / st_use)
            row[okind] = sweep
            grid = sorted(float(k) for k in sweep)
            sel = min(grid, key=lambda x: sweep[f"{x:g}"]["val"])
            orc = min(grid, key=lambda x: sweep[f"{x:g}"]["test"])
            row[f"sel_lr_{okind}"] = sel
            row[f"sel_benefit_{okind}"] = sweep[f"{sel:g}"]["benefit"]
            row[f"oracle_lr_{okind}"] = orc
            row[f"oracle_benefit_{okind}"] = sweep[f"{orc:g}"]["benefit"]
        row["lrs"] = sorted(float(k) for k in row["sgd"])
        all_rows[(name, bb, L, H, seed)] = row
        flush()
        win = "SGD" if row["sel_benefit_sgd"] >= row["sel_benefit_adam"] else "Adam"
        print(f"[{i:2d}/{len(todo)}] L{L} H{H} {name:11s} {bb:9s} s{seed} warm={wstep:5d} "
              f"(+{len(need)} lrs) | SGD sel={row['sel_lr_sgd']:g} {row['sel_benefit_sgd']:+6.1f}% | "
              f"Adam sel={row['sel_lr_adam']:g} {row['sel_benefit_adam']:+6.1f}% "
              f"(@1e-3 {row['adam']['0.001']['benefit']:+6.1f}%) | win={win:4s} | "
              f"{time.perf_counter() - t0:5.0f}s", flush=True)
    print(f"\nwrote {OUT} ({len(all_rows)} rows total)")


if __name__ == "__main__":
    main()
