"""R2 (adversarial review, 2026-08-06): the C3 frontier was single-seed (frontier.py SEED=0),
and seed 0 turned out to be an OUTLIER for the full-model points (ETTm2/PatchTST full-SGD
rehearsed: seed0 +18.0 vs seeds1-4 +4.3..+9.3, 5-seed mean +8.9; Appliances full-SGD the
opposite tail: seed0 +24.7 vs mean +33.8). This script re-measures EVERY frontier point over
5 seeds under the identical fair protocol (warm_and_select + select_online_lr, which now uses
the R1-extended 10-point LR grid), so the paper's Fig. 4 can carry mean +/- std whiskers and
the full-model points stay consistent with the (extended) lr_fairness.jsonl cells.

New-prefix rule: results go to frontier_seeds.jsonl (frontier_data.json, the old seed-0
artifact, is left untouched). One JSON line per (dataset, seed, strategy) point; resumable --
completed (dataset, seed) groups are skipped on restart.
"""
from __future__ import annotations
import json, os, time

import torch

from online_eval import (VAL_FRAC, load_csv, prep, select_online_lr, stream_eval,
                         warm_and_select)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "tsf_edge", "frontier_seeds.jsonl")
DATASETS = ["appliances", "ETTm2"]
COMBOS = [  # (backbone, strategy, label) -- keep labels identical to frontier.py for reuse
    ("patchtst", "full_sgd", "PatchTST full·SGD"),
    ("patchtst", "full_adam", "PatchTST full·Adam"),
    ("patchtst", "head_sgd", "PatchTST head·SGD"),
    ("patchtst", "calib_sgd", "PatchTST calib·SGD"),
    ("dlinear", "full_sgd", "DLinear full·SGD"),
    ("dlinear", "head_sgd", "DLinear head·SGD"),
    # R3 (adversarial review, 2026-08-08): the recipe recommends BOTH Adam at its rehearsed
    # rate AND updating few parameters, but their combination was never on the frontier -- so
    # the memory axis had only 4 B/param (SGD) and 12 B/param (Adam) and nothing between.
    # calib/head x Adam are the PEFT x Adam points; SGD+momentum is the missing 8 B/param rung.
    # stream_eval resolves "<prefix>_<optimizer>" generically, so no new STRATEGIES entry is
    # needed and the optimizer state is MEASURED rather than assumed for these points.
    ("patchtst", "calib_adam", "PatchTST calib·Adam"),
    ("patchtst", "head_adam", "PatchTST head·Adam"),
    ("patchtst", "full_sgdm", "PatchTST full·SGD+m"),
    ("dlinear", "head_adam", "DLinear head·Adam"),
    ("dlinear", "full_sgdm", "DLinear full·SGD+m"),
    # completes the strategy x optimizer grid: every parameter subset now has all three
    # optimizers, so no cell of the recipe is inferred rather than observed.
    ("patchtst", "head_sgdm", "PatchTST head·SGD+m"),
    ("patchtst", "calib_sgdm", "PatchTST calib·SGD+m"),
    ("dlinear", "full_adam", "DLinear full·Adam"),
    ("dlinear", "head_sgdm", "DLinear head·SGD+m"),
]
SEEDS = [0, 1, 2, 3, 4]
L, H, dev = 96, 24, "cuda"


def main():
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                r = json.loads(line)
                done.add((r["dataset"], r["seed"], r["label"]))
            except json.JSONDecodeError:
                pass
    for name in DATASETS:
        data = load_csv(os.path.join(ROOT, "experiments/tsf_edge/data", f"{name}.csv"))
        for seed in SEEDS:
            todo = [c for c in COMBOS if (name, seed, c[2]) not in done]
            if not todo:
                continue
            t0 = time.perf_counter()
            d, n_warm, C = prep(data, device=dev)
            n_train = int(n_warm * (1 - VAL_FRAC))
            warmed, base = {}, {}
            for bb in {c[0] for c in todo}:
                model, wstep, _ = warm_and_select(bb, L, H, C, d, n_train, n_warm, seed)
                base[bb] = stream_eval(model, d, bb, n_warm, L, H, "static", device=dev)["mse"]
                warmed[bb] = (model, wstep)
            for bb, strat, lab in todo:
                model, wstep = warmed[bb]
                sel, _ = select_online_lr(model, d, bb, n_train, n_warm, L, H, strat,
                                          device=dev)
                r = stream_eval(model, d, bb, n_warm, L, H, strat, lr=sel, device=dev)
                r0 = stream_eval(model, d, bb, n_warm, L, H, strat, device=dev)  # fixed 1e-3
                row = dict(dataset=name, seed=seed, label=lab, backbone=bb, strategy=strat,
                           L=L, H=H, warmup=wstep, static=base[bb],
                           params=r["n_adapt_params"], ms=r["adapt_ms"], lr=sel,
                           opt_state_bytes=r["opt_state_bytes"],
                           peak_adapt_mem_kb=r["peak_adapt_mem_kb"],
                           benefit=100 * (base[bb] - r["mse"]) / base[bb],
                           benefit_fixed=100 * (base[bb] - r0["mse"]) / base[bb])
                with open(OUT, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"{name:11s} s{seed} {lab:20s} warm={wstep:5d} lr={sel:g} "
                      f"benefit={row['benefit']:+6.1f}% (fixed {row['benefit_fixed']:+6.1f}%)",
                      flush=True)
            print(f"  [{name} s{seed}] {time.perf_counter() - t0:5.0f}s", flush=True)
    print(f"frontier_seeds done -> {OUT}")


if __name__ == "__main__":
    main()
