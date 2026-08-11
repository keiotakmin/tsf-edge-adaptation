"""Per-update adaptation wall-clock for the C3 frontier and the W1 scale points, measured
under controlled GPU state.

WHY THIS EXISTS (2026-08-10). The timings carried in frontier_seeds.jsonl and
scale_timing_sgdm.json are the MEAN over every update of one sequential run per arm, and that
estimator has two defects that together invert the SGD+m-vs-Adam ordering on Appliances:

  (1) The mean includes a start-of-stream transient. The first update of a stream costs up to
      223 ms against a 3.5 ms steady state, and the arm needs O(100) updates to settle, so the
      contamination scales as 1/n_updates -- Appliances has 407 updates, ETTm2 1447, i.e. ETTm2
      dilutes the same transient 3.6x more. That is exactly why the anomaly is visible on the
      Appliances panel and hidden on the ETTm2 one.
  (2) Arms are measured one after another while the GPU clock is still ramping, so whichever
      arm runs first is penalised by ~0.5 ms. Swapping the run order FLIPS THE SIGN of the
      measured Adam-minus-SGD+m difference (-0.275 ms with SGD+m first, +0.269 ms with Adam
      first, on appliances/full).

The true difference is not in doubt: the two arms share a bit-identical forward and backward
pass and differ only in opt.step(), where Adam does strictly more arithmetic. Isolated,
warmed, min-of-repeats: 358 vs 117 us (full), 145 vs 61 (head), 161 vs 65 (calib). The stored
data also contains impossible orderings -- plain SGD, which carries NO optimizer state,
measures slower than Adam in 6 of 50 same-session pairs.

METHOD HERE. (a) A fixed pre-load brings the GPU to steady clocks before anything is recorded.
(b) Arms are INTERLEAVED and the whole cycle is repeated, so residual drift hits every arm
alike. (c) Within a run the first DISCARD updates are dropped and the MEDIAN over the rest is
taken, not the mean -- robust to the transient and to occasional OS/GPU hiccups. (d) The
reported value is the median over cycles; the cycle spread is recorded so the stability of the
measurement can be checked rather than assumed.

Per-update cost does not depend on the weights (the forward/backward is identical across arms
and across training states), so no warmup training is needed -- verified: fwd+bwd differs by
<30 us between the two optimizer arms of the same strategy.

-> results/tsf_edge/frontier_timing.json   (NEW file; the two source files are left untouched)
Run from the project root:  .venv/bin/python experiments/tsf_edge/frontier_timing.py
"""
from __future__ import annotations
import json, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frontier import COMBOS, DATASETS
from online_eval import build_model, load_csv, make_online_optimizer, prep, set_trainable

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
L, H, dev = 96, 24, "cuda"
CYCLES = 9              # interleaved repeats of the whole point set
DISCARD = 30            # updates dropped at the start of each run (transient)
MAX_UPDATES = 150       # short burst: the estimator is a MINIMUM over cycles, so each point
                        # needs several chances to land in a quiet window rather than one long
                        # window that is certain to overlap some background load
PRELOAD_S = 30.0        # continuous work before the first measurement, to settle GPU clocks
SCALE_SETS = ["bdg2", "bdg2_fleet", "bdg2_rat_all"]     # W1 scalability points

WHICH = {"full": "all", "head": "head", "calib": "calib"}
OUT = os.path.join(ROOT, "results", "tsf_edge", "frontier_timing.json")


def load_timing():
    """{(dataset, label): ms} from this controlled measurement, or {} if it has not been run.
    Consumers (gen_macros, paper_figs) fall back to the in-stream mean carried in
    frontier_seeds.jsonl / scale_timing_sgdm.json, which is the estimator this file exists to
    replace -- see the module docstring for why it is not trustworthy."""
    if not os.path.exists(OUT):
        return {}
    return {tuple(k.split("|")): e["ms"] for k, e in json.load(open(OUT))["points"].items()}


def measure(d, n_warm, C, backbone, strategy, lr):
    """Median per-update wall-clock over the stream, warm-up updates discarded."""
    torch.manual_seed(0)
    model = build_model(backbone, L, H, C).to(dev)
    subset = strategy.rsplit("_", 1)[0]
    set_trainable(model, backbone, WHICH[subset])
    okind = strategy.rsplit("_", 1)[1]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = make_online_optimizer(okind, params, lr)
    ts, t = [], n_warm + L
    while t + H <= len(d) and len(ts) < DISCARD + MAX_UPDATES:
        xa, ya = d[t - L:t].unsqueeze(0), d[t:t + H].unsqueeze(0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad()
        F.mse_loss(model(xa), ya).backward()
        opt.step()
        torch.cuda.synchronize()
        ts.append(1000 * (time.perf_counter() - t0))
        t += H
    finite = all(torch.isfinite(p).all().item() for p in params)
    v = np.array(ts[DISCARD:]) if len(ts) > DISCARD else np.array(ts)
    return float(np.median(v)), len(v), finite


def main():
    # every point measured in this sweep: the frontier combos, then the W1 scale points
    points = [(ds, bb, strat, lab) for ds in DATASETS for bb, strat, lab, _, _ in COMBOS]
    points += [(ds, "patchtst", s, f"scale {s}") for ds in SCALE_SETS
               for s in ("full_sgdm", "full_adam")]

    # deployed rates, so the measurement matches the reported configuration (timing is
    # rate-independent; this only removes any doubt about that)
    lrs = {}
    for line in open(os.path.join(ROOT, "results/tsf_edge/frontier_seeds.jsonl")):
        r = json.loads(line)
        lrs.setdefault((r["dataset"], r["label"]), r["lr"])
    lrs.update({(ds, "scale full_sgdm"): 1e-3 for ds in SCALE_SETS})
    lrs.update({(ds, "scale full_adam"): 1e-4 for ds in SCALE_SETS})

    prepped = {}
    for ds in {p[0] for p in points}:
        data = load_csv(os.path.join(ROOT, "experiments/tsf_edge/data", f"{ds}.csv"))
        prepped[ds] = prep(data, device=dev)

    ds0 = DATASETS[0]
    d0, n_warm0, C0 = prepped[ds0]
    print(f"pre-loading the GPU for {PRELOAD_S:.0f}s to settle clocks ...")
    t_end = time.time() + PRELOAD_S
    while time.time() < t_end:
        measure(d0, n_warm0, C0, "patchtst", "full_adam", 1e-4)

    cyc = {}
    for c in range(CYCLES):
        t0 = time.time()
        for ds, bb, strat, lab in points:
            d, n_warm, C = prepped[ds]
            ms, n, finite = measure(d, n_warm, C, bb, strat, lrs.get((ds, lab), 1e-4))
            cyc.setdefault((ds, lab), {"backbone": bb, "strategy": strat,
                                       "channels": C, "n_timed": n, "ms_cycles": []})
            cyc[(ds, lab)]["ms_cycles"].append(ms)
            if not finite:
                cyc[(ds, lab)]["nonfinite"] = True
        print(f"  cycle {c + 1}/{CYCLES} done ({time.time() - t0:.0f}s)")

    # Estimator = MINIMUM over cycles. The per-update cost is CPU-launch-bound (a few hundred
    # tiny kernels per update), so every sample is the true cost PLUS whatever contention the
    # shared host happened to add; the minimum is the least-contaminated observation, and
    # `near_min` reports how many cycles landed within 5% of it, i.e. whether the floor is a
    # stable plateau or a lucky outlier.
    out = {}
    print(f"\n{'dataset':<15}{'point':<24}{'ms':>7}{'near_min':>9}{'spread':>8}{'n':>6}")
    for (ds, lab), e in cyc.items():
        v = np.array(e["ms_cycles"])
        e["ms"] = float(v.min())
        e["ms_near_min"] = int((v <= 1.05 * v.min()).sum())
        e["ms_spread"] = float(v.max() - v.min())
        out[f"{ds}|{lab}"] = e
        flag = "  NONFINITE" if e.get("nonfinite") else ""
        print(f"{ds:<15}{lab:<24}{e['ms']:>7.3f}{e['ms_near_min']:>6}/{CYCLES}"
              f"{e['ms_spread']:>8.3f} {e['n_timed']:>6}{flag}")

    # sanity gate: Adam must never measure faster than SGD+m at the same strategy
    bad = []
    for (ds, lab), e in cyc.items():
        if "SGD+m" not in lab and "sgdm" not in lab:
            continue
        twin = lab.replace("SGD+m", "Adam").replace("sgdm", "adam")
        if (ds, twin) in cyc and cyc[(ds, twin)]["ms"] < e["ms"]:
            bad.append(f"{ds} {lab} {e['ms']:.3f} > {twin} {cyc[(ds, twin)]['ms']:.3f}")
    print("\nsanity (Adam >= SGD+m per-update): " + ("OK" if not bad else "VIOLATED"))
    for b in bad:
        print("   " + b)

    json.dump(dict(meta=dict(cycles=CYCLES, discard=DISCARD, max_updates=MAX_UPDATES,
                             preload_s=PRELOAD_S, estimator="median over updates within a "
                             "burst, MINIMUM over interleaved cycles"),
                   points=out), open(OUT, "w"), indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
