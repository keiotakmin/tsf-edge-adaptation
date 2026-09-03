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


def measure(d, n_warm, C, backbone, strategy, lr, opt_factory=None):
    """Median per-update wall-clock over the stream, warm-up updates discarded."""
    torch.manual_seed(0)
    model = build_model(backbone, L, H, C).to(dev)
    # partition, NOT rsplit. A strategy is "<subset>_<optimizer>" and an optimizer name may
    # itself contain underscores, which rsplit splits on the wrong one -- it would take the
    # trailing fragment as the optimizer and silently measure something else. online_eval.
    # stream_eval has always used partition; this file disagreed with it until 2026-08-31.
    subset, _, okind = strategy.partition("_")
    set_trainable(model, backbone, WHICH[subset])
    params = [p for p in model.parameters() if p.requires_grad]
    # opt_factory lets a caller supply its own builder -- e.g. to hold the IMPLEMENTATION
    # fixed across arms (torch's built-ins default to the multi-tensor path, a hand-written
    # rule does not) so that a comparison is between update rules rather than between the
    # people who wrote their kernels. Default is the project factory.
    opt = (opt_factory or make_online_optimizer)(okind, params, lr)
    # Some update rules bound the step using the SCALAR LOSS and therefore take it as an
    # argument; online_eval.stream_eval branches on the same `needs_loss` flag. Calling step()
    # bare raises for those, and this timing path had only ever driven torch built-ins.
    needs_loss = getattr(opt, "needs_loss", False)
    ts, t = [], n_warm + L
    while t + H <= len(d) and len(ts) < DISCARD + MAX_UPDATES:
        xa, ya = d[t - L:t].unsqueeze(0), d[t:t + H].unsqueeze(0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad()
        loss = F.mse_loss(model(xa), ya)
        loss.backward()
        opt.step(loss=loss.detach()) if needs_loss else opt.step()
        torch.cuda.synchronize()
        ts.append(1000 * (time.perf_counter() - t0))
        t += H
    finite = all(torch.isfinite(p).all().item() for p in params)
    v = np.array(ts[DISCARD:]) if len(ts) > DISCARD else np.array(ts)
    return float(np.median(v)), len(v), finite, len(params)


def run_cycles(points, prepped, lrs, cycles, opt_factory=None):
    """Interleave every point and repeat the whole set `cycles` times, so residual drift on the
    host hits every arm alike. Points are (dataset, backbone, strategy, label); `opt_factory`
    is passed through to measure(). Returns {(dataset, label): record with ms_cycles}."""
    cyc = {}
    for c in range(cycles):
        t0 = time.time()
        for ds, bb, strat, lab in points:
            d, n_warm, C = prepped[ds]
            ms, n, finite, ntens = measure(d, n_warm, C, bb, strat, lrs.get((ds, lab), 1e-4),
                                           opt_factory=opt_factory)
            cyc.setdefault((ds, lab), {"backbone": bb, "strategy": strat,
                                       "channels": C, "n_timed": n, "ms_cycles": [],
                                       "n_param_tensors": ntens})
            cyc[(ds, lab)]["ms_cycles"].append(ms)
            if not finite:
                cyc[(ds, lab)]["nonfinite"] = True
        print(f"  cycle {c + 1}/{cycles} done ({time.time() - t0:.0f}s)", flush=True)
    return cyc


def preload(prepped, seconds=None):
    """Bring the device to steady clocks before anything is recorded."""
    d0, n_warm0, C0 = prepped[DATASETS[0]]
    seconds = PRELOAD_S if seconds is None else seconds
    print(f"pre-loading the GPU for {seconds:.0f}s to settle clocks ...", flush=True)
    t_end = time.time() + seconds
    while time.time() < t_end:
        measure(d0, n_warm0, C0, "patchtst", "full_adam", 1e-4)


def main(out_path=None, cycles=None):
    # OUT is what load_timing() reads, i.e. what macros.tex and the frontier figure are built
    # from. Re-measuring in a NEW session lands tens of microseconds away from the stored
    # values, so writing back to OUT would silently move numbers that are already in a
    # submitted paper. Any exploratory re-measurement should therefore be given its own file
    # with --out, and only a deliberate, documented re-run should overwrite OUT.
    out_path = out_path or OUT
    cycles = cycles or CYCLES
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

    preload(prepped)

    cyc = run_cycles(points, prepped, lrs, cycles)

    # Estimator = MEDIAN over cycles (changed 2026-08-31; the code was taking the MINIMUM
    # while this module's own docstring said median -- the code had drifted from its
    # documentation. The stored melon run is minima, but its cycles span 3.69-3.72 ms, so
    # min and median differ there by under 1% and its published numbers do not move).
    #
    # The minimum rested on "contention can only make a reading slower, so the smallest sample
    # is the cleanest". On iris that premise is false. In one sweep, cycle 4 contained a ~20 s
    # window in which six CONSECUTIVE points all measured about 2.3x faster than the same
    # points in the other eight cycles, and 12 of 64 points ended up reporting one of these
    # transients as their value -- ETTm2 PatchTST full-SGD+m came out at 3.156 ms against
    # 7.478 in every other cycle. Measuring that one configuration 30 times back to back gives
    # 7.57-7.71 ms with no low outlier at all, so 7.67 is the steady state and 3.16 was the
    # excursion. The host is an AMD EPYC 7543P under the schedutil governor; melon, where the
    # BigData numbers were taken, is an Intel Xeon Silver 4310 under powersave. A launch-bound
    # measurement follows the CPU, and this one apparently follows its frequency excursions in
    # BOTH directions.
    #
    # The median is robust to excursions either way. `ms_min`, `ms_spread` and `near_min` stay
    # in the output as the stability diagnostics they always were -- a point whose min sits far
    # below its median is a point measured on a host that was not in one state throughout.
    out = {}
    print(f"\n{'dataset':<15}{'point':<26}{'ms':>7}{'near_min':>9}{'spread':>8}{'n':>6}  impl")
    for (ds, lab), e in cyc.items():
        v = np.array(e["ms_cycles"])
        e["ms"] = float(np.median(v))
        e["ms_min"] = float(v.min())
        e["ms_near_min"] = int((v <= 1.05 * v.min()).sum())
        e["ms_spread"] = float(v.max() - v.min())
        e["ms_unstable"] = bool(v.min() < 0.9 * np.median(v))   # excursion, not a clean floor
        out[f"{ds}|{lab}"] = e
        flag = ("  NONFINITE" if e.get("nonfinite") else "") + \
               ("  UNSTABLE-HOST" if e["ms_unstable"] else "")
        print(f"{ds:<15}{lab:<26}{e['ms']:>7.3f}{e['ms_near_min']:>6}/{cycles}"
              f"{e['ms_spread']:>8.3f} {e['n_timed']:>6}  {e.get('impl', '')}{flag}")

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

    # Second gate: a host that was not in ONE state for the whole sweep cannot be published
    # from. `ms_unstable` marks a point whose fastest cycle sits more than 10% below its own
    # median -- i.e. the machine briefly ran faster, so the point's cycles are not samples of
    # one thing. On 2026-08-31 an iris sweep had 12 of 64 points like this, clustered in a
    # ~20 s window inside one cycle, and one configuration came out at 3.156 ms there against
    # 7.478 ms in every other cycle; measuring it 30 times back to back gave 7.57-7.71 with no
    # low outlier, so the excursion was the artifact and not the floor.
    n_unstable = sum(1 for e in cyc.values() if e["ms_unstable"])
    verdict = "USABLE" if not (bad or n_unstable) else "DO NOT PUBLISH"
    print(f"\nVERDICT: {verdict}   ({n_unstable}/{len(cyc)} points flagged UNSTABLE-HOST)")

    json.dump(dict(meta=dict(cycles=cycles, discard=DISCARD, max_updates=MAX_UPDATES,
                             preload_s=PRELOAD_S, estimator="median over updates within a "
                             "burst, MEDIAN over interleaved cycles",
                             host_note="per-update cost at batch 1 is kernel-launch "
                                       "bound, so it follows the HOST's CPU: readings are not "
                                       "comparable across machines, and the ratio between two "
                                       "machines is not even constant across configurations."),
                   points=out), open(out_path, "w"), indent=1)
    print("saved", out_path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="output json; defaults to the file load_timing() reads. Give an "
                         "exploratory re-measurement its own path so the stored timings, "
                         "which are already in a submitted paper, do not move.")
    ap.add_argument("--cycles", type=int, default=None)
    a = ap.parse_args()
    main(out_path=a.out, cycles=a.cycles)
