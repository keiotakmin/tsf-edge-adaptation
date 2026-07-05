"""(0) Verify our harness is leakage-free (DSOF, ICLR'25) and DECOMPOSE the leaky-vs-clean
difference (referee M3). DSOF's leak: a model is scored on steps already used in backprop. It
arises from OVERLAPPING multi-step windows (stride < H) with per-step adaptation: the overlap
y[t+s:t+H] is trained on at step t, then re-scored at step t+s. Our default stride=H
(non-overlapping, score-before-adapt) has no such overlap ⇒ leakage-free by construction.

Three arms per (dataset, backbone), warmed once (referee M3: leaky-vs-clean alone confounds
the leak with the eval-window SET and the update FREQUENCY, both of which change with stride):
  leaky    stride=1, adapt on the just-scored window      -> leak + dense eval + dense updates
  delayed  stride=1, adapt on the trailing FULLY-REVEALED  -> NO leak, same dense eval set and
           window (adapt_on="trailing")                       same update count as leaky
  clean    stride=H (our default protocol)                 -> no leak, sparse eval + updates
Decomposition: leak_pt = leaky - delayed  (the information leak proper, everything else held
fixed); evalset_pt = delayed - clean (protocol-density difference absent any leak);
inflation_pt = leaky - clean (their sum; the headline number of the 2-arm version).
This (a) validates our default is clean, (b) reproduces the DSOF leak effect with the
confound removed, (c) shows the warmup confound (a BASELINE-budget issue) is orthogonal to
leakage (an ONLINE-phase issue). Warm once per (dataset,backbone) for speed.
"""
from __future__ import annotations
import json, os
import numpy as np
import torch

from online_eval import load_csv, prep, warmup_model, stream_eval

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMBOS = [("ETTh2", "patchtst"), ("appliances", "patchtst"),
          ("ETTh2", "dlinear"), ("appliances", "dlinear")]
L, H, WARM, dev = 96, 24, 2000, "cuda"

print(f"{'dataset':11s} {'backbone':9s} {'protocol':18s} {'static':>8s} {'adapted':>8s} {'benefit%':>9s}")
dump = {}                                   # -> leakage_check.json for gen_macros.py
for name, bb in COMBOS:
    data = load_csv(os.path.join(ROOT, "experiments/tsf_edge/data", f"{name}.csv"))
    torch.manual_seed(0); np.random.seed(0)
    d, n_warm, C = prep(data, device=dev)
    model = warmup_model(bb, L, H, C, d, n_warm, WARM, device=dev)      # warm ONCE
    s1 = stream_eval(model, d, bb, n_warm, L, H, "static", device=dev, stride=1)["mse"]
    sH = stream_eval(model, d, bb, n_warm, L, H, "static", device=dev, stride=H)["mse"]
    row, bens = {}, {}
    for key, stride, adapt_on, s in [("leaky", 1, "current", s1),
                                     ("delayed", 1, "trailing", s1),
                                     ("clean", H, "current", sH)]:
        a = stream_eval(model, d, bb, n_warm, L, H, "full_sgd", device=dev,
                        stride=stride, adapt_on=adapt_on)["mse"]
        ben = 100 * (s - a) / s
        bens[key] = ben
        row[key + "_static"], row[key + "_adapted"], row[key + "_benefit"] = s, a, ben
        tag = {"leaky": "LEAKY(stride1)", "delayed": "delayed(stride1)",
               "clean": "clean(strideH)"}[key]
        print(f"{name:11s} {bb:9s} {tag:18s} {s:>8.4f} {a:>8.4f} {ben:>+8.1f}%")
    row["leak_pt"] = bens["leaky"] - bens["delayed"]        # the information leak proper
    row["evalset_pt"] = bens["delayed"] - bens["clean"]     # eval-set/frequency difference
    row["inflation_pt"] = bens["leaky"] - bens["clean"]     # their sum (2-arm headline)
    dump[f"{name}|{bb}"] = row
    print(f"{'':11s} {'':9s} -> leak proper {row['leak_pt']:+.1f} pt | eval-set/frequency "
          f"{row['evalset_pt']:+.1f} pt | total {row['inflation_pt']:+.1f} pt\n")

json.dump(dump, open(os.path.join(ROOT, "results", "tsf_edge", "leakage_check.json"), "w"), indent=2)
print("saved", os.path.join(ROOT, "results", "tsf_edge", "leakage_check.json"))

print("Verdict: leak_pt isolates the DSOF information leak with eval set and update count "
      "held fixed; our default stride=H protocol remains the leakage-free one.")
