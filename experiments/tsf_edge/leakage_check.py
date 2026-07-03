"""(0) Verify our harness is leakage-free (DSOF, ICLR'25) and separate it from the warmup
confound. DSOF's leak: a model is scored on steps already used in backprop. It arises from
OVERLAPPING multi-step windows (stride < H) with per-step adaptation: the overlap y[t+s:t+H]
is trained on at step t, then re-scored at step t+s. Our default stride=H (non-overlapping,
score-before-adapt) has no such overlap ⇒ leakage-free by construction.

Here we confirm it: run stride=1 (leaky, overlapping) vs stride=H (clean) and show the leaky
protocol INFLATES the reported adaptation benefit. This (a) validates our default is clean,
(b) reproduces the DSOF leak effect, (c) shows the warmup confound (a BASELINE-budget issue)
is orthogonal to leakage (an ONLINE-phase issue). Warm once per (dataset,backbone) for speed.
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

print(f"{'dataset':11s} {'backbone':9s} {'protocol':16s} {'static':>8s} {'adapted':>8s} {'benefit%':>9s}")
dump = {}                                   # -> leakage_check.json for gen_macros.py
for name, bb in COMBOS:
    data = load_csv(os.path.join(ROOT, "experiments/tsf_edge/data", f"{name}.csv"))
    torch.manual_seed(0); np.random.seed(0)
    d, n_warm, C = prep(data, device=dev)
    model = warmup_model(bb, L, H, C, d, n_warm, WARM, device=dev)      # warm ONCE
    bens, row = {}, {}
    for stride, tag in [(1, "LEAKY(stride1)"), (H, "clean(strideH)")]:
        s = stream_eval(model, d, bb, n_warm, L, H, "static", device=dev, stride=stride)["mse"]
        a = stream_eval(model, d, bb, n_warm, L, H, "full_sgd", device=dev, stride=stride)["mse"]
        ben = 100 * (s - a) / s
        bens[tag] = ben
        key = "leaky" if stride == 1 else "clean"
        row[key + "_static"], row[key + "_adapted"], row[key + "_benefit"] = s, a, ben
        print(f"{name:11s} {bb:9s} {tag:16s} {s:>8.4f} {a:>8.4f} {ben:>+8.1f}%")
    infl = bens["LEAKY(stride1)"] - bens["clean(strideH)"]
    row["inflation_pt"] = infl
    dump[f"{name}|{bb}"] = row
    print(f"{'':11s} {'':9s} -> leak inflates benefit by {infl:+.1f} pts "
          f"(clean = our default = leakage-free)\n")

json.dump(dump, open(os.path.join(ROOT, "results", "tsf_edge", "leakage_check.json"), "w"), indent=2)
print("saved", os.path.join(ROOT, "results", "tsf_edge", "leakage_check.json"))

print("Verdict: if LEAKY benefit >> clean benefit, our default stride=H protocol is the "
      "leakage-free one; the warmup confound is measured ON TOP of it, orthogonally.")
