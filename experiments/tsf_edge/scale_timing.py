"""W1 scalability measurement: per-update adaptation wall-clock vs channel count.

PatchTST is channel-independent (each channel is a separate sequence in the batch), so the
per-device update cost should grow linearly in the number of meters; this script measures it
on the 15-meter shipped subset, the 240-meter fleet, and the 280-meter full site, for both
online optimizers at plateau-appropriate rates. Timing only -- a short fixed warmup suffices
(update cost does not depend on model quality). GPU should be otherwise idle.
-> results/tsf_edge/scale_timing.json (consumed by gen_macros.py, macros \\Sc*).
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from online_eval import load_csv, prep, stream_eval, warmup_model

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASETS = ["bdg2", "bdg2_fleet", "bdg2_rat_all"]
BB, L, H, WARM, dev = "patchtst", 96, 24, 500, "cuda"

dump = {}
print(f"{'dataset':14s} {'C':>4s} {'SGD ms/upd':>11s} {'Adam ms/upd':>12s}")
for name in DATASETS:
    data = load_csv(os.path.join(ROOT, "experiments/tsf_edge/data", f"{name}.csv"))
    torch.manual_seed(0); np.random.seed(0)
    d, n_warm, C = prep(data, device=dev)
    model = warmup_model(BB, L, H, C, d, n_warm, WARM, device=dev)
    r_sgd = stream_eval(model, d, BB, n_warm, L, H, "full_sgd", lr=1e-3, device=dev)
    r_adm = stream_eval(model, d, BB, n_warm, L, H, "full_adam", lr=1e-4, device=dev)
    dump[name] = dict(channels=C, sgd_ms=r_sgd["adapt_ms"], adam_ms=r_adm["adapt_ms"],
                      n_updates=r_sgd["n_updates"])
    print(f"{name:14s} {C:>4d} {r_sgd['adapt_ms']:>11.2f} {r_adm['adapt_ms']:>12.2f}")

out = os.path.join(ROOT, "results", "tsf_edge", "scale_timing.json")
json.dump(dump, open(out, "w"), indent=2)
print("saved", out)
