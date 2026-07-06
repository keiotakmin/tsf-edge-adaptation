"""Guard experiment: is the collapse of high online rates (Fig. 5A) a startup TRANSIENT that
a longer stream would amortize ("with more training time the higher rate would win"), or a
STEADY-STATE tracking failure?

Offline intuition does not transfer here: adaptation is prequential (one update per revealed
window -- the stream fixes the update budget; every scored prediction is deployment cost) and
every rate starts from the IDENTICAL fairly-warmed checkpoint. This script settles the
residual question empirically: per-window online MSE by stream QUARTER for the collapsed arm
(Adam@1e-2) vs static / plateau-Adam@1e-4 / SGD@1e-2, on the longest stream (ETTm2, ~1.45k
windows) and on ETTh2 (where a partial transient does exist).
-> results/tsf_edge/lr_transient.json (consumed by gen_macros.py, macros \\Lt*).
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from online_eval import VAL_FRAC, _clone, load_csv, prep, set_trainable, warm_and_select

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARMS = [(None, 0.0, "static"), ("adam", 1e-4, "adam_lo"),
        ("adam", 1e-2, "adam_hi"), ("sgd", 1e-2, "sgd_hi")]
BB, L, H, SEED, dev = "patchtst", 96, 24, 0, "cuda"


def per_window_errors(model, d, n_warm, okind, lr):
    T = d.shape[0]
    m = _clone(model)
    set_trainable(m, BB, "all" if okind else None)
    ps = [p for p in m.parameters() if p.requires_grad]
    opt = (torch.optim.SGD(ps, lr=lr) if okind == "sgd" else
           torch.optim.Adam(ps, lr=lr) if okind == "adam" else None)
    errs, t = [], n_warm
    while t + H <= T:
        x, y = d[t - L:t].unsqueeze(0), d[t:t + H].unsqueeze(0)
        with torch.no_grad():
            errs.append(F.mse_loss(m(x), y).item())
        if opt:
            opt.zero_grad(); F.mse_loss(m(x), y).backward(); opt.step()
        t += H
    return np.array(errs)


dump = {}
for name in ("ETTm2", "ETTh2"):
    data = load_csv(os.path.join(ROOT, "experiments/tsf_edge/data", f"{name}.csv"))
    d, n_warm, C = prep(data, device=dev)
    n_train = int(n_warm * (1 - VAL_FRAC))
    torch.manual_seed(SEED); np.random.seed(SEED)
    model, wstep, _ = warm_and_select(BB, L, H, C, d, n_train, n_warm, SEED)
    entry = {"warm": wstep, "arms": {}}
    print(f"\n=== {name}/{BB} warm={wstep} (per-window MSE by stream quarter) ===")
    for okind, lr, tag in ARMS:
        e = per_window_errors(model, d, n_warm, okind, lr)
        q = len(e) // 4
        quarters = [float(e[i * q:(i + 1) * q].mean()) for i in range(4)]
        entry["arms"][tag] = dict(lr=lr, n_windows=len(e), quarters=quarters,
                                  last_hundred=float(e[-100:].mean()))
        print(f"{tag:8s} lr={lr:g}: Q1..Q4 = " + " ".join(f"{v:.4f}" for v in quarters)
              + f"  last100={e[-100:].mean():.4f}  (n={len(e)})")
    dump[f"{name}|{BB}"] = entry

out = os.path.join(ROOT, "results", "tsf_edge", "lr_transient.json")
json.dump(dump, open(out, "w"), indent=2)
print("\nsaved", out)
print("Read: if adam_hi's Q4 >= Q1 (ETTm2) the collapse is steady-state, not transient; "
      "where it does decay (ETTh2) it still settles far above static's Q4.")
