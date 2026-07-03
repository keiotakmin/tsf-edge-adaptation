"""(1) Turn C1 from a diagnostic into an IMPLEMENTABLE fair protocol.

The oracle 'sweet spot' = argmin over warmup of the static TEST error uses test data (oracle
early-stopping). A deployable protocol must pick the warmup budget WITHOUT the test set. We split
the pre-drift region into train + a held-out VALIDATION slice (the most-recent pre-drift data),
warm on train only, early-stop on the validation loss, and ask: does validation-early-stopping
approximate the oracle sweet spot -- in warmup step AND in the honest adaptation improvement?
If yes, C1 is an implementable protocol; if the drift makes pre-drift-val diverge from the
post-drift sweet spot, that divergence is itself the finding (fair warmup needs test-aware criteria).
improvement% = 100*(static - adapted)/static  (positive = adaptation helps).
"""
from __future__ import annotations
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from online_eval import build_model, stream_eval, load_csv, prep

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MILESTONES = [50, 100, 200, 500, 1000, 2000, 4000, 8000, 20000, 50000]   # matches warmup_confound
DATASETS = ["ETTm2", "appliances"]
BACKBONES = ["dlinear", "patchtst"]
SEEDS = [0, 1, 2]                                                        # matches warmup_confound
L, H, dev, VAL_FRAC = 96, 24, "cuda", 0.2


def val_mse(model, d, a, b):                       # non-overlapping windows in [a,b], no adapt
    errs, t = [], a + L
    while t + H <= b:
        with torch.no_grad():
            errs.append(F.mse_loss(model(d[t - L:t].unsqueeze(0)), d[t:t + H].unsqueeze(0)).item())
        t += H
    return float(np.mean(errs))


def trajectory(data, bb, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    d, n_warm, C = prep(data, device=dev)
    n_train = int(n_warm * (1 - VAL_FRAC))         # train = earlier pre-drift; val = last 20%
    model = build_model(bb, L, H, C).to(dev); model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rows = []
    for step in range(1, max(MILESTONES) + 1):
        ii = np.random.randint(L, n_train - H, size=32)          # TRAIN region only
        x = torch.stack([d[i - L:i] for i in ii]); y = torch.stack([d[i:i + H] for i in ii])
        opt.zero_grad(); F.mse_loss(model(x), y).backward(); opt.step()
        if step in MILESTONES:
            vl = val_mse(model, d, n_train, n_warm)              # held-out pre-drift validation
            st = stream_eval(model, d, bb, n_warm, L, H, "static", device=dev)["mse"]
            ad = stream_eval(model, d, bb, n_warm, L, H, "full_sgd", device=dev)["mse"]
            rows.append(dict(warmup=step, val=vl, static=st, adapted=ad,
                             improve=100 * (st - ad) / st))
    return rows


fig, axes = plt.subplots(len(DATASETS), len(BACKBONES), figsize=(5 * len(BACKBONES), 4 * len(DATASETS)))
print(f"{'dataset':11s} {'backbone':9s} | {'oracle(min test)':>22s} {'val-early-stop':>20s} {'Δimprove':>9s}")
dump = {}                                   # -> validation_protocol.json for gen_macros.py
for r, name in enumerate(DATASETS):
    data = load_csv(os.path.join(ROOT, "experiments/tsf_edge/data", f"{name}.csv"))
    for c, bb in enumerate(BACKBONES):
        seed_rows = [trajectory(data, bb, seed=s) for s in SEEDS]        # 3-seed, matches warmup_confound
        w = np.array(MILESTONES, float)
        vl = np.mean([[x["val"] for x in rr] for rr in seed_rows], axis=0)
        st = np.mean([[x["static"] for x in rr] for rr in seed_rows], axis=0)
        imp = np.mean([[x["improve"] for x in rr] for rr in seed_rows], axis=0)
        j_oracle = int(np.argmin(st))               # oracle: min static TEST error
        j_val = int(np.argmin(vl))                  # deployable: min held-out val loss
        print(f"{name:11s} {bb:9s} | {MILESTONES[j_oracle]:>6d}st imp={imp[j_oracle]:>+5.1f}%   "
              f"{MILESTONES[j_val]:>6d}st imp={imp[j_val]:>+5.1f}%   {imp[j_val]-imp[j_oracle]:>+7.1f}pt")
        dump[f"{name}|{bb}"] = dict(milestones=MILESTONES, val_mean=vl.tolist(),
                                    static_mean=st.tolist(), improve_mean=imp.tolist(),
                                    oracle_step=MILESTONES[j_oracle], val_step=MILESTONES[j_val],
                                    imp_oracle=float(imp[j_oracle]), imp_val=float(imp[j_val]),
                                    delta=float(imp[j_val] - imp[j_oracle]))
        ax = axes[r, c]
        ax.plot(w, st, "o-", color="0.35", label="static TEST MSE (oracle target)")
        ax.axvline(MILESTONES[j_oracle], color="0.35", ls=":", lw=1.2, label="oracle sweet spot")
        ax.set_xscale("log"); ax.set_ylabel("static TEST MSE"); ax.set_xlabel("warmup steps")
        ax.set_title(f"{name} / {bb}"); ax.grid(alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(w, vl, "s--", color="#1f77b4", label="held-out VAL MSE (deployable)")
        ax2.axvline(MILESTONES[j_val], color="#1f77b4", ls=":", lw=1.2, label="val early-stop")
        ax2.set_ylabel("held-out validation MSE", color="#1f77b4"); ax2.tick_params(axis="y", labelcolor="#1f77b4")
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=6.5, loc="upper right")

fig.suptitle("Deployable fair protocol: does held-out (pre-drift) validation early-stopping pick the "
             "same warmup as the oracle (min static TEST error)?", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = os.path.join(ROOT, "results", "tsf_edge")
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(out, f"validation_protocol.{ext}"), dpi=150, bbox_inches="tight")
json.dump(dump, open(os.path.join(out, "validation_protocol.json"), "w"), indent=2)
print("\nsaved", os.path.join(out, "validation_protocol.png"))
print("Read: if val-early-stop improvement ≈ oracle improvement (small Δ), pre-drift validation is a "
      "sound deployable stand-in for the fair sweet spot; a large Δ means drift breaks it (a finding).")
