"""(3)+(4) combined grid: one sweep that yields BOTH the robustness/breadth evidence for C2
and the regime-predictor validation data for C3 (Fable Point 2). Per cell (dataset x backbone
x L x H x seed) we:
  - pick a FAIR warmup by held-out pre-drift validation early-stopping, training the base model
    ONCE and reusing the min-val checkpoint for BOTH the SGD and Adam adaptation cells
    (warmup is optimizer-independent -> no blow-up);
  - measure honest benefit for full-SGD and full-Adam (+ optimizer-state bytes);
  - measure THREE deploy-time probes, all optimizer-INDEPENDENT (Fable Point 2, no circularity):
      P1 noise    = mean-over-channels variance of the stream's first difference (raw noise),
      P2 grad-cos = mean cosine of consecutive per-window gradients on the FROZEN base model
                    (gradient consistency; a high value ~ SGD-friendly, low ~ Adam-friendly),
      P3 drift    = static test-MSE / static val-MSE of the frozen base model (drift strength).
One JSONL row per cell -> the C2 frontier and the C3 regime table both build from this file.
Cells run in claim-driven priority order (Adam-favourable high-signal first). Datasets pluggable
(add BDG2 later with zero rework)."""
from __future__ import annotations
import json, os
import numpy as np
import torch
import torch.nn.functional as F

from online_eval import VAL_FRAC, load_csv, prep, stream_eval, val_mse, warm_and_select

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "tsf_edge", "grid.jsonl")
# priority order: Adam-favourable / high-drift first so the C3 judgment call can be made early
DATASETS = ["appliances", "bdg2", "ETTm2", "ETTh2", "ETTm1", "ETTh1"]  # +bdg2 (2nd real building)
BACKBONES = ["patchtst", "dlinear"]
HS = [24, 48, 96]
LS = [96, 192]                 # robustness batch: + L=192
SEEDS = [0, 1, 2, 3, 4]        # robustness batch: 5 seeds (Crowded-Valley discipline)
dev = "cuda"                   # warmup selection: shared warm_and_select (WARM_GRID→20k) in online_eval


def probes(model, backbone, d, n_train, n_warm, L, H):
    T = d.shape[0]
    # P1: raw noise = mean-over-channels variance of the first difference over the test region
    diff = d[n_warm + 1:] - d[n_warm:-1]
    p1 = float(diff.var(dim=0).mean())
    # P2: cosine of consecutive per-window gradients on the FROZEN model (optimizer-independent)
    params = [p for p in model.parameters() if p.requires_grad]
    grads, t, npair = [], n_warm, 0
    while t + H <= T and npair < 150:
        model.zero_grad()
        F.mse_loss(model(d[t - L:t].unsqueeze(0)), d[t:t + H].unsqueeze(0)).backward()
        grads.append(torch.cat([p.grad.detach().reshape(-1) for p in params]).clone())
        t += H; npair += 1
    model.zero_grad()
    cos = [float(F.cosine_similarity(grads[i], grads[i + 1], dim=0)) for i in range(len(grads) - 1)]
    p2 = float(np.mean(cos)) if cos else float("nan")
    # P3: drift strength = static test MSE / static val MSE (frozen model)
    val = val_mse(model, d, n_train, n_warm, L, H)
    test = stream_eval(model, d, backbone, n_warm, L, H, "static", device=dev)["mse"]
    p3 = test / max(val, 1e-8)
    return p1, p2, p3


def main():
    done_rows = []
    if os.path.exists(OUT):                 # resume: keep completed cells, drop a partial last line
        for line in open(OUT):
            try:
                done_rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        with open(OUT, "w") as f:
            for r in done_rows:
                f.write(json.dumps(r) + "\n")
    done_k = {(r["dataset"], r["backbone"], r["L"], r["H"], r["seed"]) for r in done_rows}
    n_cells = len(DATASETS) * len(BACKBONES) * len(HS) * len(LS) * len(SEEDS)
    done = len(done_k)
    if done:
        print(f"resume: {done}/{n_cells} cells already in {OUT}", flush=True)
    for name in DATASETS:
        data = load_csv(os.path.join(ROOT, "experiments/tsf_edge/data", f"{name}.csv"))
        for bb in BACKBONES:
            for L in LS:
                for H in HS:
                    for seed in SEEDS:
                        if (name, bb, L, H, seed) in done_k:
                            continue
                        d, n_warm, C = prep(data, device=dev)
                        n_train = int(n_warm * (1 - VAL_FRAC))
                        model, wstep, wval = warm_and_select(bb, L, H, C, d, n_train, n_warm, seed)
                        st = stream_eval(model, d, bb, n_warm, L, H, "static", device=dev)["mse"]
                        r_sgd = stream_eval(model, d, bb, n_warm, L, H, "full_sgd", device=dev)
                        r_adm = stream_eval(model, d, bb, n_warm, L, H, "full_adam", device=dev)
                        p1, p2, p3 = probes(model, bb, d, n_train, n_warm, L, H)
                        row = dict(dataset=name, backbone=bb, L=L, H=H, seed=seed,
                                   warmup=wstep, static=st,
                                   benefit_sgd=100 * (st - r_sgd["mse"]) / st,
                                   benefit_adam=100 * (st - r_adm["mse"]) / st,
                                   optstate_sgd=r_sgd["opt_state_bytes"],
                                   optstate_adam=r_adm["opt_state_bytes"],
                                   n_params=r_sgd["n_adapt_params"],
                                   p1_noise=p1, p2_gradcos=p2, p3_drift=p3)
                        with open(OUT, "a") as f:
                            f.write(json.dumps(row) + "\n")
                        done += 1
                        win = "SGD" if row["benefit_sgd"] >= row["benefit_adam"] else "Adam"
                        print(f"[{done:2d}/{n_cells}] {name:11s} {bb:9s} L{L} H{H:2d} | warm={wstep:5d} "
                              f"SGD={row['benefit_sgd']:+5.1f}% Adam={row['benefit_adam']:+6.1f}% win={win:4s} "
                              f"| P1={p1:.3f} P2={p2:+.2f} P3={p3:.2f}", flush=True)
    print(f"\nwrote {OUT} ({done} cells)")


if __name__ == "__main__":
    main()
