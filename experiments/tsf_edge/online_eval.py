"""Online-adaptation evaluation harness for time-series forecasting under drift, with
RESOURCE PROFILING and TWO backbones (DLinear, compact PatchTST). See PLAN.md.

Streaming protocol (FSNet/OneNet/online-TSF): warm up on an initial segment, then stream --
predict horizon from lookback, score vs the revealed truth, then ADAPT on that pair. We vary
the adaptation STRATEGY (which params + which optimizer + update SCHEDULE) and log forecast
quality plus adaptation cost (optimizer-state bytes, peak memory, wall-clock per update).
"""
from __future__ import annotations
import argparse, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

P_EDGE_W = 5.0                                    # assumed edge device power for energy proxy


def load_csv(path):
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in ("date", "rv1", "rv2") if c in df.columns], errors="ignore")
    return df.select_dtypes("number").values.astype("float32")          # [T,C]


class DLinear(nn.Module):
    def __init__(self, L, H, C, kernel=25):
        super().__init__()
        self.k = kernel
        self.lin_season = nn.Linear(L, H)
        self.lin_trend = nn.Linear(L, H)

    def forward(self, x):                                              # [B,L,C]->[B,H,C]
        pad = self.k // 2
        xp = F.pad(x.transpose(1, 2), (pad, pad), mode="replicate")
        trend = F.avg_pool1d(xp, self.k, stride=1).transpose(1, 2)
        s = self.lin_season((x - trend).transpose(1, 2)).transpose(1, 2)
        t = self.lin_trend(trend.transpose(1, 2)).transpose(1, 2)
        return s + t


class PatchTST(nn.Module):
    """Compact channel-independent PatchTST (Nie et al. 2023). in_affine = a per-channel
    calibration used by the parameter-efficient 'calib' strategy."""
    def __init__(self, L, H, C, P=16, S=8, d=64, nhead=4, nlayers=2):
        super().__init__()
        self.P, self.S = P, S
        self.np = (L - P) // S + 1
        self.in_affine_w = nn.Parameter(torch.ones(C))
        self.in_affine_b = nn.Parameter(torch.zeros(C))
        self.embed = nn.Linear(P, d)
        self.pos = nn.Parameter(torch.zeros(1, self.np, d))
        enc = nn.TransformerEncoderLayer(d, nhead, 2 * d, batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(enc, nlayers)
        self.head = nn.Linear(self.np * d, H)

    def forward(self, x):                                             # [B,L,C]->[B,H,C]
        B, L, C = x.shape
        x = x * self.in_affine_w + self.in_affine_b
        xci = x.permute(0, 2, 1).reshape(B * C, L)
        patches = xci.unfold(1, self.P, self.S)                       # [BC,np,P]
        z = self.embed(patches) + self.pos
        z = self.encoder(z)
        out = self.head(z.reshape(B * C, -1))                        # [BC,H]
        return out.reshape(B, C, -1).permute(0, 2, 1)


def build_model(backbone, L, H, C):
    return DLinear(L, H, C) if backbone == "dlinear" else PatchTST(L, H, C)


def set_trainable(model, backbone, which):
    for p in model.parameters():
        p.requires_grad_(which == "all")
    if which == "all":
        return
    if backbone == "dlinear":
        for p in model.lin_trend.parameters():                       # head = trend map
            p.requires_grad_(True)
    else:
        for p in model.head.parameters():
            p.requires_grad_(True)
        if which == "calib":                                         # + per-channel calibration
            model.in_affine_w.requires_grad_(True); model.in_affine_b.requires_grad_(True)


# strategy -> (which, online optimizer, optimizer-state multiplier)
STRATEGIES = {
    "static":    (None,   None,   0),
    "full_sgd":  ("all",  "sgd",  0),
    "full_adam": ("all",  "adam", 2),
    "head_sgd":  ("head", "sgd",  0),
    "calib_sgd": ("calib", "sgd", 0),      # PatchTST only (PEFT-style)
}


@torch.no_grad()
def _clone(model):
    import copy
    return copy.deepcopy(model)


def stream_eval(model, d, backbone, n_warm, L, H, strategy="full_sgd", lr=1e-3,
                device="cuda", schedule="every", k=1, tau=1.5, ema_beta=0.9, stride=None):
    """Online phase on an ALREADY-WARMED model: predict / score / adapt, rolling forward.
    Adapts a CLONE so the caller's warmed model is left untouched (for the warmup sweep).

    stride=H (default, NON-OVERLAPPING) is leakage-free per DSOF (ICLR'25): every eval target
    y[t:t+H] is scored before it is ever used in a gradient, and the model only ever adapts on
    fully-arrived (past) ground truth. stride<H (overlapping) reproduces the FSNet/OneNet
    information leak (the overlap y[t+stride:t+H] is trained on at step t, then re-scored at t+stride)."""
    T = d.shape[0]
    stride = H if stride is None else stride
    which, okind, state_mult = STRATEGIES[strategy]
    model = _clone(model)
    set_trainable(model, backbone, which)
    adapt_params = [p for p in model.parameters() if p.requires_grad] if which else []
    opt_online = (torch.optim.SGD(adapt_params, lr=lr) if okind == "sgd" else
                  torch.optim.Adam(adapt_params, lr=lr) if okind == "adam" else None)

    errs, adapt_t, nupd, peak_mem = [], 0.0, 0, 0
    ema, widx, nwin = None, 0, 0
    t = n_warm
    while t + H <= T:
        x, y = d[t - L:t].unsqueeze(0), d[t:t + H].unsqueeze(0)
        with torch.no_grad():
            err = F.mse_loss(model(x), y).item()
        errs.append(err); nwin += 1
        do_adapt = opt_online is not None
        if do_adapt and schedule == "every":
            do_adapt = (widx % k == 0)
        elif do_adapt and schedule == "drift":
            if ema is None:
                ema = err
            else:
                do_adapt = err > tau * ema
            ema = ema_beta * ema + (1 - ema_beta) * err
        widx += 1
        if do_adapt:
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            t0 = time.perf_counter()
            opt_online.zero_grad(); F.mse_loss(model(x), y).backward(); opt_online.step()
            if device == "cuda":
                torch.cuda.synchronize(); peak_mem = max(peak_mem, torch.cuda.max_memory_allocated())
            adapt_t += time.perf_counter() - t0; nupd += 1
        t += stride

    n_adapt = sum(p.numel() for p in adapt_params)
    ms = 1000 * adapt_t / max(nupd, 1)
    return dict(backbone=backbone, strategy=strategy, mse=float(np.mean(errs)),
                adapt_ms=ms, n_updates=nupd, update_frac=nupd / max(nwin, 1),
                n_adapt_params=n_adapt, opt_state_bytes=state_mult * n_adapt * 4,
                peak_adapt_mem_kb=peak_mem / 1024, energy_mj=ms * P_EDGE_W)


def prep(data, warmup_frac=0.5, device="cuda"):
    T, C = data.shape
    n_warm = int(T * warmup_frac)
    mean, std = data[:n_warm].mean(0), data[:n_warm].std(0) + 1e-8
    return torch.tensor((data - mean) / std, device=device), n_warm, C


def warmup_model(backbone, L, H, C, d, n_warm, warmup_steps, lr=1e-3, device="cuda", bs=32):
    model = build_model(backbone, L, H, C).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr); model.train()
    for _ in range(warmup_steps):
        ii = np.random.randint(L, n_warm - H, size=bs)
        x = torch.stack([d[i - L:i] for i in ii]); y = torch.stack([d[i:i + H] for i in ii])
        opt.zero_grad(); F.mse_loss(model(x), y).backward(); opt.step()
    return model


VAL_FRAC = 0.2   # held-out most-recent pre-drift slice used for warmup early-stopping (C1 protocol)
WARM_GRID = [200, 500, 1000, 2000, 4000, 8000, 20000]   # 20k covers the ETTm2/DLinear sweet spot;
                                                        # the old 8k cap censored 14/108 grid cells
LR_GRID = [3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]   # online-LR rehearsal grid (M1
                                                             # fair-LR protocol; lr_fairness.py)


def val_mse(model, d, a, b, L, H):
    """Static MSE over non-overlapping windows in [a, b] (the held-out pre-drift validation slice)."""
    errs, t = [], a + L
    while t + H <= b:
        with torch.no_grad():
            errs.append(F.mse_loss(model(d[t - L:t].unsqueeze(0)), d[t:t + H].unsqueeze(0)).item())
        t += H
    return float(np.mean(errs))


def select_online_lr(model, d, backbone, n_train, n_warm, L, H, strategy, lr_grid=None,
                     device="cuda"):
    """M1 fair-LR protocol: REHEARSE online adaptation (same score-then-adapt streaming) on the
    held-out pre-drift validation slice at each candidate LR; return (best_lr, {lr: val MSE}).
    Deployable -- no test data. Companion to `warm_and_select` (C1); a strategy's online LR must
    be selected per deployment or the optimizer comparison is confounded (lr_fairness.py)."""
    lr_grid = LR_GRID if lr_grid is None else lr_grid
    d_val = d[:n_warm]                    # stream ends where the test region begins
    scores = {lr: stream_eval(model, d_val, backbone, n_train, L, H, strategy, lr=lr,
                              device=device)["mse"] for lr in lr_grid}
    best = min(lr_grid, key=lambda x: scores[x] if scores[x] == scores[x] else float("inf"))
    return best, scores


def warm_and_select(backbone, L, H, C, d, n_train, n_warm, seed, warm_grid=None, lr=1e-3, bs=32):
    """FAIR warmup = the C1 deployable protocol: train the base model on the TRAIN region only,
    checkpoint at each grid milestone, return the min-held-out-validation checkpoint
    (+ its warmup step and val MSE). Shared by combined_grid / frontier / staleness so every
    downstream figure reads the baseline at the same fair warmup."""
    import copy
    warm_grid = WARM_GRID if warm_grid is None else warm_grid
    torch.manual_seed(seed); np.random.seed(seed)
    model = build_model(backbone, L, H, C).to(d.device); model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, best_state, best_step = float("inf"), None, warm_grid[0]
    for step in range(1, max(warm_grid) + 1):
        ii = np.random.randint(L, n_train - H, size=bs)
        x = torch.stack([d[i - L:i] for i in ii]); y = torch.stack([d[i:i + H] for i in ii])
        opt.zero_grad(); F.mse_loss(model(x), y).backward(); opt.step()
        if step in warm_grid:
            v = val_mse(model, d, n_train, n_warm, L, H)
            if v < best_val:
                best_val, best_state, best_step = v, copy.deepcopy(model.state_dict()), step
    model.load_state_dict(best_state)
    return model, best_step, best_val


def run(data, backbone="dlinear", L=96, H=24, warmup_frac=0.5, warmup_steps=2000,
        strategy="full_sgd", lr=1e-3, device="cuda", seed=0,
        schedule="every", k=1, tau=1.5, ema_beta=0.9, stride=None):
    torch.manual_seed(seed); np.random.seed(seed)
    d, n_warm, C = prep(data, warmup_frac, device)
    model = warmup_model(backbone, L, H, C, d, n_warm, warmup_steps, lr, device)
    return stream_eval(model, d, backbone, n_warm, L, H, strategy, lr, device,
                       schedule, k, tau, ema_beta, stride)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="ETTh2,ETTm2,appliances")
    ap.add_argument("--backbones", default="dlinear,patchtst")
    ap.add_argument("--L", type=int, default=96); ap.add_argument("--H", type=int, default=24)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for name in args.datasets.split(","):
        data = load_csv(f"experiments/tsf_edge/data/{name}.csv")
        print(f"\n=== {name}  shape={data.shape} ===")
        print(f"{'backbone':9s} {'strategy':10s} {'MSE':>8s} {'vs static':>9s} "
              f"{'ms/upd':>7s} {'optstate B':>11s} {'peak KB':>8s} {'#params':>8s}")
        for bb in args.backbones.split(","):
            base = run(data, backbone=bb, strategy="static", device=dev)["mse"]
            strats = ["static", "full_sgd", "full_adam", "head_sgd"] + (["calib_sgd"] if bb == "patchtst" else [])
            for strat in strats:
                r = run(data, backbone=bb, strategy=strat, device=dev)
                rel = "" if strat == "static" else f"{100*(r['mse']-base)/base:+.1f}%"
                print(f"{bb:9s} {strat:10s} {r['mse']:>8.4f} {rel:>9s} {r['adapt_ms']:>7.3f} "
                      f"{r['opt_state_bytes']:>11d} {r['peak_adapt_mem_kb']:>8.1f} {r['n_adapt_params']:>8d}")


if __name__ == "__main__":
    main()
