"""C2 consolidated frontier: every adaptation strategy on the quality vs resource plane,
Pareto frontier marked. Quality = adaptation benefit % over the fair-warmup static baseline,
where FAIR = the C1 deployable protocol (held-out pre-drift validation early-stopping per
dataset x backbone, `warm_and_select`) — NOT a fixed warmup: the former fixed 2000 sat off
the sweet spot for e.g. ETTm2/DLinear (@20k) and skewed those points per C1's own logic.
FAIR now also covers the ONLINE LR (referee M1 / lr_fairness.py): every strategy's online LR
is rehearsal-selected on the pre-drift validation slice (`select_online_lr`), uniformly —
NOT the former fixed lr=1e-3, which sat inside SGD's safety plateau but outside Adam's and
manufactured the "full·Adam dominated" reading on ETTm2. Each cached point carries both the
fair benefit (`benefit`, at `lr`) and the old fixed-1e-3 reading (`benefit_fixed`) so the
artifact remains reportable. Resource = adaptation footprint (trainable params ~ memory) and
per-update compute (ms). Point data is cached to frontier_data.json so the figure can be
restyled without recomputing (pass --recompute to refresh).
"""
from __future__ import annotations
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASETS = ["appliances", "ETTm2"]
COMBOS = [   # (backbone, strategy, label, marker, color)
    # R3: momentum-free SGD is no longer reported (torch.optim.SGD's momentum is a free
    # argument and nobody deploys the 0-momentum form), so the SGD family here IS SGD+momentum
    # and takes the family's blue. Its points remain in frontier_seeds.jsonl for reference.
    # colour = OPTIMIZER only (paper_figs.PAL); backbone and parameter subset are carried
    # by marker shape (PatchTST o/s/^ for full/head/calib, DLinear P/X for full/head). The
    # previous per-backbone hues put SGD+momentum-orange next to DLinear-green, a pair that is
    # indistinguishable under protanopia (dE 1.4, check_palette.py).
    ("patchtst", "full_adam",  "PatchTST full·Adam",  "o", "#d62728"),
    ("patchtst", "full_sgdm",  "PatchTST full·SGD+m", "o", "#1f77b4"),
    ("patchtst", "head_adam",  "PatchTST head·Adam",  "s", "#d62728"),
    ("patchtst", "calib_adam", "PatchTST calib·Adam", "^", "#d62728"),
    ("dlinear",  "full_sgdm",  "DLinear full·SGD+m",  "P", "#1f77b4"),
    ("dlinear",  "head_adam",  "DLinear head·Adam",   "X", "#d62728"),
    # completes the strategy x optimizer grid: every parameter subset is now measured with
    # all three optimizers, so no cell of the recipe is inferred rather than observed.
    ("patchtst", "head_sgdm",  "PatchTST head·SGD+m",  "s", "#1f77b4"),
    ("patchtst", "calib_sgdm", "PatchTST calib·SGD+m", "^", "#1f77b4"),
    ("dlinear",  "full_adam",  "DLinear full·Adam",    "P", "#d62728"),
    ("dlinear",  "head_sgdm",  "DLinear head·SGD+m",   "X", "#1f77b4"),
]
L, H, SEED, dev = 96, 24, 0, "cuda"
CACHE = os.path.join(ROOT, "results", "tsf_edge", "frontier_data.json")


def compute():
    from online_eval import VAL_FRAC, load_csv, prep, select_online_lr, stream_eval, warm_and_select
    out = {}
    for name in DATASETS:
        data = load_csv(os.path.join(ROOT, "experiments/tsf_edge/data", f"{name}.csv"))
        d, n_warm, C = prep(data, device=dev)
        n_train = int(n_warm * (1 - VAL_FRAC))
        warmed, base = {}, {}
        for bb in {c[0] for c in COMBOS}:
            model, wstep, _ = warm_and_select(bb, L, H, C, d, n_train, n_warm, SEED)
            base[bb] = stream_eval(model, d, bb, n_warm, L, H, "static", device=dev)["mse"]
            warmed[bb] = (model, wstep)
            print(f"{name}/{bb}: fair warmup={wstep} static={base[bb]:.4f}", flush=True)
        rows = []
        for bb, strat, lab, mk, col in COMBOS:
            model, wstep = warmed[bb]
            sel, _ = select_online_lr(model, d, bb, n_train, n_warm, L, H, strat, device=dev)
            r = stream_eval(model, d, bb, n_warm, L, H, strat, lr=sel, device=dev)
            r0 = stream_eval(model, d, bb, n_warm, L, H, strat, device=dev)   # fixed-1e-3 ref
            rows.append(dict(label=lab, params=r["n_adapt_params"], ms=r["adapt_ms"],
                             benefit=100 * (base[bb] - r["mse"]) / base[bb], lr=sel,
                             benefit_fixed=100 * (base[bb] - r0["mse"]) / base[bb],
                             warmup=wstep))
            print(f"  {lab:20s} lr={sel:g}  benefit={rows[-1]['benefit']:+6.1f}%  "
                  f"(fixed 1e-3: {rows[-1]['benefit_fixed']:+6.1f}%)", flush=True)
        out[name] = rows
    json.dump(out, open(CACHE, "w"), indent=2)
    return out


def adapt_mem_bytes(row):
    """Adaptation memory footprint: gradient buffer (4 B/param, fp32) + optimizer state
    (Adam: 2 extra copies; SGD: none). Trainable params ship with the model either way, so
    they are not counted; this axis separates full·Adam (12 B/param) from full·SGD (4 B/param)
    at equal trainable-parameter count — the old 'trainable params' axis could not, which
    mattered once fair-LR selection made full·Adam the top-quality point."""
    # R3: prefer the MEASURED optimizer state when the run recorded it. The label-based rule
    # below silently mis-costs anything that is neither plain SGD nor Adam -- SGD+momentum
    # carries one state copy (8 B/param total) but matches neither "Adam" nor a 0-state SGD,
    # and was billed at 4 B/param until this was checked against stream_eval's measurement.
    if row.get("opt_state_bytes") is not None:
        return 4 * row["params"] + row["opt_state_bytes"]
    return 4 * row["params"] * (3 if "Adam" in row["label"] else 1)


def pareto(pts):                                    # (x LOW better, y HIGH better)
    keep = [(x, y) for i, (x, y) in enumerate(pts)
            if not any(qx <= x and qy >= y and (qx < x or qy > y)
                       for j, (qx, qy) in enumerate(pts) if j != i)]
    return sorted(keep)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()
    data = compute() if (args.recompute or not os.path.exists(CACHE)) else json.load(open(CACHE))
    style = {lab: (mk, col) for _, _, lab, mk, col in COMBOS}

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                         "xtick.labelsize": 10, "ytick.labelsize": 10})
    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(12, 5 * len(DATASETS)))
    for r, name in enumerate(DATASETS):
        rows = data[name]
        for c, (xget, xlab, xname) in enumerate(
                [(adapt_mem_bytes, "adaptation memory: gradients + optimizer state (bytes, ↓ better)",
                  "memory"),
                 (lambda row: row["ms"], "per-update compute (ms, ↓ better)", "compute")]):
            ax = axes[r, c]
            for row in rows:
                mk, col = style[row["label"]]
                ax.scatter(xget(row), row["benefit"], marker=mk, s=180, color=col,
                           edgecolor="k", lw=0.8, zorder=3)
            if xname == "memory":
                pf = pareto([(xget(row), row["benefit"]) for row in rows])
                ax.plot([p[0] for p in pf], [p[1] for p in pf], "--", color="0.5", lw=1.3, zorder=1)
                ax.set_xscale("log")
            ax.axhline(0, color="0.7", ls=":", lw=1)
            ax.set_xlabel(xlab); ax.set_ylabel("adaptation benefit %  (↑ better)")
            ax.set_title(f"{name}: quality vs {xname}")
            ax.grid(alpha=0.3)

    handles = [Line2D([0], [0], marker=mk, color="w", markerfacecolor=col, markeredgecolor="k",
                      markersize=13, label=lab) for _, _, lab, mk, col in COMBOS]
    handles.append(Line2D([0], [0], ls="--", color="0.5",
                          label="nondominated (memory axis)"))
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=10, frameon=True,
               bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("Resource frontier of on-device adaptation (fair warmup + rehearsal-selected "
                 "online LR)", fontsize=12, y=1.045)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(ROOT, "results", "tsf_edge")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"frontier.{ext}"), dpi=150, bbox_inches="tight")
    print("saved", os.path.join(out, "frontier.png"))


if __name__ == "__main__":
    main()
