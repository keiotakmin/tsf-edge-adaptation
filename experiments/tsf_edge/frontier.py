"""C2 consolidated frontier: every adaptation strategy on the quality vs resource plane,
Pareto frontier marked. Quality = adaptation benefit % over the fair-warmup static baseline,
where FAIR = the C1 deployable protocol (held-out pre-drift validation early-stopping per
dataset x backbone, `warm_and_select`) — NOT a fixed warmup: the former fixed 2000 sat off
the sweet spot for e.g. ETTm2/DLinear (@20k) and skewed those points per C1's own logic.
Resource = adaptation footprint (trainable params ~ memory) and per-update compute (ms).
Message: PEFT (calib/head·SGD) is Pareto-efficient; full·Adam is regime-dependent (best on
noisy Appliances, Pareto-dominated on ETTm2). Point data is cached to frontier_data.json so
the figure can be restyled without recomputing (pass --recompute to refresh).
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
    ("patchtst", "full_sgd",  "PatchTST full·SGD",  "o", "#1f77b4"),
    ("patchtst", "full_adam", "PatchTST full·Adam", "o", "#d62728"),
    ("patchtst", "head_sgd",  "PatchTST head·SGD",  "s", "#1f77b4"),
    ("patchtst", "calib_sgd", "PatchTST calib·SGD", "^", "#1f77b4"),
    ("dlinear",  "full_sgd",  "DLinear full·SGD",   "P", "#2ca02c"),
    ("dlinear",  "head_sgd",  "DLinear head·SGD",   "X", "#2ca02c"),
]
L, H, SEED, dev = 96, 24, 0, "cuda"
CACHE = os.path.join(ROOT, "results", "tsf_edge", "frontier_data.json")


def compute():
    from online_eval import VAL_FRAC, load_csv, prep, stream_eval, warm_and_select
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
            r = stream_eval(model, d, bb, n_warm, L, H, strat, device=dev)
            rows.append(dict(label=lab, params=r["n_adapt_params"], ms=r["adapt_ms"],
                             benefit=100 * (base[bb] - r["mse"]) / base[bb], warmup=wstep))
        out[name] = rows
    json.dump(out, open(CACHE, "w"), indent=2)
    return out


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
        for c, (xkey, xlab) in enumerate([("params", "trainable params  (memory footprint, ↓ better)"),
                                          ("ms", "per-update compute (ms, ↓ better)")]):
            ax = axes[r, c]
            for row in rows:
                mk, col = style[row["label"]]
                ax.scatter(row[xkey], row["benefit"], marker=mk, s=180, color=col,
                           edgecolor="k", lw=0.8, zorder=3)
            if xkey == "params":
                pf = pareto([(row["params"], row["benefit"]) for row in rows])
                ax.plot([p[0] for p in pf], [p[1] for p in pf], "--", color="0.5", lw=1.3, zorder=1)
                ax.set_xscale("log")
            ax.axhline(0, color="0.7", ls=":", lw=1)
            ax.set_xlabel(xlab); ax.set_ylabel("adaptation benefit %  (↑ better)")
            ax.set_title(f"{name}: quality vs {'memory' if xkey=='params' else 'compute'}")
            ax.grid(alpha=0.3)

    handles = [Line2D([0], [0], marker=mk, color="w", markerfacecolor=col, markeredgecolor="k",
                      markersize=13, label=lab) for _, _, lab, mk, col in COMBOS]
    handles.append(Line2D([0], [0], ls="--", color="0.5", label="Pareto frontier (memory)"))
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=10, frameon=True,
               bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("Resource frontier of on-device adaptation (fair warmup): PEFT (calib/head·SGD) "
                 "is Pareto-efficient; full·Adam is regime-dependent", fontsize=12, y=1.045)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(ROOT, "results", "tsf_edge")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"frontier.{ext}"), dpi=150, bbox_inches="tight")
    print("saved", os.path.join(out, "frontier.png"))


if __name__ == "__main__":
    main()
