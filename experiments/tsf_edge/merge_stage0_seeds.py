"""Fold the per-host seed-3/4 files into one results/tsf_edge/stage0_seeds34.jsonl.

Deliberately NOT merged into stage0_optimizers.jsonl / stage0c_optimizers.jsonl: those hold the
216-cell screen on which all 20 arms are compared, and appending 144 cells that carry only the
seven headline arms would make every pooled column ragged -- 360 cells for nine arms, 216 for
the rest. The 5-seed reading is a robustness statement about the headline arms, computed by
stage0_pool.seed_robustness(), not a replacement for the screen.

Checks before it writes:
  * disjoint   -- no cell in two host files, and none already in the 216-cell artifacts
  * seeds      -- every row is seed 3 or 4
  * complete   -- every row carries all seven arms with their full grids
  * static     -- each row's static agrees with lr_fairness.jsonl (which HAS seeds 3 and 4)
                  within 0.1%; this is the same cross-host check the runner logs as a WARNING,
                  re-applied here so a host mismatch cannot reach the tables silently

Run from the project root:
    .venv/bin/python experiments/tsf_edge/merge_stage0_seeds.py            # dry run
    .venv/bin/python experiments/tsf_edge/merge_stage0_seeds.py --write
"""
from __future__ import annotations
import argparse, glob, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results", "tsf_edge")
DST = os.path.join(RES, "stage0_seeds34.jsonl")
ARMS = ("lion", "adafactor", "signsgd", "obsign", "obsign_t3e3", "obsign_t1e3", "relsign")
SHARED = [3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
KEY = lambda r: (r["dataset"], r["backbone"], r["L"], r["H"], r["seed"])


def load(path):
    return {KEY(r): r for r in (json.loads(l) for l in open(path))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(RES, "stage0_seeds34_*.jsonl")))
    if not files:
        sys.exit("no stage0_seeds34_*.jsonl found")
    ref = load(os.path.join(RES, "lr_fairness.jsonl"))
    screen = set()
    for f in ("stage0_optimizers.jsonl", "stage0b_optimizers.jsonl", "stage0c_optimizers.jsonl"):
        screen |= set(load(os.path.join(RES, f)))

    merged, owner, problems = {}, {}, []
    for f in files:
        host = os.path.basename(f)[len("stage0_seeds34_"):-len(".jsonl")]
        rows = load(f)
        print(f"  {os.path.basename(f)}: {len(rows)} rows")
        for k, r in rows.items():
            if k in owner:
                problems.append(f"{k} appears in both {owner[k]} and {host}")
                continue
            if k in screen:
                problems.append(f"{k} ({host}) is already in the 216-cell screen")
            owner[k] = host
            if r["seed"] not in (3, 4):
                problems.append(f"{k} ({host}) is not a seed-3/4 cell")
            for a in ARMS:
                miss = {f"{lr:g}" for lr in SHARED} - set(r.get(a, {}))
                if miss:
                    problems.append(f"{k} ({host}): {a} missing {sorted(miss)}")
            rr = ref.get(k)
            if rr is None:
                problems.append(f"{k} ({host}): no lr_fairness row to cross-check static")
            elif abs(r["static"] - rr["static"]) > 1e-3 * max(rr["static"], 1e-8):
                problems.append(f"{k} ({host}): static {r['static']:.5f} != lr_fairness "
                                f"{rr['static']:.5f} -- host mismatch")
            merged[k] = dict(r, seed_ext_host=host)

    if problems:
        print(f"\n{len(problems)} PROBLEM(S) -- refusing to write:")
        for p in problems[:40]:
            print("  -", p)
        sys.exit(1)
    seeds = sorted({k[4] for k in merged})
    print(f"\nall checks passed; {len(merged)} cells, seeds {seeds}, "
          f"{len({k[:4] for k in merged})} distinct (dataset,backbone,L,H)")
    if not args.write:
        print("dry run -- pass --write to commit")
        return
    tmp = DST + ".tmp"
    with open(tmp, "w") as f:
        for r in merged.values():
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, DST)
    print(f"wrote {len(merged)} rows -> {DST}")


if __name__ == "__main__":
    main()
