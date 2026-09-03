"""Fold the per-host P0-1 fill-in files back into stage0_optimizers.jsonl.

run_stage0_fillin.sh gives each host a DISJOINT set of (L,H) shapes and its own
results/tsf_edge/stage0_fillin_<host>.jsonl, so no two processes ever write one file. Each
fill-in row is already complete: stage0_optimizers.py loaded the existing 8-rate row as `old`,
appended the two new rates, and recomputed sel_/oracle_ over the merged grid.

This merger therefore only has to (1) prove the fill-in did not disturb anything it should not
have and (2) replace the rows. It refuses to write unless every check passes:

  * disjoint     -- no cell appears in two host files
  * complete     -- every merged row has all 10 shared rates for lion/adafactor/signsgd
  * unchanged    -- the 8 pre-existing rates, `static` and `warmup` are bit-identical to the
                    original file (a mismatch means the host's numerics drifted from melon's,
                    which would make the within-cell rate comparison apples-to-oranges)
  * bracketed    -- reports how many per-cell oracles still sit on a grid EDGE, which is the
                    whole point of the pass

Run from the project root:
    .venv/bin/python experiments/tsf_edge/merge_stage0_fillin.py            # dry run
    .venv/bin/python experiments/tsf_edge/merge_stage0_fillin.py --write    # writes + .bak
"""
from __future__ import annotations
import argparse, glob, json, os, shutil, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results", "tsf_edge")
BASE = os.path.join(RES, "stage0_optimizers.jsonl")
ARMS = ("lion", "adafactor", "signsgd")
SHARED = [3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
KEY = lambda r: (r["dataset"], r["backbone"], r["L"], r["H"], r["seed"])


def load(path):
    return {KEY(r): r for r in (json.loads(l) for l in open(path))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    base = load(BASE)
    files = sorted(glob.glob(os.path.join(RES, "stage0_fillin_*.jsonl")))
    if not files:
        sys.exit("no stage0_fillin_*.jsonl found")
    print(f"base {BASE}: {len(base)} rows")

    merged, owner, problems = dict(base), {}, []
    for f in files:
        host = os.path.basename(f)[len("stage0_fillin_"):-len(".jsonl")]
        rows = load(f)
        print(f"  {os.path.basename(f)}: {len(rows)} rows")
        for k, r in rows.items():
            if k in owner:
                problems.append(f"cell {k} appears in both {owner[k]} and {host}")
                continue
            owner[k] = host
            old = base.get(k)
            if old is None:
                problems.append(f"cell {k} ({host}) is not in the base file")
                continue
            for fld in ("static", "warmup", "val_static"):
                if r.get(fld) != old.get(fld):
                    problems.append(f"{k} ({host}): {fld} changed {old.get(fld)} -> {r.get(fld)}")
            for a in ARMS:
                have = {f"{lr:g}" for lr in SHARED}
                miss = have - set(r.get(a, {}))
                if miss:
                    problems.append(f"{k} ({host}): {a} still missing {sorted(miss)}")
                for lr, v in old.get(a, {}).items():          # the 8 pre-existing rates
                    if r.get(a, {}).get(lr) != v:
                        problems.append(f"{k} ({host}): {a}@{lr} changed {v} -> "
                                        f"{r.get(a, {}).get(lr)}")
            r = dict(r, fillin_host=host)
            merged[k] = r

    if problems:
        print(f"\n{len(problems)} PROBLEM(S) -- refusing to write:")
        for p in problems[:40]:
            print("  -", p)
        sys.exit(1)
    print(f"\nall checks passed; {len(owner)}/{len(base)} cells filled in")

    edges = {f"{SHARED[0]:g}", f"{SHARED[-1]:g}"}
    for a in ARMS:
        on_edge = sum(1 for r in merged.values() if f"{r[f'oracle_lr_{a}']:g}" in edges)
        at_old_top = sum(1 for r in merged.values() if abs(r[f"oracle_lr_{a}"] - 1e-2) < 1e-12)
        print(f"  {a:10s} oracle on a grid edge: {on_edge}/{len(merged)}"
              f"   (was-the-edge 1e-2: {at_old_top})")

    if not args.write:
        print("\ndry run -- pass --write to commit")
        return
    shutil.copy2(BASE, BASE + ".bak_pre_fillin")
    tmp = BASE + ".tmp"
    with open(tmp, "w") as f:
        for r in merged.values():
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, BASE)
    print(f"\nwrote {len(merged)} rows -> {BASE}  (backup: {BASE}.bak_pre_fillin)")


if __name__ == "__main__":
    main()
