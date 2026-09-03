#!/usr/bin/env bash
# P0-1 fill-in pass: close the bracketing break in Stage 0's three LR-ful arms.
#
# WHY: lion/adafactor/signsgd were run 2026-07-10/11, BEFORE the shared grid's top extension
# to 3e-2 and 1e-1 (2026-08-06, R1). They therefore hold only 8 of the 10 shared rates, and
# AdaFactor's per-cell oracle lands on the grid EDGE 1e-2 in 45/216 cells (21%) -- a direct
# violation of the bracketing rule, which biases AdaFactor's quality low and its mis1x
# optimistically (the "1 decade up" step has no 3e-2 to read). lion 0/216 and signsgd 0/216
# are healthy, but they get the same two rates so the three arms keep an identical grid.
# The ObSign-vs-AdaFactor "dominates on all three axes" claim is provisional until this lands.
#
# 216 cells x 3 arms x 2 rates = 1296 (opt,lr) points. stage0_optimizers.py is resumable by
# construction: need_of() recomputes only the MISSING (opt,lr) points, so each cell pays its
# warmup once and then 6 online passes.
#
# PARALLEL SAFETY: melon is occupied, so this runs on iris/modulex. Each host takes a disjoint
# slice (SHAPES x BACKBONES) and writes its OWN jsonl seeded with only its own rows -- two
# hosts writing one file would clobber each other (flush() rewrites the whole file from the
# rows it loaded at startup). results/tsf_edge/stage0_optimizers.jsonl is NEVER written by
# this pass; merge_stage0_fillin.py folds the per-host files back in afterwards.
#
# WHICH HOST GETS WHAT is not free: the 8 pre-existing rates in every cell were computed on
# melon (A100 80GB PCIe), and this pass re-runs warm_and_select to rebuild the model before
# adding the two new rates. A 2-cell probe on 2026-08-31 showed iris (A100, same part)
# reproduces melon's static baseline inside the built-in 0.1% cross-check, while modulex
# (RTX 4090, sm_89) misses it on PatchTST by 0.92% (0.42539 vs 0.42152 on appliances/s0) and
# passes on DLinear. Mixing a melon-warmed 8-point surface with a 4090-warmed 2-point one
# inside ONE cell would make that cell's rate comparison apples-to-oranges -- the exact
# unfairness this paper is about -- so PatchTST goes to iris and DLinear to modulex, and the
# log's WARNING lines are audited afterwards (any warned cell is recomputed on iris).
#
# Launch (survives the Claude Code session dying; see CLAUDE.md):
#   ssh iris    'BACKBONES=patchtst setsid nohup bash \
#       /auto/proj/fujimoto/tsf_edge/experiments/tsf_edge/run_stage0_fillin.sh >/dev/null 2>&1 &'
#   ssh modulex 'BACKBONES=dlinear  setsid nohup bash ... &'
# Watch:  tail -f /var/tmp/fujimoto/stage0_fillin_<host>.log
set -u
cd /auto/proj/fujimoto/tsf_edge || exit 1
HOST=$(hostname)
SHAPES=${SHAPES:-"96 24|192 24|96 48|192 48|96 96|192 96"}
BACKBONES=${BACKBONES:?set BACKBONES: patchtst or dlinear (see the host note above)}
OUT=results/tsf_edge/stage0_fillin_$HOST.jsonl
LOG=/var/tmp/fujimoto/stage0_fillin_$HOST.log
mkdir -p /var/tmp/fujimoto

# Seed this host's file with the EXISTING 8-rate rows for its shapes only -- without them the
# resume logic would see no `old` row and recompute all 10 rates (5x the work) into a file
# whose readings then came from a different GPU than the paper's. Never re-seed a file that
# already exists: that would discard work done before a restart.
if [ ! -f "$OUT" ]; then
    .venv/bin/python - "$OUT" "$SHAPES" "$BACKBONES" >>"$LOG" 2>&1 <<'PY'
import json, sys
out, shapes, backbones = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
want = {tuple(int(x) for x in s.split()) for s in shapes.split("|")}
rows = [json.loads(l) for l in open("results/tsf_edge/stage0_optimizers.jsonl")]
mine = [r for r in rows if (r["L"], r["H"]) in want and r["backbone"] in backbones]
open(out, "w").write("".join(json.dumps(r) + "\n" for r in mine))
print(f"seeded {len(mine)} rows for shapes {sorted(want)} x {backbones} -> {out}", flush=True)
PY
else
    echo "=== $(date -Is) resuming: $OUT already exists ===" >>"$LOG"
fi

echo "=== FILLIN START $HOST $(date -Is) shapes=$SHAPES backbones=$BACKBONES ===" >>"$LOG"
IFS='|' read -ra SH <<< "$SHAPES"
for LH in "${SH[@]}"; do
    set -- $LH
    echo "=== L=$1 H=$2 start $(date -Is) ===" >>"$LOG"
    .venv/bin/python experiments/tsf_edge/stage0_optimizers.py --stage 0 \
        --opts-lrful lion,adafactor,signsgd --opts-lrfree "" --backbones "$BACKBONES" \
        --L "$1" --H "$2" --out "$OUT" >>"$LOG" 2>&1
done
echo "=== FILLIN ALL DONE $HOST $(date -Is) ===" >>"$LOG"
