#!/usr/bin/env bash
# P0-2: extend the HEADLINE arms from 3 seeds to 5.
#
# WHY: the extension grid is 216 cells at seeds 0-2 while the BigData grid is 360 cells at
# seeds 0-4. Three seeds meets the project's minimum, and the reference arms already agree
# across the two grids (Adam +14.16 vs +14.17, negative-cell rate 49% vs 48%), so the whole
# 20-arm screen does not need re-running. What is worth closing is the gap on the arms the
# extension's claims actually rest on.
#
# ARMS: lion, adafactor, signsgd (Stage 0) + obsign x3, relsign (Stage 0c) = 7.
# adam and sgdm are NOT here -- lr_fairness.jsonl already carries seeds 3 and 4 for them
# (72 cells each), and the stage files read those arms from there.
#
# WHERE THE RESULT GOES: its OWN file, never merged into the 216-row artifacts. The headline
# tables stay on the 216 common cells so all 20 arms are compared on one cell set; the 5-seed
# numbers are a separate robustness statement ("extending these 9 arms to 5 seeds moves the
# mean by at most X pt and does not change the ordering"). Merging would make the main tables
# ragged -- 360 cells for nine arms, 216 for the rest -- which is a worse comparison than the
# one it would be trying to improve.
#
# HOST SPLIT: as in the P0-1 fill-in, PatchTST on an A100 and DLinear on the 4090, because a
# 2-cell probe showed modulex missing melon's static baseline by 0.92% on PatchTST and passing
# on DLinear. stage0_optimizers.py re-checks every cell's static against lr_fairness.jsonl --
# which HAS seeds 3 and 4 -- and prints a WARNING when they disagree, so the split is verified
# rather than assumed. Audit the logs for WARNING lines before using the output.
#
# Launch:
#   ssh iris    'BACKBONES=patchtst setsid nohup bash \
#       /auto/proj/fujimoto/tsf_edge/experiments/tsf_edge/run_stage0_seeds.sh >/dev/null 2>&1 &'
#   ssh modulex 'BACKBONES=dlinear  setsid nohup bash ... &'
# Watch: tail -f /var/tmp/fujimoto/stage0_seeds_<host>.log
set -u
cd /auto/proj/fujimoto/tsf_edge || exit 1
HOST=$(hostname)
BACKBONES=${BACKBONES:?set BACKBONES: patchtst or dlinear (see the host note above)}
SEEDS=${SEEDS:-3,4}
ARMS=lion,adafactor,signsgd,obsign,obsign_t3e3,obsign_t1e3,relsign
OUT=results/tsf_edge/stage0_seeds34_$HOST.jsonl
LOG=/var/tmp/fujimoto/stage0_seeds_$HOST.log
mkdir -p /var/tmp/fujimoto

echo "=== SEEDS START $HOST $(date -Is) backbones=$BACKBONES seeds=$SEEDS ===" >>"$LOG"
for LH in "96 24" "192 24" "96 48" "192 48" "96 96" "192 96"; do
    set -- $LH
    echo "=== L=$1 H=$2 start $(date -Is) ===" >>"$LOG"
    .venv/bin/python experiments/tsf_edge/stage0_optimizers.py --stage 0c \
        --opts-lrful "$ARMS" --opts-lrfree "" --backbones "$BACKBONES" --seeds "$SEEDS" \
        --L "$1" --H "$2" --out "$OUT" >>"$LOG" 2>&1
done
echo "=== SEEDS ALL DONE $HOST $(date -Is) ===" >>"$LOG"
