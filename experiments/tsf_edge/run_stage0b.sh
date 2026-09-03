#!/usr/bin/env bash
# Stage 0b full grid: 6 (L,H) shapes x 6 datasets x 2 backbones x 3 seeds = 216 cells,
# 52 (optimizer, lr) points per cell. ~10 h on an idle A100 (smoke-measured: 82 s/cell
# DLinear, 255 s/cell PatchTST at L=96/H=24).
#
# Resumable by construction: stage0_optimizers.py recomputes only the missing (opt, lr)
# points, so re-running after a kill costs only the per-cell warmup. Writes to
# results/tsf_edge/stage0b_optimizers.jsonl -- a SEPARATE file, so the 216-row Stage-0
# artifact is never rewritten.
#
# Launch (survives the Claude Code session dying; see CLAUDE.md):
#   loginctl enable-linger "$USER"
#   setsid nohup bash experiments/tsf_edge/run_stage0b.sh >/dev/null 2>&1 &
# Watch:  tail -f /var/tmp/fujimoto/stage0b_optimizers.log
set -u
cd /auto/proj/fujimoto/tsf_edge || exit 1
LOG=/var/tmp/fujimoto/stage0b_optimizers.log
mkdir -p /var/tmp/fujimoto

# Gate on the correctness suite: never spend grid time on unverified update rules.
if ! .venv/bin/python experiments/tsf_edge/test_online_optimizers.py >>"$LOG" 2>&1; then
    echo "=== ABORT $(date -Is): test_online_optimizers.py FAILED ===" >>"$LOG"
    exit 1
fi

for LH in "96 24" "192 24" "96 48" "192 48" "96 96" "192 96"; do
    set -- $LH
    echo "=== L=$1 H=$2 start $(date -Is) ===" >>"$LOG"
    .venv/bin/python experiments/tsf_edge/stage0_optimizers.py --stage 0b \
        --L "$1" --H "$2" >>"$LOG" 2>&1
done
echo "=== ALL DONE $(date -Is) ===" >>"$LOG"
