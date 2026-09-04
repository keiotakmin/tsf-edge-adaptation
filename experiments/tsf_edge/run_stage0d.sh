#!/usr/bin/env bash
# Stage 0d (2026-09-04): three more values of tau, on the same 216 cells, the same protocol and
# the SAME shared 10-rate grid as every other contender.
#
#   obsign_t1p5e3   tau = 1.5e-3   knee 8.0e-5   1.10 decades below the shipped 1e-3
#   obsign_t2e3     tau = 2e-3     knee 1.1e-4   0.98 decades below
#   obsign_t5e3     tau = 5e-3     knee 2.7e-4   0.58 decades below
#
# WHY. The paper's design rule for tau is a MARGIN rule: put the knee at least a decade below
# the rate you intend to ship. Stage 0c sampled that margin at three points only -- 1.28
# decades (tau=1e-3, deployable), 0.80 (3e-3, 10 cells below the static baseline) and 0.28
# (1e-2, 49 cells) -- so the rule could be stated but its boundary could not be located, which
# is what the limitations section had to concede. These three bracket the crossing to inside
# 0.1-0.2 decades: 1.10 and 0.98 sit either side of one decade, and 0.58 fills the failing side
# between 3e-3 and 1e-2. A reviewer asking "why one decade and not 0.5?" gets a measured answer.
#
# NOT a re-run of anything: writes results/tsf_edge/stage0d_optimizers.jsonl, a FOURTH file, so
# the Stage-0/0b/0c artifacts are untouched (CLAUDE.md: a new prefix for every added run).
# Resumable: stage0_optimizers.need_of() recomputes only the missing (opt, lr) points, so an
# interrupted pass re-pays each cell's warmup once and nothing else.
#
# Launch (survives the Claude Code session dying -- CLAUDE.md):
#   loginctl enable-linger fujimoto
#   setsid nohup bash experiments/tsf_edge/run_stage0d.sh >/dev/null 2>&1 &
# Watch:   tail -f /var/tmp/fujimoto/stage0d_optimizers.log
# Done when the log's last line is "=== ALL DONE".
set -u
cd /auto/proj/fujimoto/tsf_edge || exit 1
LOG=/var/tmp/fujimoto/stage0d_optimizers.log
OUT=results/tsf_edge/stage0d_optimizers.jsonl
mkdir -p /var/tmp/fujimoto

# Same gate as Stage 0c: the guard identity (below the knee ObSign IS signSGD; above it the
# rate cancels) is what the whole tau argument rests on, so it is asserted before any GPU time
# is spent.
if ! .venv/bin/python experiments/tsf_edge/test_online_optimizers.py >>"$LOG" 2>&1; then
    echo "=== ABORT $(date -Is): test_online_optimizers.py FAILED ===" >>"$LOG"
    exit 1
fi

for LH in "96 24" "192 24" "96 48" "192 48" "96 96" "192 96"; do
    set -- $LH
    echo "=== L=$1 H=$2 start $(date -Is) ===" >>"$LOG"
    .venv/bin/python experiments/tsf_edge/stage0_optimizers.py --stage 0d \
        --out "$OUT" --L "$1" --H "$2" >>"$LOG" 2>&1
done
echo "=== ALL DONE $(date -Is) ===" >>"$LOG"
