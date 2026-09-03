#!/usr/bin/env bash
# Stage 0c: the two optimizers DESIGNED FROM the Stage-0b frontier (online_optimizers.ObSign),
# on the same 216 cells and the same protocol as every other contender.
#
#   obsign   0x state, 12 rates (shared grid + 3e-1, 1.0 -- the plateau above the knee needs
#            reach). alpha = min(lr, tau*RMS(p)), tau pinned at 1e-2 and NOT tuned, so lr stays
#            the single swept hyperparameter and this is exactly "signSGD + one guard".
#   relsign  0x state, 10 rates (shared grid). alpha = tau*RMS(p); lr IS tau, dimensionless.
#
# tau is SWEPT as three arms (1e-2 / 3e-3 / 1e-3) because it has no published default; the
# first run showed tau=1e-2 pins the plateau at +2.7 while RelSign says 1e-3 should pin it near
# +13.1. The target is the empty top-left of both trade-off panels: signSGD's quality (mean +14.1,
# and +14.7 at a single global rate) with ObGD's flat response to a mis-set rate (mis1x 0.09
# against signSGD's 16.6). Falsified if obsign's plateau sits far below signSGD's peak.
#
# Writes results/tsf_edge/stage0c_optimizers.jsonl -- a THIRD file, so neither the Stage-0
# (216 rows) nor the Stage-0b artifact is touched. Resumable like the others.
#
# Launch:  setsid nohup bash experiments/tsf_edge/run_stage0c.sh >/dev/null 2>&1 &
# Watch:   tail -f /var/tmp/fujimoto/stage0c_optimizers.log
set -u
cd /auto/proj/fujimoto/tsf_edge || exit 1
LOG=/var/tmp/fujimoto/stage0c_optimizers.log
OUT=results/tsf_edge/stage0c_optimizers.jsonl
mkdir -p /var/tmp/fujimoto

if ! .venv/bin/python experiments/tsf_edge/test_online_optimizers.py >>"$LOG" 2>&1; then
    echo "=== ABORT $(date -Is): test_online_optimizers.py FAILED ===" >>"$LOG"
    exit 1
fi

for LH in "96 24" "192 24" "96 48" "192 48" "96 96" "192 96"; do
    set -- $LH
    echo "=== L=$1 H=$2 start $(date -Is) ===" >>"$LOG"
    .venv/bin/python experiments/tsf_edge/stage0_optimizers.py --stage 0c \
        --out "$OUT" --L "$1" --H "$2" >>"$LOG" 2>&1
done
echo "=== ALL DONE $(date -Is) ===" >>"$LOG"
