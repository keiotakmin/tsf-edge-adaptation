#!/usr/bin/env bash
# Stage 0b fill-in pass -- ONE sweep that closes every gap the main run left, so each cell pays
# its warmup only once:
#   * shapes 1-4 predate `autostep` and predate UPGD's top extension -> those points are missing
#   * every shape needs 3e-1 and 1.0 for the SGD-family rules (sgdm/dons/idbd/autostep/upgd),
#     which selected the shared grid's TOP rate at H=96 (fewer, larger updates at stride=H)
#   * sgd and adam get MATCHED readings at exactly those two rates, so the bracketing question
#     is answered for the references and not only for the contenders
# Grids come from LR_GRID_BY_OPT; the runner recomputes only missing (optimizer, rate) points.
# Headline columns are computed on SHARED_LR (3e-6..1e-1) regardless, so these extra rates
# inform the bracketing check without widening anyone's effective search.
#
# Launch after the main run reports ALL DONE:
#   setsid nohup bash experiments/tsf_edge/run_stage0b_fillin.sh >/dev/null 2>&1 &
set -u
cd /auto/proj/fujimoto/tsf_edge || exit 1
LOG=/var/tmp/fujimoto/stage0b_optimizers.log
OPTS=sgdm,obgd,adaptive_obgd,dons,upgd,idbd,autostep,sgd,adam

if ! .venv/bin/python experiments/tsf_edge/test_online_optimizers.py >>"$LOG" 2>&1; then
    echo "=== FILLIN ABORT $(date -Is): test_online_optimizers.py FAILED ===" >>"$LOG"
    exit 1
fi

for LH in "96 24" "192 24" "96 48" "192 48" "96 96" "192 96"; do
    set -- $LH
    echo "=== FILLIN L=$1 H=$2 start $(date -Is) ===" >>"$LOG"
    .venv/bin/python experiments/tsf_edge/stage0_optimizers.py --stage 0b \
        --opts-lrful "$OPTS" --opts-lrfree "" --L "$1" --H "$2" >>"$LOG" 2>&1
done
echo "=== FILLIN ALL DONE $(date -Is) ===" >>"$LOG"
