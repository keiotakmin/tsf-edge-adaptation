#!/usr/bin/env bash
# G2 -- the deployment-configuration slice.
#
# WHY. The conference recipe tells people to adapt FEW parameters (item ii), but every one of
# the extension's 216 cells adapts the full model. Section VIII would therefore recommend a
# configuration on which its own arms were never measured. This runs the four arms the recipe
# names through the two PEFT strategies, on the backbone where PEFT is actually needed
# (PatchTST; DLinear full-model already fits an MCU).
#
# SCOPE. 6 datasets x patchtst x seeds 0-2 x L96/H24 = 18 cells, x {calib, head}. The screen's
# artifacts are never touched: --which writes stage0_optimizers_{calib,head}.jsonl.
#
# RESUMABLE. stage0_optimizers.py re-reads its own output and only runs missing (opt, lr)
# points, so re-running this script after any interruption continues where it stopped. The two
# strategies run SEQUENTIALLY IN THIS SCRIPT -- no watcher process, no pgrep wait loop.
set -eu
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
ARMS=obsign_t1e3,adafactor,adam,sgdm

for W in calib head; do
  echo "=== which=$W  $(date -Is) ==="
  $PY experiments/tsf_edge/stage0_optimizers.py \
      --which "$W" --backbones patchtst --seeds 0,1,2 --L 96 --H 24 \
      --opts-lrful "$ARMS" --opts-lrfree ""
done
echo "=== G2 done $(date -Is) ==="
