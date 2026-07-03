#!/usr/bin/env bash
# Download the non-shipped datasets (ETT x4, UCI Appliances) into this directory and verify
# them against checksums.sha256 = the exact files used in the paper. bdg2.csv is shipped.
set -uo pipefail
cd "$(dirname "$0")"

ETT_BASE=https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small
for f in ETTh1 ETTh2 ETTm1 ETTm2; do
  [ -f "$f.csv" ] || curl -fL -o "$f.csv" "$ETT_BASE/$f.csv"
done

if [ ! -f appliances.csv ]; then
  curl -fL -o appliances.csv \
    https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv \
  || curl -fL -o appliances.csv \
    https://raw.githubusercontent.com/LuisM78/Appliances-energy-prediction-data/master/energydata_complete.csv
fi

echo
if sha256sum -c checksums.sha256; then
  echo "OK: all files match the versions used in the paper."
else
  echo "WARNING: some checksums differ from the files used in the paper (upstream may have"
  echo "changed). Results may deviate slightly; the paper's artifacts in results/tsf_edge/"
  echo "were produced from the checksummed versions."
  exit 1
fi
