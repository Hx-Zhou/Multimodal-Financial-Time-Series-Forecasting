#!/bin/bash
set -eu
SHARED=/mnt/nas/share2/home/liangzixuan/hs300_stageopt_20261101
LOCAL=/home/liangzixuan/hs300_stageopt_20261101
export PYTHONPATH="$LOCAL/runtime:$SHARED/project"
cd "$SHARED/project"
OUT="$SHARED/round2"
until [ -f "$OUT/FINAL_SELECTION.json" ]; do sleep 15; done
joint=$(python3 -c "import json;print(json.load(open('$OUT/FINAL_SELECTION.json'))['multimodal'])")
for seed in 43 44; do
  until [ -f "$OUT/$joint/seed$seed/development/result.json" ]; do
    if grep -q 'Traceback' "$OUT"/logs/replicate*.log 2>/dev/null; then echo 'Replication failed'; exit 1; fi
    sleep 15
  done
done
export CUDA_VISIBLE_DEVICES=4
python3 -u -m experiments.temporal_robust_round2.finalize \
  --root "$OUT" --cache "$SHARED/cache" --package "$LOCAL/project/package" --action export
# Export must pass standalone and raw-unlabeled development checks before any final test.
python3 -u -m experiments.temporal_robust_round2.finalize \
  --root "$OUT" --cache "$SHARED/cache" --package "$LOCAL/project/package" --action test
echo 'All optimization, replication and the single final holdout batch completed.'
