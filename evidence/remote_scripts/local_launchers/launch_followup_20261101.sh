#!/bin/bash
set -eu
SHARED=/mnt/nas/share2/home/liangzixuan/hs300_stageopt_20261101
export CUDA_VISIBLE_DEVICES="$1"
LANE=$2
export PYTHONPATH="$SHARED/runtime:$SHARED/project"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
cd "$SHARED/project"
if [ "$LANE" = reports ]; then
  python3 -u -m experiments.gpu_targeted_followup.ratios --package /home/liangzixuan/hs300_stageopt_20261101/project/package --cache "$SHARED/cache"
  while ! grep -q 'COMPLETE patch' "$SHARED/logs/243_patch.log"; do sleep 5; done
fi
python3 -u -m experiments.gpu_targeted_followup.run --cache "$SHARED/cache" --output "$SHARED/followup_results" --first-output "$SHARED/results" --lane "$LANE"
echo "COMPLETE followup $LANE $(date -Iseconds)"
