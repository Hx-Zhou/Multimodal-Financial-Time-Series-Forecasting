#!/bin/bash
set -eu
SHARED=/mnt/nas/share2/home/liangzixuan/hs300_stageopt_20261101
GPU=$1
LANES=$2
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$SHARED/runtime:$SHARED/project"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
while [ ! -f "$SHARED/logs/setup_ready" ]; do sleep 10; done
cd "$SHARED/project"
python3 -u -m experiments.gpu_stage_optimization.validate
for LANE in $LANES; do
  echo "START $LANE $(date -Iseconds)"
  python3 -u -m experiments.gpu_stage_optimization.train --cache "$SHARED/cache" --output "$SHARED/results" --lane "$LANE" --seed 42 --epochs 12 --batch 512
  echo "COMPLETE $LANE $(date -Iseconds)"
done
