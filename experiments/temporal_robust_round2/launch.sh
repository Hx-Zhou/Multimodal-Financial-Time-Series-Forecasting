#!/bin/bash
set -eu
SHARED=/mnt/nas/share2/home/liangzixuan/hs300_stageopt_20261101
LOCAL=/home/liangzixuan/hs300_stageopt_20261101
export PYTHONPATH="$LOCAL/runtime:$SHARED/project"
cd "$SHARED/project"
mkdir -p "$SHARED/round2/logs"
launch_lane() {
  lane="$1"
  gpu="$2"
  if [ -f "$SHARED/round2/logs/$lane.pid" ] && kill -0 "$(cat "$SHARED/round2/logs/$lane.pid")" 2>/dev/null; then
    echo "$lane already running"
    return
  fi
  module=experiments.temporal_robust_round2.run
  case "$lane" in sym_*|news|reports|joint) module=experiments.temporal_robust_round2.debias ;; esac
  nohup env CUDA_VISIBLE_DEVICES="$gpu" python3 -u -m "$module" \
    --cache "$SHARED/cache" --calendar experiments/temporal_robust_round2/calendar.npy \
    --output "$SHARED/round2" --lane "$lane" \
    > "$SHARED/round2/logs/$lane.log" 2>&1 < /dev/null &
  echo $! > "$SHARED/round2/logs/$lane.pid"
  echo "Started $lane on GPU $gpu PID $!"
}
if [ "$#" -eq 2 ]; then
  launch_lane "$1" "$2"
else
  launch_lane linear 2
  launch_lane recurrent 7
  launch_lane patch 4
fi
