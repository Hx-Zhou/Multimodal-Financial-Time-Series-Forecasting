#!/bin/bash
set -eu
SHARED=/mnt/nas/share2/home/liangzixuan/hs300_stageopt_20261101
LOCAL=/home/liangzixuan/hs300_stageopt_20261101
export PYTHONPATH="$LOCAL/runtime:$SHARED/project"
cd "$SHARED/project"
OUT="$SHARED/round2"
LAUNCH=experiments/temporal_robust_round2/launch.sh
wait_complete() {
  for name in "$@"; do
    until [ -f "$OUT/$name/seed42/development/result.json" ]; do
      if grep -q 'Traceback' "$OUT"/logs/*.log; then echo 'A training log contains a failure'; exit 1; fi
      sleep 15
    done
  done
}
wait_complete E1_MAE E1_Huber E2_GRU64 E2_GRU16 E3_drop01 E3_drop03 E4_full E4_small E5_layers2 E5_layers1 E6_raw E6_stationary \
  S_E1_Huber S_E2_GRU16 S_E3_drop03 S_E4_small S_E5_layers1 S_E6_stationary
python3 -u -m experiments.temporal_robust_round2.select --root "$OUT" --phase market > "$OUT/logs/market_decision.log"
bash "$LAUNCH" news 2
bash "$LAUNCH" reports 7
wait_complete E7_daily E7_pooled E7_coverage E8_values E8_ratios E8_masks E10_modes13 E10_modes3
python3 -u -m experiments.temporal_robust_round2.select --root "$OUT" --phase modal > "$OUT/logs/modal_decision.log"
bash "$LAUNCH" joint 4
wait_complete E9_dropout01 E9_dropout05
python3 -u -m experiments.temporal_robust_round2.select --root "$OUT" --phase final > "$OUT/logs/final_decision.log"
for seed in 43 44; do
  gpu=2; if [ "$seed" -eq 44 ]; then gpu=7; fi
  nohup env CUDA_VISIBLE_DEVICES="$gpu" python3 -u -m experiments.temporal_robust_round2.debias \
    --cache "$SHARED/cache" --calendar experiments/temporal_robust_round2/calendar.npy \
    --output "$OUT" --lane replicate --seed "$seed" \
    > "$OUT/logs/replicate$seed.log" 2>&1 < /dev/null &
  echo $! > "$OUT/logs/replicate$seed.pid"
done
echo 'Stage selection complete; replications launched. Holdout has not been evaluated.'
