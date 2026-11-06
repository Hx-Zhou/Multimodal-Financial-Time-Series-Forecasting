set -eu
BASE=/home/liangzixuan/hs300_stageopt_20261101
SHARED=/mnt/nas/share2/home/liangzixuan/hs300_stageopt_20261101
while [ ! -f "$BASE/logs/environment_ready" ]; do sleep 10; done
python3 -m pip install --target "$BASE/runtime" numpy==1.26.4 --index-url https://pypi.tuna.tsinghua.edu.cn/simple --disable-pip-version-check
mkdir -p "$SHARED/project" "$SHARED/logs"
cp -a "$BASE/runtime" "$SHARED/runtime"
cd "$BASE/project"
tar --exclude=package --exclude=__pycache__ -cf - . | tar -xf - -C "$SHARED/project"
export PYTHONPATH="$BASE/runtime:$BASE/project"
CUDA_VISIBLE_DEVICES=0 python3 -u -m experiments.gpu_stage_optimization.validate > "$BASE/logs/gpu_integrity.json"
python3 -u -m experiments.gpu_stage_optimization.prepare --package "$BASE/project/package" --output "$SHARED/cache"
touch "$SHARED/logs/setup_ready"
