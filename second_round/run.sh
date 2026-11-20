#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
phase="${1:-dev}"
case "$phase" in dev|test) ;; *) echo 'Usage: bash second_round/run.sh dev|test'; exit 2;; esac
source ../.venv/bin/activate
export HF_HOME="${HF_HOME:-$(cd .. && pwd)/cache/huggingface}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=60
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python prepare_cloud.py "$phase"
run_dir="outputs/${phase}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$run_dir"
cp "tasks/$phase/manifest.json" "$run_dir/manifest.json"
cp image_split.json "$run_dir/image_split.json"
cp "metadata_$phase.json" "$run_dir/metadata.json"
cp -r "tasks/$phase" "$run_dir/task_configs"
python -m pip freeze > "$run_dir/environment.txt"
git -C ../lmms-eval rev-parse HEAD > "$run_dir/lmms_eval_commit.txt"
nvidia-smi > "$run_dir/gpu.txt"
model="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
printf '%s\n' "$model" > "$run_dir/model.txt"
accelerate launch --num_processes 1 -m lmms_eval \
  --model qwen2_5_vl \
  --model_args "pretrained=$model,attn_implementation=sdpa,max_pixels=1003520" \
  --include_path "tasks/$phase" --tasks pope_A,pope_B,pope_C \
  --batch_size 1 --log_samples --output_path "$run_dir" 2>&1 | tee "$run_dir/run.log"
python compare.py "$run_dir"
echo "SUCCESS: $run_dir"
