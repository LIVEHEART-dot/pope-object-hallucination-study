#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
export HF_HOME="${HF_HOME:-$PWD/cache/huggingface}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mode="${1:-smoke}"
model="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
limit=()
case "$mode" in
  smoke) limit=(--limit 12) ;;
  full) ;;
  *) echo 'Usage: bash run.sh smoke|full'; exit 2 ;;
esac
run_dir="outputs/${mode}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$run_dir"
printf '%s\n' "$model" > "$run_dir/model.txt"
nvidia-smi > "$run_dir/gpu.txt"
python -m pip freeze > "$run_dir/environment.txt"
git -C lmms-eval rev-parse HEAD > "$run_dir/lmms_eval_commit.txt"
accelerate launch --num_processes 1 -m lmms_eval \
  --model qwen2_5_vl \
  --model_args "pretrained=$model,attn_implementation=sdpa,max_pixels=1003520" \
  --tasks pope --batch_size 1 --log_samples \
  --output_path "$run_dir" "${limit[@]}" 2>&1 | tee "$run_dir/run.log"
echo "Output: $run_dir"
