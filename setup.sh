#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__, torch.cuda.get_device_name(0))'
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
if [ ! -d lmms-eval ]; then
  git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
fi
git -C lmms-eval checkout --detach 776c61623d8ba796dcf1986781ae99cd92b47297
python -m pip install -e ./lmms-eval
python -m pip install qwen-vl-utils
mkdir -p outputs
git -C lmms-eval rev-parse HEAD > outputs/lmms_eval_commit.txt
python -m pip freeze > outputs/environment.txt
python -c 'from transformers import Qwen2_5_VLForConditionalGeneration; import qwen_vl_utils; import lmms_eval; print("Imports OK")'
