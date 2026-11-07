#!/usr/bin/env bash
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

# export CUDA_VISIBLE_DEVICES=0

python eval_utils/speed_acte2e_eval.py \
  --model-path /path/to/Llama-3.1-8B \
  --weight-checkpoint outputs/mixnlq_w4.pth \
  --device 0 \
  --batch-size 1 \
  --prompt-length 2048 \
  --new-tokens 128 \
  --activation-bits 4 \
  --output-json outputs/speed_acte2e.json \
  "$@"
