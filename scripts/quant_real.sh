#!/usr/bin/env bash
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

# export CUDA_VISIBLE_DEVICES=0

python eval_utils/quant_real.py \
  --model-path /path/to/Llama-3.1-8B \
  --quant-mode mixnlq \
  --group-size 128 \
  --output outputs/mixnlq_w4_packed.pth \
  --devices 0
  "$@"
