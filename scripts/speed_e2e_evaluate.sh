#!/usr/bin/env bash
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

# export CUDA_VISIBLE_DEVICES=0

python eval_utils/speed_e2e_evaluate.py \
  --model-path /path/to/Llama-3.1-8B \
  --mixnlq-checkpoint outputs/mixnlq_w4_packed.pth \
  --device 0 \
  --batch-size 1 \
  --prompt-length 2048 \
  --new-tokens 128 \
  --output-json outputs/speed_e2e.json \
  "$@"

#can select
  # --mixnlq-checkpoint outputs/mixnlq_w4_packed.pth \
  # --linear-checkpoint outputs/linear_w4_packed.pth \
  # --lut-checkpoint outputs/lut_w4_packed.pth \
  # --spinquant-checkpoint outputs/linear_w4_packed.pth \
