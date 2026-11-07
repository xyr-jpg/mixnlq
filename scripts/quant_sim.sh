#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

# export CUDA_VISIBLE_DEVICES=0

# weight quantization
python eval_utils/quant_sim.py \
  --model-path /path/to/Llama-3.1-8B \
  --devices 0 \
  --weight-quant-mode mixnlq \
  --weight-bits 4 \
  --weight-group-size 128 \
  --weight-iterations 100 \
  --activation-quant-mode none \
  --output outputs/mixnlq_w4.pth \
  "$@"

#MIXNLQ calib quantization
# python eval_utils/quant_sim.py \
#   --model-path /path/to/Llama-3.1-8B \
#   --devices 0 \
#   --weight-quant-mode mixnlq \
#   --weight-bits 4 \
#   --weight-group-size 128 \
#   --usecalib \
#   --calibration-data wiki2 \
#   --calibration-samples 128 \
#   --calibration-seqlen 2048 \
#   --output outputs/mixnlq_calib_w4.pth \
#   "$@"

#activation quantization
# python eval_utils/quant_sim.py \
#   --model-path /path/to/Llama-3.1-8B \
#   --devices 0 \
#   --weight-quant-mode none \
#   --weight-bits 4 \
#   --activation-quant-mode mixnlq \
#   --activation-bits 3 \
#   --activation-granularity token \
#   --dynamic-token-iterations 5 \
#   --output outputs/mixnlq_w4a4.pth \
#   "$@"





