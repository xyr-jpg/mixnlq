#!/usr/bin/env bash
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

# export CUDA_VISIBLE_DEVICES=0

python eval_utils/evalreal_lm_eval.py \
  --model-path /path/to/Llama-3.1-8B \
  --checkpoint outputs/mixnlq_w4_packed.pth \
  --quant-mode mixnlq \
  --tasks arc_challenge arc_easy boolq piqa hellaswag \
  --ppl-tasks wikitext2 c4 \
  --output-json outputs/eval_real.json \
  --device 0 \
  "$@"
