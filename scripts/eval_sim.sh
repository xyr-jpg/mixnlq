#!/usr/bin/env bash
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

# export CUDA_VISIBLE_DEVICES=0

python eval_utils/evalsim_lm_eval.py \
  --model-path /path/to/Llama-3.1-8B \
  --sim-checkpoint outputs/mixnlq_w4.pth \
  --tasks arc_challenge arc_easy boolq piqa hellaswag \
  --ppl-tasks wikitext2 c4 \
  --devices 0 \
  --output-json outputs/eval_sim.json
  "$@"
