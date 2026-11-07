# MixNLQ
This repository contains the code of our Neurips2026 paper "MixNLQ: An Effective Post-Training Nonlinear Low-Bit Quantization Method for Large Language Models" 

This README covers the quantization, model evaluation, and end-to-end speed benchmark scripts. Run all `scripts/*.sh` scripts from the project root. 

## Environment Setup

```bash
conda create -n mixnlq python=3.10 -y
conda activate mixnlq

# Install PyTorch for the CUDA version used by your setup. CUDA 12.4 is shown here.
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

Before building the CUDA extension, make sure `CUDA_HOME` points to a CUDA Toolkit installation containing `bin/nvcc` and is compatible with the CUDA version used to build PyTorch:

```bash
export CUDA_HOME=/usr/local/cuda-12.4  # Change this to your local CUDA Toolkit path
export PATH="$CUDA_HOME/bin:$PATH"
python setup_cuda.py build_ext --inplace
```

## Script Overview

| Script | Purpose |
| --- | --- |
| `scripts/quant_sim.sh` | Apply pseudo-quantization to a BF16 model. Supports weight quantization, activation quantization, and optional PPL evaluation; saves a pseudo-quantized checkpoint. |
| `scripts/quant_real.sh` | Export a checkpoint with weights actually packed at 3 or 4 bits. |
| `scripts/eval_sim.sh` | Run lm-evaluation-harness on a BF16 model or, with `--sim-checkpoint`, a `quant_sim.py` checkpoint. WikiText-2 and C4 PPL evaluation is enabled by default. |
| `scripts/eval_real.sh` | Run lm-evaluation-harness on a BF16 model or a packed checkpoint. WikiText-2 and C4 PPL evaluation is enabled by default. |
| `scripts/speed_e2e_evaluate.sh` | Benchmark end-to-end inference speed for BF16 and supplied MixNLQ, Linear, LUT, and other supported checkpoints. |
| `scripts/speed_acte2e_eval.sh` | Compare end-to-end speed for BF16 activations, MixNLQ activation pseudo-quantization, and Linear/RTN activation pseudo-quantization. |

Replace the model and checkpoint paths in the examples with paths on your machine. The shell scripts provide default arguments; extra arguments appended to the command are forwarded to the Python program and can override those defaults.

## Usage Examples

### 1. Pseudo-Quantize Weights

This example applies MixNLQ W4A16 pseudo-quantization with a group size of 128. Use `--output` to choose where to save the checkpoint:

```bash
bash scripts/quant_sim.sh \
  --model-path /path/to/Llama-3.1-8B \
  --devices 0 \
  --weight-quant-mode mixnlq \
  --weight-bits 4 \
  --weight-group-size 128 \
  --activation-quant-mode none \
  --output outputs/mixnlq_w4.pth
```

To use calibration data for weight quantization:

```bash
bash scripts/quant_sim.sh \
  --model-path /path/to/Llama-3.1-8B \
  --weight-quant-mode mixnlq \
  --weight-bits 4 \
  --usecalib \
  --calibration-data wiki2 \
  --calibration-samples 128 \
  --output outputs/mixnlq_calib_w4.pth
```

To quantize activations while keeping weights in BF16:

```bash
bash scripts/quant_sim.sh \
  --model-path /path/to/Llama-3.1-8B \
  --weight-quant-mode none \
  --activation-quant-mode mixnlq \
  --activation-bits 4 \
  --activation-granularity token \
  --output outputs/mixnlq_activation_w16a4.pth
```

#### Use Local Calibration Datasets

`quant_sim.sh` and `quant_real.sh` load calibration data from Hugging Face with `load_dataset` by default. Add `--use-local` to load the datasets from local directories instead. Both directories must be readable by `datasets.load_dataset(path, "default", split=...)` and provide a `text` column. Override them with `--local-wiki2-path` and `--local-c4-path`. For example, to calibrate from a local C4 copy:

```bash
bash scripts/quant_sim.sh \
  --model-path /path/to/Llama-3.1-8B \
  --weight-quant-mode mixnlq \
  --weight-bits 4 \
  --usecalib \
  --calibration-data c4 \
  --calibration-samples 128 \
  --use-local \
  --local-c4-path /path/to/c4-10k \
  --output outputs/mixnlq_c4_calib_w4.pth
```

For mixed calibration sets, such as `wiki2-c4`, pass both local paths. The same local-data options are available in `quant_real.sh`. They affect quantization calibration; the evaluation scripts' PPL dataset loading is configured separately.

### 2. Export a Packed Quantized Checkpoint

This example exports MixNLQ W4A16. The current `quant_real.py` modes are `mixnlq`, `mixnlq3`, `linear`, `linear3`, `lut`, and `lut3`:

```bash
bash scripts/quant_real.sh \
  --model-path /path/to/Llama-3.1-8B \
  --quant-mode mixnlq \
  --group-size 128 \
  --output outputs/mixnlq_w4_packed.pth \
  --devices 0
```

For 3-bit MixNLQ, use `--quant-mode mixnlq3`. To use calibration data for activation-aware weight quantization, add `--usecalib` and set `--calibration-data` and `--calibration-samples` as needed.

### 3. Evaluate Accuracy and PPL

`eval_sim.sh` evaluates a BF16 model by default. Add `--sim-checkpoint` to evaluate a pseudo-quantized checkpoint:

```bash
bash scripts/eval_sim.sh \
  --model-path /path/to/Llama-3.1-8B \
  --sim-checkpoint outputs/mixnlq_w4.pth \
  --tasks arc_challenge arc_easy boolq piqa hellaswag \
  --ppl-tasks wikitext2 c4 \
  --output-json outputs/eval_sim.json
```

`eval_real.sh` evaluates a BF16 model or a packed checkpoint. Omit `--checkpoint` for BF16 evaluation. This example evaluates MixNLQ W4A16:

```bash
bash scripts/eval_real.sh \
  --model-path /path/to/Llama-3.1-8B \
  --checkpoint outputs/mixnlq_w4_packed.pth \
  --quant-mode mixnlq \
  --tasks arc_challenge arc_easy boolq piqa hellaswag \
  --ppl-tasks wikitext2 c4 \
  --output-json outputs/eval_real.json
```

Both evaluation programs accept `--tasks` for lm-evaluation-harness tasks, `--num-fewshot` for the number of few-shot examples, `--limit` to cap the number of examples, and `--ppl-tasks` to select `wikitext2` and/or `c4`. To skip PPL evaluation, pass `--ppl-tasks` without any dataset names. `eval_sim.sh` uses `--devices` and can run across multiple GPUs (for example, `0,1`); `eval_real.sh` uses the single-GPU `--device` option.

### 4. Benchmark End-to-End Speed

`speed_e2e_evaluate.sh` can compare multiple model variants. Supply the checkpoint for each variant you want to include. This example compares the BF16 baseline with MixNLQ W4:

```bash
bash scripts/speed_e2e_evaluate.sh \
  --model-path /path/to/Llama-3.1-8B \
  --mixnlq-checkpoint outputs/mixnlq_w4_packed.pth \
  --device 0 \
  --batch-size 1 \
  --prompt-length 2048 \
  --new-tokens 128 \
  --output-json outputs/speed_e2e.json
```

You can also provide options such as `--linear-checkpoint`, `--mixnlq3-checkpoint`, `--linear3-checkpoint`, and `--lut-checkpoint`/`--lut3-checkpoint` to include more variants. To measure only the BF16 baseline, provide `--model-path` without quantized checkpoints.

`speed_acte2e_eval.sh` compares BF16, MixNLQ, and Linear/RTN activation pseudo-quantization:

```bash
bash scripts/speed_acte2e_eval.sh \
  --model-path /path/to/Llama-3.1-8B \
  --weight-checkpoint outputs/mixnlq_w4.pth \
  --device 0 \
  --batch-size 1 \
  --prompt-length 2048 \
  --new-tokens 128 \
  --activation-bits 4 \
  --output-json outputs/speed_acte2e.json
```

The speed scripts use random token IDs by default. Set `--input-mode wikitext2` to use WikiText-2 text instead. Use `--warmup-runs` and `--benchmark-runs` to adjust the warm-up and measurement iterations. Both scripts run on a single GPU using `--device`.
