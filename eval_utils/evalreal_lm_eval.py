"""Evaluate a BF16 model or a real packed MixNLQ checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.eval_utils import PPL_TASKS, evaluate_ppl_tasks

os.environ["HF_ALLOW_CODE_EVAL"] = "0"
DEFAULT_TASKS = (
    "arc_challenge",
    "arc_easy",
    "boolq",
    "piqa",
    "social_iqa",
    "hellaswag",
    "openbookqa",
    "winogrande",
)


def _parse_limit(value: str) -> int | float:
    """Accept either an example count (1000) or a fraction (0.1)."""
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--limit must be positive")
    if parsed.is_integer() and parsed >= 1:
        return int(parsed)
    if parsed > 1:
        raise argparse.ArgumentTypeError(
            "a fractional --limit must be in the interval (0, 1]"
        )
    return parsed


def _parse_batch_size(value: str) -> int | str:
    normalized = value.strip().lower()
    if normalized == "auto" or normalized.startswith("auto:"):
        return normalized
    try:
        batch_size = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--batch-size must be a positive integer, auto, or auto:N"
        ) from exc
    if batch_size <= 0:
        raise argparse.ArgumentTypeError("--batch-size must be positive")
    return batch_size


def _normalize_tasks(values: list[str]) -> list[str]:
    tasks = [
        task.strip()
        for value in values
        for task in value.split(",")
        if task.strip()
    ]
    if not tasks:
        raise argparse.ArgumentTypeError("--tasks must include at least one task name")
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate BF16 or real packed MixNLQ W3A16/W4A16 models",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Hugging Face base model directory",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional packed checkpoint exported by quant_real.py; omit for BF16",
    )
    parser.add_argument(
        "--quant-mode",
        choices=(
            "mixnlq", "mixnlq3",
            "linear", "linear3",
            "nf4", "nf4_lut", "lut3",
        ),
        default="mixnlq",
        help="mixnlq is W4A16; mixnlq3 is W3A16",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Physical CUDA device id (single GPU only; default: 0)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASKS),
        help="lm-evaluation-harness tasks, space- or comma-separated",
    )
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument(
        "--ppl-tasks",
        nargs="*",
        choices=PPL_TASKS,
        default=list(PPL_TASKS),
        help="PPL datasets (default: wikitext2 c4); pass at end with no values to skip",
    )
    parser.add_argument(
        "--c4-ppl-max-samples",
        type=int,
        default=1000,
        help="Number of C4 validation records to read for PPL (default: 1000)",
    )
    parser.add_argument(
        "--batch-size",
        type=_parse_batch_size,
        default=1,
        help="lm-eval batch size: positive integer, auto, or auto:N",
    )
    parser.add_argument(
        "--limit",
        type=_parse_limit,
        default=None,
        help="Optional example count (for example 1000) or fraction (0.1)",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument(
        "--fuse-projections",
        action="store_true",
        help="Fuse Llama QKV and Gate/Up projections after loading",
    )
    parser.add_argument(
        "--spinquant-r4",
        action="store_true",
        help=(
            "Insert the online SpinQuant R4 transform; use only when the "
            "checkpoint weights already contain the offline rotations"
        ),
    )
    parser.add_argument(
        "--decode-backend",
        choices=("auto", "cuda", "triton"),
        default="auto",
        help="M=1 packed INT4 Decode backend",
    )
    parser.add_argument(
        "--cuda-kernel",
        choices=("auto", "scalar", "vec2"),
        default="auto",
        help="Native CUDA MixNLQ Decode kernel",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path for complete lm-eval results",
    )
    args = parser.parse_args()
    try:
        args.tasks = _normalize_tasks(args.tasks)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    return args


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def main() -> None:
    args = parse_args()
    if args.num_fewshot < 0:
        raise ValueError("--num-fewshot cannot be negative")
    if args.c4_ppl_max_samples <= 0:
        raise ValueError("--c4-ppl-max-samples must be positive")
    if args.checkpoint is None and (args.fuse_projections or args.spinquant_r4):
        raise ValueError("--fuse-projections and --spinquant-r4 require --checkpoint")
    if args.device < 0:
        raise ValueError("--device must be a non-negative CUDA device id")

    # Set visibility before the first CUDA API call.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    os.environ["INT4_DECODE_BACKEND"] = args.decode_backend
    os.environ["MIXNLQ_CUDA_KERNEL"] = args.cuda_kernel

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("evalreal_lm_eval.py requires a CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Requested physical CUDA device {args.device}, but PyTorch does not "
            "see exactly one visible CUDA device"
        )
    device = torch.device("cuda:0")

    # Import after backend environment variables have been configured.
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.utils import make_table
    print(f"Loading tokenizer from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_format = "packed" if args.checkpoint is not None else "bf16"
    device_map = {"": device}
    print(
        f"Loading {model_format} model:\n"
        f"  base model: {args.model_path}\n"
        f"  checkpoint: {args.checkpoint}\n"
        f"  CUDA mapping: physical {args.device} -> cuda:0"
    )
    if args.checkpoint is None:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map=device_map,
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation=args.attn_implementation,
        )
    else:
        from speed_e2e_evaluate import load_model
        from utils.triton_quant_linear import TritonInt4Linear

        model = load_model(
            model_path=args.model_path,
            device=device,
            checkpoint_path=args.checkpoint,
            quant_mode=args.quant_mode,
            fuse_projections=args.fuse_projections,
            attn_implementation=args.attn_implementation,
            spinquant_r4=args.spinquant_r4,
            device_map=device_map,
        )
        packed_layers = sum(
            isinstance(module, TritonInt4Linear)
            for module in model.modules()
        )
        if packed_layers == 0:
            raise RuntimeError(
                "No packed TritonInt4Linear layers were installed; refusing to "
                "run what would otherwise be a BF16 evaluation"
            )
        packed_devices = sorted(
            {
                str(module.packed_weight.device)
                for module in model.modules()
                if isinstance(module, TritonInt4Linear)
            }
        )
        if any(not name.startswith("cuda:") for name in packed_devices):
            raise RuntimeError(
                f"Packed layers must be on CUDA devices, found {packed_devices}"
            )
        bits = 3 if args.quant_mode in {"mixnlq3", "linear3", "lut3"} else 4
        print(
            f"Verified {packed_layers} real packed {bits}-bit Linear modules "
            f"on {', '.join(packed_devices)}."
        )

    evaluator_model = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        trust_remote_code=False,
    )
    print(
        f"Starting lm-eval: tasks={args.tasks}, "
        f"num_fewshot={args.num_fewshot}, batch_size={args.batch_size}, "
        f"limit={args.limit}"
    )
    results = lm_eval.simple_evaluate(
        model=evaluator_model,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        confirm_run_unsafe_code=True,
    )

    print("\n" + "=" * 72)
    print(f"{model_format} lm-eval results")
    print(f"Base model: {args.model_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print("=" * 72)
    print(make_table(results))

    ppl_results = evaluate_ppl_tasks(
        model, tokenizer, args.ppl_tasks, args.c4_ppl_max_samples
    )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_path": args.model_path,
            "model_format": model_format,
            "checkpoint": args.checkpoint,
            "mixnlq_checkpoint": args.checkpoint,
            "quant_mode": args.quant_mode if args.checkpoint else None,
            "lm_eval": results,
            "ppl": ppl_results,
        }
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            )
        print(f"Saved complete results to {output_path}")


if __name__ == "__main__":
    main()
