"""Evaluate a BF16 model or a pseudo-quantized checkpoint from ``quant_sim.py``."""

from __future__ import annotations

import argparse
import json
import os
import random
import warnings
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.eval_utils import PPL_TASKS, evaluate_ppl_tasks

os.environ["HF_ALLOW_CODE_EVAL"] = "0"
DEFAULT_TASKS = (
    # "gsm8k",
    # "humaneval",  # HF_ALLOW_CODE_EVAL=1
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
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--limit must be positive")
    if parsed.is_integer() and parsed >= 1:
        return int(parsed)
    if parsed > 1:
        raise argparse.ArgumentTypeError(
            "fractional --limit must be in the interval (0, 1]"
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
        raise argparse.ArgumentTypeError(
            "--tasks must include at least one task name"
        )
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate BF16 or quant_sim.py pseudo-quantized models with lm-eval",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Hugging Face BF16 base model directory",
    )
    parser.add_argument(
        "--sim-checkpoint",
        default=None,
        help="Optional structured checkpoint from quant_sim.py; omit for BF16 evaluation",
    )
    parser.add_argument(
        "--devices",
        default="0",
        help=(
            "Comma-separated physical CUDA ids. Use `0` for one GPU or "
            "`0,1,2,3` for multiple GPUs."
        ),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASKS),
        help=(
            "lm-evaluation-harness task sequence. Supports either "
            "`--tasks gsm8k humaneval` or `--tasks gsm8k,humaneval`."
        ),
    )
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument(
        "--ppl-tasks",
        nargs="*",
        choices=PPL_TASKS,
        default=list(PPL_TASKS),
        help="PPL datasets to evaluate (default: wikitext2 c4); pass at end with no values to skip",
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
        help=(
            "lm-eval batch size. Default 1 keeps channel-wise activation "
            "pseudo quantization independent of batch packing"
        ),
    )
    parser.add_argument(
        "--limit",
        type=_parse_limit,
        default=None,
        help="Optional example count (1000) or dataset fraction (0.1)",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="sdpa",
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


def _visible_devices(args: argparse.Namespace) -> list[str]:
    raw = args.devices
    devices = [item.strip() for item in raw.split(",")]
    if not all(devices):
        raise ValueError("--devices must contain at least one CUDA device id")
    if len(devices) != len(set(devices)):
        raise ValueError(f"duplicate CUDA device ids in {raw!r}")
    return devices


def main() -> None:
    args = parse_args()
    if args.num_fewshot < 0:
        raise ValueError("--num-fewshot cannot be negative")
    if args.c4_ppl_max_samples <= 0:
        raise ValueError("--c4-ppl-max-samples must be positive")

    # Configure physical-to-logical GPU remapping before any CUDA API call.
    physical_devices = _visible_devices(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(physical_devices)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("evalsim_lm_eval.py requires a CUDA GPU")
    visible_device_count = torch.cuda.device_count()
    if visible_device_count != len(physical_devices):
        raise RuntimeError(
            f"Requested {len(physical_devices)} GPUs ({physical_devices}), "
            f"but PyTorch sees {visible_device_count} visible CUDA devices"
        )
    multi_gpu = visible_device_count > 1

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

    device_map = "balanced" if multi_gpu else {"": torch.device("cuda:0")}
    model_format = "simulated" if args.sim_checkpoint is not None else "bf16"
    print(
        f"Loading {model_format} model:\n"
        f"  base model: {args.model_path}\n"
        f"  checkpoint: {args.sim_checkpoint}\n"
        "  CUDA mapping: "
        + ", ".join(
            f"physical {physical} -> cuda:{logical}"
            for logical, physical in enumerate(physical_devices)
        )
    )
    sim_config = None
    if args.sim_checkpoint is None:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map=device_map,
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation=args.attn_implementation,
        )
    else:
        from quant_sim import SimulatedQuantLinear, load_simulated_model

        model = load_simulated_model(
            model_path=args.model_path,
            checkpoint_path=args.sim_checkpoint,
            device_map=device_map,
            attn_implementation=args.attn_implementation,
        )
        sim_config = model.quant_sim_config
        simulated_layers = sum(
            isinstance(module, SimulatedQuantLinear)
            for module in model.modules()
        )
        activation_mode = sim_config["activation_quant_mode"]
        if activation_mode != "none" and simulated_layers == 0:
            raise RuntimeError(
                "Checkpoint requests activation pseudo quantization but no "
                "SimulatedQuantLinear modules were restored"
            )
        if (
            activation_mode != "none"
            and sim_config["activation_granularity"] == "channel"
            and args.batch_size != 1
        ):
            warnings.warn(
                "Channel-wise activation pseudo quantization computes its runtime "
                "absmax over the token rows in each invocation. Using batch_size "
                f"{args.batch_size!r} can therefore make metrics depend on lm-eval "
                "batch packing; batch_size=1 is recommended for reproducibility.",
                RuntimeWarning,
                stacklevel=2,
            )

        print(
            "Verified simulated configuration: "
            f"W={sim_config['weight_quant_mode']}"
            f"{sim_config['weight_bits']} "
            f"(group_size={sim_config['weight_group_size']}), "
            f"weight activation calibration={sim_config.get('usecalib', False)}, "
            f"token activation calibration="
            f"{sim_config.get('calibrate_token_activations', False)}, "
            f"dynamic token iterations="
            f"{sim_config.get('dynamic_token_iterations', 2)}, "
            f"A={activation_mode}{sim_config['activation_bits']} "
            f"(granularity={sim_config['activation_granularity']}), "
            f"activation wrappers={simulated_layers}."
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
    print(f"Checkpoint: {args.sim_checkpoint}")
    if sim_config is not None:
        print(
            f"W={sim_config['weight_quant_mode']}{sim_config['weight_bits']}, "
            f"A={activation_mode}{sim_config['activation_bits']}"
        )
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
            "checkpoint": args.sim_checkpoint,
            "sim_checkpoint": args.sim_checkpoint,
            "quant_sim_config": sim_config,
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
