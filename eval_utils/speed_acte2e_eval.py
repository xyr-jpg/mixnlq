"""End-to-end benchmark for Triton activation pseudo quantization.

The benchmark compares one shared model-weight state under three activation
paths:

* BF16 activations without QDQ;
* dynamic per-token MixNLQ activation QDQ;
* dynamic per-token affine Linear/RTN activation QDQ.

Both quantized variants quantize and immediately dequantize activations with
Triton before the ordinary BF16 Linear operation. They are activation
pseudo-quantization benchmarks, not packed W4A4 integer GEMM benchmarks.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import inspect
import json
import os
import statistics
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

from quant_sim import (
    load_existing_weights_,
    quantize_without_activation_calibration,
)
from utils.triton_mixnlq_activation import (
    dynamic_mixnlq_token_qdq,
    dynamic_rtn_qdq,
)


@dataclass
class BenchmarkResult:
    name: str
    prefill_ms: float
    prefill_tokens_per_second: float
    decode_ms_per_token: float
    decode_tokens_per_second: float
    e2e_ms: float
    generated_tokens_per_second: float
    speedup_vs_bf16: float = 1.0
    speedup_vs_linear: float = 1.0
    peak_memory_gib: float = 0.0
    model_storage_gib: float = 0.0
    loaded_cuda_gib: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare BF16, Triton MixNLQ and Triton Linear/RTN activation "
            "pseudo quantization end to end"
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--weight-checkpoint",
        default=None,
        help=(
            "Optional quant_sim.py checkpoint. Only its pseudo-quantized "
            "weights are loaded; every benchmark variant uses the same weights"
        ),
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Single physical CUDA id; benchmarking uses logical cuda:0",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-length", type=int, default=2048)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--activation-bits", type=int, default=4)
    parser.add_argument(
        "--dynamic-token-iterations",
        type=int,
        default=2,
        help="Least-squares iterations used by dynamic MixNLQ per-token QDQ",
    )
    parser.add_argument(
        "--input-mode",
        choices=("random", "wikitext2", "wiki2"),
        default="random",
    )
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--benchmark-runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "eager", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help=(
            "Compile prefill/decode wrappers. Disabled by default because the "
            "raw Triton QDQ calls may cause graph breaks on older PyTorch"
        ),
    )
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def patch_transformers_output_capturing() -> None:
    """Work around Transformers versions missing a module-level torch import."""
    try:
        output_capturing = importlib.import_module(
            "transformers.utils.output_capturing"
        )
    except ImportError:
        return
    if "torch" not in output_capturing.__dict__:
        output_capturing.torch = torch
        warnings.warn(
            "Injected torch into transformers.utils.output_capturing.",
            RuntimeWarning,
            stacklevel=2,
        )


def model_storage_gib(model: nn.Module) -> float:
    seen: set[tuple[str, int]] = set()
    total_bytes = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor.numel() == 0:
            continue
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        if key in seen:
            continue
        seen.add(key)
        total_bytes += storage.nbytes()
    return total_bytes / (1024**3)


def load_wikitext2_input(
    model_path: str,
    batch_size: int,
    prompt_length: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=False,
    )
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    dataset_name = "Salesforce/wikitext/wikitext-2-raw-v1"

    encoded = tokenizer(
        "\n\n".join(dataset["text"]),
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids
    token_count = encoded.shape[1]
    if token_count < prompt_length:
        raise RuntimeError(
            f"{dataset_name} has {token_count} tokens, fewer than "
            f"--prompt-length={prompt_length}"
        )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    starts = torch.randint(
        0,
        token_count - prompt_length + 1,
        (batch_size,),
        generator=generator,
    ).tolist()
    input_ids = torch.cat(
        [encoded[:, start : start + prompt_length] for start in starts],
        dim=0,
    )
    print(
        f"Loaded {dataset_name} (test), token_count={token_count}, "
        f"window_starts={starts}."
    )
    return input_ids.to(device=device, non_blocking=True)


def _load_model(
    args: argparse.Namespace,
    device: torch.device,
    activation_mode: str,
) -> tuple[nn.Module, float]:
    allocated_before = torch.cuda.memory_allocated(device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map={"": device},
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
    )
    if args.weight_checkpoint is not None:
        count, _ = load_existing_weights_(model, args.weight_checkpoint)
        print(
            f"Loaded {count} shared weight tensors from "
            f"{args.weight_checkpoint}."
        )

    if activation_mode != "none":
        replaced = quantize_without_activation_calibration(
            model=model,
            weight_mode="none",
            weight_bits=4,
            weight_group_size=0,
            weight_iterations=0,
            activation_mode=activation_mode,
            activation_bits=args.activation_bits,
            activation_granularity="token",
            activation_iterations=0,
            dynamic_token_iterations=args.dynamic_token_iterations,
        )
        print(
            f"Wrapped {replaced} Linear layers with Triton "
            f"{activation_mode} A{args.activation_bits} QDQ."
        )

    model.eval()
    model.config.use_cache = True
    loaded_gib = (
        torch.cuda.memory_allocated(device) - allocated_before
    ) / (1024**3)
    return model, loaded_gib


class StaticRunner:
    def __init__(
        self,
        model: nn.Module,
        batch_size: int,
        prompt_length: int,
        new_tokens: int,
        device: torch.device,
        compile_enabled: bool,
    ) -> None:
        self.model = model
        max_length = prompt_length + new_tokens
        try:
            self.cache = StaticCache(
                config=model.config,
                max_cache_len=max_length,
            )
        except TypeError:
            self.cache = StaticCache(
                config=model.config,
                max_batch_size=batch_size,
                max_cache_len=max_length,
                device=device,
                dtype=torch.bfloat16,
            )

        self.attention_mask = torch.ones(
            (batch_size, max_length),
            dtype=torch.long,
            device=device,
        )
        self.prefill_position = torch.arange(
            prompt_length,
            dtype=torch.long,
            device=device,
        )
        self.decode_position = torch.empty(
            1,
            dtype=torch.long,
            device=device,
        )
        self.token_buffer = torch.empty(
            (batch_size, 1),
            dtype=torch.long,
            device=device,
        )
        self.logits_to_keep_supported = (
            "logits_to_keep" in inspect.signature(model.forward).parameters
        )
        mark_static_address = getattr(
            getattr(torch, "_dynamo", None),
            "mark_static_address",
            None,
        )
        if mark_static_address is not None:
            for tensor in (
                self.attention_mask,
                self.prefill_position,
                self.decode_position,
                self.token_buffer,
            ):
                mark_static_address(tensor)

        def forward_step(input_ids, attention_mask, cache_position):
            kwargs: dict[str, Any] = {}
            if self.logits_to_keep_supported:
                kwargs["logits_to_keep"] = 1
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=self.cache,
                cache_position=cache_position,
                use_cache=True,
                **kwargs,
            )
            return output.logits[:, -1:].argmax(dim=-1)

        if compile_enabled:
            self.prefill_forward = torch.compile(
                forward_step,
                mode="max-autotune",
                fullgraph=False,
            )
            self.decode_step = torch.compile(
                forward_step,
                mode="reduce-overhead",
                fullgraph=False,
            )
        else:
            self.prefill_forward = forward_step
            self.decode_step = forward_step

    @torch.inference_mode()
    def timed_run(
        self,
        input_ids: torch.Tensor,
        new_tokens: int,
    ) -> tuple[float, float]:
        self.cache.reset()
        start = torch.cuda.Event(enable_timing=True)
        prefill_done = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        next_token = self.prefill_forward(
            input_ids,
            self.attention_mask,
            self.prefill_position,
        )
        prefill_done.record()
        for index in range(new_tokens - 1):
            self.token_buffer.copy_(next_token)
            self.decode_position.fill_(input_ids.shape[1] + index)
            next_token = self.decode_step(
                self.token_buffer,
                self.attention_mask,
                self.decode_position,
            )
        end.record()
        end.synchronize()
        return start.elapsed_time(prefill_done), start.elapsed_time(end)


def benchmark_model(
    name: str,
    model: nn.Module,
    input_ids: torch.Tensor,
    loaded_cuda_gib: float,
    new_tokens: int,
    warmup_runs: int,
    benchmark_runs: int,
    compile_enabled: bool,
) -> BenchmarkResult:
    runner = StaticRunner(
        model=model,
        batch_size=input_ids.shape[0],
        prompt_length=input_ids.shape[1],
        new_tokens=new_tokens,
        device=input_ids.device,
        compile_enabled=compile_enabled,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(input_ids.device)

    for _ in range(warmup_runs):
        runner.timed_run(input_ids, new_tokens)

    prefill_samples: list[float] = []
    decode_samples: list[float] = []
    e2e_samples: list[float] = []
    for _ in range(benchmark_runs):
        prefill_ms, e2e_ms = runner.timed_run(input_ids, new_tokens)
        prefill_samples.append(prefill_ms)
        decode_samples.append(e2e_ms - prefill_ms)
        e2e_samples.append(e2e_ms)

    prefill_ms = statistics.median(prefill_samples)
    decode_ms = statistics.median(decode_samples)
    e2e_ms = statistics.median(e2e_samples)
    batch_size, prompt_length = input_ids.shape
    decode_steps = new_tokens - 1
    return BenchmarkResult(
        name=name,
        prefill_ms=prefill_ms,
        prefill_tokens_per_second=(
            batch_size * prompt_length * 1000.0 / prefill_ms
        ),
        decode_ms_per_token=decode_ms / decode_steps,
        decode_tokens_per_second=(
            batch_size * decode_steps * 1000.0 / decode_ms
        ),
        e2e_ms=e2e_ms,
        generated_tokens_per_second=(
            batch_size * new_tokens * 1000.0 / e2e_ms
        ),
        peak_memory_gib=(
            torch.cuda.max_memory_allocated(input_ids.device) / (1024**3)
        ),
        model_storage_gib=model_storage_gib(model),
        loaded_cuda_gib=loaded_cuda_gib,
    )


def print_results(results: list[BenchmarkResult]) -> None:
    headers = (
        "model",
        "storage GiB",
        "loaded GiB",
        "prefill ms",
        "prefill tok/s",
        "decode ms/tok",
        "decode tok/s",
        "E2E ms",
        "gen tok/s",
        "vs BF16",
        "vs Linear",
        "peak GiB",
    )
    rows = [
        (
            result.name,
            f"{result.model_storage_gib:.2f}",
            f"{result.loaded_cuda_gib:.2f}",
            f"{result.prefill_ms:.2f}",
            f"{result.prefill_tokens_per_second:.1f}",
            f"{result.decode_ms_per_token:.3f}",
            f"{result.decode_tokens_per_second:.1f}",
            f"{result.e2e_ms:.2f}",
            f"{result.generated_tokens_per_second:.1f}",
            f"{result.speedup_vs_bf16:.3f}x",
            f"{result.speedup_vs_linear:.3f}x",
            f"{result.peak_memory_gib:.2f}",
        )
        for result in results
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    print("  ".join(value.ljust(widths[i]) for i, value in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.prompt_length <= 0:
        raise ValueError("--prompt-length must be positive")
    if args.new_tokens < 2:
        raise ValueError("--new-tokens must be at least 2")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs cannot be negative")
    if args.benchmark_runs <= 0:
        raise ValueError("--benchmark-runs must be positive")
    if not 2 <= args.activation_bits <= 6:
        raise ValueError(
            "--activation-bits must be in [2, 6] so both Triton paths support it"
        )
    if args.dynamic_token_iterations <= 0:
        raise ValueError("--dynamic-token-iterations must be positive")

    physical_device = args.device.strip()
    if not physical_device or "," in physical_device:
        raise ValueError("--device must select exactly one physical CUDA device")
    os.environ["CUDA_VISIBLE_DEVICES"] = physical_device
    patch_transformers_output_capturing()
    if not torch.cuda.is_available():
        raise RuntimeError("speed_acte2e_eval.py requires CUDA")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "--device must select exactly one GPU for CUDA-event timing"
        )
    if dynamic_mixnlq_token_qdq is None or dynamic_rtn_qdq is None:
        raise RuntimeError("Triton activation QDQ kernels are unavailable")

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    input_ids = (
        load_wikitext2_input(
            model_path=args.model_path,
            batch_size=args.batch_size,
            prompt_length=args.prompt_length,
            seed=args.seed,
            device=device,
        )
        if args.input_mode in {"wikitext2", "wiki2"}
        else None
    )

    variants = (
        ("bf16", "none"),
        (f"mixnlq-a{args.activation_bits}", "mixnlq"),
        (f"linear-rtn-a{args.activation_bits}", "rtn"),
    )
    results: list[BenchmarkResult] = []
    for name, activation_mode in variants:
        dynamo_reset = getattr(getattr(torch, "_dynamo", None), "reset", None)
        if dynamo_reset is not None:
            dynamo_reset()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        print(f"\nLoading {name} ...")
        model, loaded_cuda_gib = _load_model(
            args,
            device,
            activation_mode,
        )
        if input_ids is None:
            input_ids = torch.randint(
                low=0,
                high=model.config.vocab_size,
                size=(args.batch_size, args.prompt_length),
                generator=generator,
                dtype=torch.long,
            ).to(device)
        print(f"Benchmarking {name} ...")
        result = benchmark_model(
            name=name,
            model=model,
            input_ids=input_ids,
            loaded_cuda_gib=loaded_cuda_gib,
            new_tokens=args.new_tokens,
            warmup_runs=args.warmup_runs,
            benchmark_runs=args.benchmark_runs,
            compile_enabled=args.compile,
        )
        results.append(result)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    bf16_ms = results[0].e2e_ms
    linear_ms = results[-1].e2e_ms
    for result in results:
        result.speedup_vs_bf16 = bf16_ms / result.e2e_ms
        result.speedup_vs_linear = linear_ms / result.e2e_ms

    print("\nMedian end-to-end activation-QDQ results")
    print_results(results)
    print(
        "\nDecode uses median(e2e_i - prefill_i) over benchmark runs. "
        "MixNLQ and Linear/RTN both use dynamic per-token Triton QDQ, share "
        "QKV and Gate/Up QDQ results, and retain BF16 Linear matmul."
    )

    if args.output_json is not None:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(
                [asdict(result) for result in results],
                handle,
                indent=2,
            )
        print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
