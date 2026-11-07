"""Export packed 3/4-bit checkpoints.

Weight-only MixNLQ, affine linear/RTN, and group-LUT methods are supported
at 3 or 4 bits.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from quant_sim import (
    ActivationSampleCollector,
    _calibration_loader,
    _capture_first_layer_inputs,
    _extract_layer_hidden_states,
    _linear_children,
    _supported_layer_kwargs,
    _tree_to_device,
)
from utils.triton_quant_linear import TritonInt4Linear
from utils.calibration_data import DEFAULT_LOCAL_C4_PATH, DEFAULT_LOCAL_WIKI2_PATH


CHECKPOINT_FORMAT = "packed-weight-only-v4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a packed 3/4-bit weight-only checkpoint",
        allow_abbrev=False,
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--model-checkpoint",
        default=None,
        help="Optional trained full-model state_dict (.pth) loaded before quantization",
    )
    parser.add_argument("--output", required=True, help="Output .pth file")
    parser.add_argument(
        "--quant-mode",
        choices=(
            "mixnlq", "mixnlq3",
            "linear", "linear3",
            "lut", "lut3",
        ),
        required=True,
        help="LUT4-group uses lut; LUT3-group uses lut3",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help=(
            "Positive multiple of 32, or -1 for per-output-channel "
            "quantization (one K-wide group per channel)"
        ),
    )
    parser.add_argument("--mixnlq-iterations", type=int, default=100)
    parser.add_argument(
        "--usecalib",
        action="store_true",
        help=(
            "Use quant.py-compatible activation-aware MixNLQ weight "
            "calibration"
        ),
    )
    parser.add_argument(
        "--calibration-data",
        choices=(
            "wiki2",
            "c4",
            "wiki2-c4",
            "wiki2-c4-arcc",
            "wiki2-c4-zeros",
        ),
        default="wiki2",
    )
    parser.add_argument("--calibration-samples", type=int, default=8)
    parser.add_argument(
        "--use-local", "--use-local-dataset", dest="use_local", action="store_true",
        help="Load calibration WikiText-2/C4 from local dataset directories",
    )
    parser.add_argument("--local-wiki2-path", default=DEFAULT_LOCAL_WIKI2_PATH)
    parser.add_argument("--local-c4-path", default=DEFAULT_LOCAL_C4_PATH)
    parser.add_argument("--calibration-seqlen", type=int, default=2048)
    parser.add_argument(
        "--calibration-tokens-per-linear",
        type=int,
        default=2048,
        help=(
            "Retained for command compatibility; activation-aware weight "
            "Hessians are accumulated online from all calibration sequences"
        ),
    )
    parser.add_argument(
        "--devices",
        default="0",
        help=(
            "Comma-separated physical GPU ids, for example 0,1,2,3. The model "
            "is balanced across them and one quantization worker runs per GPU"
        ),
    )
    args = parser.parse_args()
    if args.quant_mode == "lut":
        # Preserve the packed checkpoint mode understood by existing loaders.
        args.quant_mode = "nf4_lut"
    return args


def _parse_visible_devices(args: argparse.Namespace) -> list[str]:
    raw = args.devices
    devices = [item.strip() for item in raw.split(",") if item.strip()]
    if not devices:
        raise ValueError("at least one CUDA device must be selected")
    if len(set(devices)) != len(devices):
        raise ValueError(f"duplicate CUDA device ids in {raw!r}")
    return devices


@torch.inference_mode()
def _pack_layers_on_device(
    device: torch.device,
    layers: list[tuple[str, nn.Linear]],
    quant_mode: str,
    group_size: int,
    mixnlq_iterations: int,
    progress: tqdm,
) -> dict[str, dict]:
    packed_layers = {}
    # CUDA's current device is thread-local. Set it explicitly so allocations
    # made inside quantization helpers stay on this worker's logical GPU.
    with torch.cuda.device(device):
        for name, layer in layers:
            packed = TritonInt4Linear.from_float(
                layer,
                quant_mode=quant_mode,
                group_size=group_size,
                mixnlq_iterations=(
                    mixnlq_iterations
                    if quant_mode in {"mixnlq", "mixnlq3"}
                    else 0
                ),
            )
            # Move the exported entry to CPU immediately. This prevents packed
            # tensors from every layer accumulating on the worker GPU.
            packed_layers[name] = packed.checkpoint_entry()
            del packed
            progress.update(1)
    return packed_layers


@torch.inference_mode()
def _pack_activation_calibrated_mixnlq(
    model: nn.Module,
    tokenizer,
    args: argparse.Namespace,
    progress: tqdm,
) -> dict[str, dict]:
    """Pack MixNLQ using the same activation-aware path as quant.py."""
    dataloader = _calibration_loader(
        args.calibration_data,
        args.calibration_samples,
        42,
        args.calibration_seqlen,
        model,
        tokenizer,
        use_local=args.use_local,
        local_wiki2_path=args.local_wiki2_path,
        local_c4_path=args.local_c4_path,
    )
    hidden_samples, saved_kwargs = _capture_first_layer_inputs(
        model,
        dataloader,
        args.calibration_samples,
    )
    global_names = {
        id(module): name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name != "lm_head"
    }
    packed_layers: dict[str, dict] = {}

    for layer in model.model.layers:
        linears = _linear_children(layer)
        collectors = {
            name: ActivationSampleCollector(
                args.calibration_tokens_per_linear,
                collect_samples=False,
                collect_hessian=True,
            )
            for name, _ in linears
        }
        handles = []
        for name, module in linears:
            collector = collectors[name]

            def collect_input(_module, inputs, collector=collector):
                collector.add(inputs[0])

            handles.append(module.register_forward_pre_hook(collect_input))

        layer_device = next(layer.parameters()).device
        next_hidden_samples = []
        try:
            for hidden, kwargs in zip(hidden_samples, saved_kwargs):
                moved_kwargs = _tree_to_device(kwargs, layer_device)
                output = layer(
                    hidden.to(layer_device),
                    **_supported_layer_kwargs(layer, moved_kwargs),
                )
                next_hidden_samples.append(
                    _extract_layer_hidden_states(output).detach()
                )
        finally:
            for handle in handles:
                handle.remove()

        # The outputs above are captured before quantizing this layer, matching
        # quant.py's calibration propagation order.
        for relative_name, module in linears:
            full_name = global_names[id(module)]
            activation_hessian = collectors[relative_name].hessian(
                module.weight.device
            )
            packed = TritonInt4Linear.from_float(
                module,
                quant_mode=args.quant_mode,
                group_size=args.group_size,
                mixnlq_iterations=args.mixnlq_iterations,
                mixnlq_activation_hessian=activation_hessian,
            )
            packed_layers[full_name] = packed.checkpoint_entry()
            del activation_hessian, packed
            progress.update(1)

        hidden_samples = next_hidden_samples
        del collectors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return packed_layers


@torch.inference_mode()
def export_checkpoint(args: argparse.Namespace) -> None:
    if args.group_size != -1 and (
        args.group_size <= 0 or args.group_size % 32
    ):
        raise ValueError(
            "--group-size must be -1 or a positive multiple of 32"
        )
    if args.mixnlq_iterations < 0:
        raise ValueError("--mixnlq-iterations cannot be negative")
    if args.usecalib and args.quant_mode not in {"mixnlq", "mixnlq3"}:
        raise ValueError(
            "--usecalib requires --quant-mode mixnlq or mixnlq3"
        )
    if args.calibration_samples <= 0:
        raise ValueError("--calibration-samples must be positive")
    if args.calibration_seqlen <= 0:
        raise ValueError("--calibration-seqlen must be positive")
    if args.calibration_tokens_per_linear <= 0:
        raise ValueError(
            "--calibration-tokens-per-linear must be positive"
        )

    physical_devices = _parse_visible_devices(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(physical_devices)
    if not torch.cuda.is_available():
        raise RuntimeError("quant_real.py requires CUDA")
    visible_device_count = torch.cuda.device_count()
    if visible_device_count != len(physical_devices):
        raise RuntimeError(
            f"Requested {len(physical_devices)} GPUs ({physical_devices}), "
            f"but PyTorch sees {visible_device_count} visible CUDA devices"
        )
    multi_gpu = visible_device_count > 1
    print(
        "Quantization devices: "
        + ", ".join(
            f"physical {physical} -> cuda:{logical}"
            for logical, physical in enumerate(physical_devices)
        )
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="balanced" if multi_gpu else {"": torch.device("cuda:0")},
        trust_remote_code=False,
    )
    if args.model_checkpoint:
        try:
            trained_state = torch.load(
                args.model_checkpoint,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        except TypeError:
            trained_state = torch.load(
                args.model_checkpoint,
                map_location="cpu",
                weights_only=True,
            )
        incompatible = model.load_state_dict(trained_state, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                "Unexpected trained checkpoint keys: "
                f"{incompatible.unexpected_keys[:8]}"
            )
        del trained_state
    layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name != "lm_head"
    ]
    layers_by_device: dict[torch.device, list[tuple[str, nn.Linear]]] = {}
    for name, layer in layers:
        device = layer.weight.device
        if device.type != "cuda":
            raise RuntimeError(
                f"Layer {name} was placed on {device}; all quantized Linear "
                "layers must fit across the selected CUDA devices"
            )
        layers_by_device.setdefault(device, []).append((name, layer))

    print(
        "Layer distribution: "
        + ", ".join(
            f"{device}={len(device_layers)}"
            for device, device_layers in sorted(
                layers_by_device.items(), key=lambda item: item[0].index
            )
        )
    )
    packed_layers = {}
    mode_name = {"nf4_lut": "LUT4-group", "lut3": "LUT3-group"}.get(
        args.quant_mode, args.quant_mode
    )
    with tqdm(
        total=len(layers),
        desc=(
            f"Packing {mode_name} "
            f"{3 if args.quant_mode in {'mixnlq3', 'linear3', 'lut3'} else 4}-bit"
        ),
    ) as progress:
        if args.usecalib:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_path,
                trust_remote_code=False,
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            model.eval()
            model.config.use_cache = False
            packed_layers.update(
                _pack_activation_calibrated_mixnlq(
                    model,
                    tokenizer,
                    args,
                    progress,
                )
            )
        elif len(layers_by_device) == 1:
            device, device_layers = next(iter(layers_by_device.items()))
            packed_layers.update(
                _pack_layers_on_device(
                    device,
                    device_layers,
                    args.quant_mode,
                    args.group_size,
                    args.mixnlq_iterations,
                    progress,
                )
            )
        else:
            # One worker per GPU avoids same-device kernel contention while
            # allowing independent layer quantization to run concurrently.
            with ThreadPoolExecutor(
                max_workers=len(layers_by_device),
                thread_name_prefix="quant-gpu",
            ) as executor:
                futures = [
                    executor.submit(
                        _pack_layers_on_device,
                        device,
                        device_layers,
                        args.quant_mode,
                        args.group_size,
                        args.mixnlq_iterations,
                        progress,
                    )
                    for device, device_layers in layers_by_device.items()
                ]
                for future in as_completed(futures):
                    packed_layers.update(future.result())

    checkpoint = {
        "format": CHECKPOINT_FORMAT,
        "quant_mode": args.quant_mode,
        "bits": (
            3
            if args.quant_mode in {"mixnlq3", "linear3", "lut3"}
            else 4
        ),
        "group_size": args.group_size,
        "layout": (
            "int3-n256-k128-segment-v2"
            if args.quant_mode in {"mixnlq3", "linear3", "lut3"}
            else (
                "n64-kp8"
                if args.quant_mode in {"mixnlq", "linear", "nf4_lut"}
                else "k-major"
            )
        ),
        "source_model": args.model_path,
        "source_checkpoint": args.model_checkpoint,
        "usecalib": args.usecalib,
        "calibration": (
            {
                "data": args.calibration_data,
                "samples": args.calibration_samples,
                "seqlen": args.calibration_seqlen,
                "hessian_tokens": (
                    args.calibration_samples * args.calibration_seqlen
                ),
                "seed": 42,
                "use_local": args.use_local,
                "local_paths": (
                    {"wiki2": args.local_wiki2_path, "c4": args.local_c4_path}
                    if args.use_local else None
                ),
            }
            if args.usecalib
            else None
        ),
        "layers": packed_layers,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(f"Saved {len(packed_layers)} packed layers to {output}")


def main() -> None:
    export_checkpoint(parse_args())


if __name__ == "__main__":
    main()

