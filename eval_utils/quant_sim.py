"""
This script is used to test the effect of weight/activation pseudo-quantization:

* Weight pseudo quantization writes dequantized values back to BF16 weights.
* Activation pseudo quantization quantizes and immediately dequantizes inputs
  before ``torch.nn.functional.linear``.
"""

from __future__ import annotations

import argparse
import inspect
import os
import random
import time
import weakref
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.calibration_data import (
    DEFAULT_LOCAL_C4_PATH,
    DEFAULT_LOCAL_WIKI2_PATH,
    get_c4,
    get_mixed_calibration_data,
    get_mixed_wiki2_c4_arcc_data,
    get_mixed_wiki2_c4_data,
    get_wikitext2,
)
from utils.eval_utils import (
    calculate_perplexity,
    evaluate_ppl_tasks,
    load_wikitext_data,
)
from utils.nvfp4_sim import quantize_matrix_mxfp4, quantize_matrix_nvfp4
from utils.quant_cmp_utils import (
    HQQ_quant,
    NF4_quant,
    PoT_quant,
    RTN_had_quant,
    sinq_quant,
)
from utils.quant_utils import (
    activation_aware_mixnlq_quant,
    fasterquant,
    pseudo_RTN_quant,
    pseudo_mix_quant,
    pseudo_mix_quant_bias,
    pseudo_mix_quant_apply,
)

try:
    from utils.triton_mixnlq_activation import dynamic_mixnlq_token_qdq
except ImportError:
    dynamic_mixnlq_token_qdq = None


WeightMode = Literal[
    "none",
    "mixnlq",
    "gptq",
    "rtn",
    "rtn_had",
    "nf4",
    "hqq",
    "sinq",
    "pot",
    "nvfp4",
    "mxfp4",
]
ActivationMode = Literal["none", "rtn", "mixnlq"]

SIM_CHECKPOINT_FORMAT = "pseudo-quantized-model-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply weight and/or activation pseudo quantization",
        allow_abbrev=False,
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--devices",
        default="0",
        help="Comma-separated physical CUDA ids, for example 0,1,2,3",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional pseudo-quantized checkpoint path",
    )
    parser.add_argument(
        "--weight-checkpoint",
        default=None,
        help=(
            "Load only model parameters from an existing quant_sim checkpoint, "
            "skip weight quantization, and use dynamic per-token MixNLQ "
            "activation quantization"
        ),
    )
    parser.add_argument(
        "--weight-quant-mode",
        choices=(
            "none",
            "mixnlq",
            "gptq",
            "rtn",
            "rtn_had",
            "nf4",
            "hqq",
            "sinq",
            "pot",
            "nvfp4",
            "mxfp4",
        ),
        default="mixnlq",
    )
    parser.add_argument("--weight-bits", type=int, default=4)
    parser.add_argument(
        "--weight-group-size",
        type=int,
        default=128,
        help="Positive group size; 0 or -1 means one full K group per row",
    )
    parser.add_argument("--weight-iterations", type=int, default=100)
    parser.add_argument(
        "--activation-quant-mode",
        choices=("none", "rtn", "mixnlq"),
        default="none",
    )
    parser.add_argument("--activation-bits", type=int, default=4)
    parser.add_argument(
        "--activation-granularity",
        choices=("token", "channel"),
        default="token",
        help=(
            "Activation quantization granularity. token quantizes each "
            "token over its hidden dimension; channel quantizes each input "
            "channel over tokens."
        ),
    )
    parser.add_argument("--activation-iterations", type=int, default=100)
    parser.add_argument(
        "--dynamic-token-iterations",
        type=int,
        default=2,
        help=(
            "Positive number of joint least-squares iterations in dynamic "
            "per-token MixNLQ"
        ),
    )
    parser.add_argument(
        "--usecalib",
        action="store_true",
        help=(
            "Use activation statistics for activation-aware MixNLQ weight "
            "quantization. This flag does not enable or disable activation "
            "quantization"
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
    parser.add_argument("--calibration-samples", type=int, default=128)
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
        default=262144,
        help=(
            "Maximum rows retained for calibrated activation QDQ scale "
            "fitting. GPTQ and MixNLQ --usecalib always accumulate the "
            "Hessian online from every calibration sequence"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval-wiki2-ppl",
        action="store_true",
        help="Evaluate WikiText-2 perplexity after quantization and before save",
    )
    parser.add_argument(
        "--eval-c4-ppl",
        action="store_true",
        help="Evaluate C4 validation perplexity after quantization and before save",
    )
    parser.add_argument(
        "--wiki2-ppl-max-samples",
        type=int,
        default=None,
        help="Optional maximum number of non-empty WikiText-2 text records",
    )
    parser.add_argument(
        "--wiki2-ppl-max-length",
        type=int,
        default=2048,
        help="Sliding-window context length used for WikiText-2 PPL",
    )
    parser.add_argument(
        "--wiki2-ppl-stride",
        type=int,
        default=512,
        help="Sliding-window stride used for WikiText-2 PPL",
    )
    return parser.parse_args()


def _normalized_group_size(group_size: int, width: int) -> int:
    if group_size in (0, -1):
        return 0
    if group_size <= 0:
        raise ValueError("group size must be positive, 0, or -1")
    if width % group_size:
        raise ValueError(
            f"input width {width} is not divisible by group size {group_size}"
        )
    return group_size


def _load_checkpoint_file(checkpoint_path: str) -> Any:
    try:
        return torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        return torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )


def _checkpoint_state_dict(
    checkpoint: Any,
) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("weight checkpoint must contain a state dictionary")
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, Mapping):
        raise RuntimeError("weight checkpoint state dictionary is invalid")
    return state_dict


@torch.inference_mode()
def load_existing_weights_(
    model: nn.Module,
    checkpoint_path: str,
) -> tuple[int, Mapping[str, Any]]:
    """Load parameters only, deliberately ignoring activation QDQ buffers."""
    checkpoint = _load_checkpoint_file(checkpoint_path)
    state_dict = _checkpoint_state_dict(checkpoint)
    parameters = dict(model.named_parameters())
    missing = [name for name in parameters if name not in state_dict]
    mismatched = [
        (
            name,
            (
                tuple(state_dict[name].shape)
                if isinstance(state_dict[name], torch.Tensor)
                else type(state_dict[name]).__name__
            ),
            tuple(parameter.shape),
        )
        for name, parameter in parameters.items()
        if name in state_dict
        and (
            not isinstance(state_dict[name], torch.Tensor)
            or state_dict[name].shape != parameter.shape
        )
    ]
    if missing or mismatched:
        raise RuntimeError(
            "weight checkpoint does not match the base model: "
            f"missing={missing[:8]}, mismatched={mismatched[:8]}"
        )

    for name, parameter in parameters.items():
        parameter.copy_(
            state_dict[name].to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
        )

    metadata: Mapping[str, Any]
    if "model_state_dict" in checkpoint:
        metadata = {
            key: checkpoint[key]
            for key in (
                "format",
                "base_model",
                "weight_quant_mode",
                "weight_bits",
                "weight_group_size",
                "usecalib",
            )
            if key in checkpoint
        }
    else:
        metadata = {}
    return len(parameters), metadata


def _parse_visible_devices(args: argparse.Namespace) -> list[str]:
    raw = args.devices
    devices = [item.strip() for item in raw.split(",") if item.strip()]
    if not devices:
        raise ValueError("at least one CUDA device must be selected")
    if len(devices) != len(set(devices)):
        raise ValueError(f"duplicate CUDA device ids in {raw!r}")
    return devices


@torch.inference_mode()
def pseudo_quantize_weight_(
    module: nn.Linear,
    mode: WeightMode,
    bits: int,
    group_size: int,
    iterations: int,
    activation_hessian: torch.Tensor | None = None,
) -> None:
    """Pseudo-quantize one Linear weight and store the dequantized BF16 value."""
    if mode == "none":
        return
    group_size = _normalized_group_size(group_size, module.in_features)
    weight = module.weight.detach()

    if mode == "mixnlq":
        if activation_hessian is None:
            dequantized, _ = pseudo_mix_quant(
                wbits=bits,
                W=weight.float(),
                w_group_size=group_size,
                iterations=iterations,
            )
        else:
            dequantized, _, _, _ = activation_aware_mixnlq_quant(
                W=weight.detach().clone(),
                wbits=bits,
                group_size=group_size,
                iterations=iterations,
                hessian=activation_hessian,
            )
    elif mode == "gptq":
        if activation_hessian is None:
            raise ValueError(
                "GPTQ requires a calibration Hessian; use the layer-wise "
                "calibration path"
            )
        dequantized, _, _ = fasterquant(
            W=weight.detach().clone(),
            H=activation_hessian,
            wbits=bits,
            blocksize=128,
            percdamp=0.01,
            groupsize=group_size,
            actorder=True,
            static_groups=True,
            mixquant_params=None,
        )
    elif mode == "rtn":
        _, _, _, dequantized = pseudo_RTN_quant(
            wbits=bits,
            W=weight.float(),
            group_size=group_size,
        )
    elif mode == "rtn_had":
        dequantized = RTN_had_quant(
            wbits=bits,
            W=weight,
            group_size=group_size,
        )
    elif mode == "nf4":
        dequantized = NF4_quant(bits, weight, group_size)
    elif mode == "hqq":
        dequantized = HQQ_quant(bits, weight, group_size)
    elif mode == "sinq":
        dequantized = sinq_quant(
            bits,
            weight.float(),
            group_size,
            niter=50,
        )
    elif mode == "pot":
        dequantized = PoT_quant(
            wbits=bits,
            W=weight,
            group_size=group_size,
        )
    elif mode == "nvfp4":
        if bits != 4:
            raise ValueError("nvfp4 requires --weight-bits 4")
        dequantized, _, _ = quantize_matrix_nvfp4(
            weight,
            block_size=16,
        )
    elif mode == "mxfp4":
        if bits != 4:
            raise ValueError("mxfp4 requires --weight-bits 4")
        dequantized, _ = quantize_matrix_mxfp4(
            weight,
            block_size=32,
        )
    else:
        raise ValueError(f"unsupported weight mode: {mode}")

    if not torch.isfinite(dequantized).all():
        raise RuntimeError(
            f"{mode} produced non-finite values for a "
            f"{module.out_features}x{module.in_features} Linear"
        )
    module.weight.copy_(dequantized.to(dtype=weight.dtype))


class ActivationQDQCache:
    """Short-lived cache shared by projections consuming the same tensor."""

    def __init__(self, expected_uses: int) -> None:
        self.expected_uses = expected_uses
        self._input_ref: weakref.ReferenceType | None = None
        self._output: torch.Tensor | None = None
        self._remaining_uses = 0

    def take(self, x: torch.Tensor) -> torch.Tensor | None:
        if (
            self._input_ref is None
            or self._input_ref() is not x
            or self._output is None
        ):
            self.clear()
            return None
        output = self._output
        self._remaining_uses -= 1
        if self._remaining_uses <= 0:
            self.clear()
        return output

    def store(self, x: torch.Tensor, output: torch.Tensor) -> None:
        self._input_ref = weakref.ref(x)
        self._output = output
        self._remaining_uses = self.expected_uses - 1

    def clear(self) -> None:
        self._input_ref = None
        self._output = None
        self._remaining_uses = 0


def _shared_dynamic_activation_cache(
    name: str,
    caches: dict[str, ActivationQDQCache],
) -> ActivationQDQCache | None:
    parent, _, leaf = name.rpartition(".")
    if parent.endswith("self_attn") and leaf in {
        "q_proj",
        "k_proj",
        "v_proj",
    }:
        return caches.setdefault(
            f"{parent}:qkv",
            ActivationQDQCache(3),
        )
    if parent.endswith("mlp") and leaf in {"gate_proj", "up_proj"}:
        return caches.setdefault(
            f"{parent}:gate_up",
            ActivationQDQCache(2),
        )
    return None


class SimulatedQuantLinear(nn.Module):
    """Linear layer with explicit activation quantize-dequantize simulation."""

    def __init__(
        self,
        source: nn.Linear,
        activation_mode: ActivationMode,
        activation_bits: int,
        activation_granularity: str,
        activation_iterations: int = 100,
        dynamic_token_iterations: int = 2,
        activation_cache: ActivationQDQCache | None = None,
        mixnlq_exp_scale_normalized: torch.Tensor | None = None,
        mixnlq_linear_scale_normalized: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.activation_mode = activation_mode
        self.activation_bits = activation_bits
        self.activation_granularity = activation_granularity
        self.activation_iterations = activation_iterations
        self.dynamic_token_iterations = dynamic_token_iterations
        self.activation_cache = activation_cache
        self.weight = source.weight
        self.bias = source.bias

        empty = torch.empty(
            0,
            dtype=torch.float32,
            device=self.weight.device,
        )
        self.register_buffer(
            "mixnlq_exp_scale_normalized",
            empty
            if mixnlq_exp_scale_normalized is None
            else mixnlq_exp_scale_normalized.to(
                device=self.weight.device,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "mixnlq_linear_scale_normalized",
            empty
            if mixnlq_linear_scale_normalized is None
            else mixnlq_linear_scale_normalized.to(
                device=self.weight.device,
                dtype=torch.float32,
            ),
        )

    def _quantize_rtn(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        rows = x.reshape(-1, shape[-1]).float()
        if self.activation_granularity == "token":
            # Each token is independently quantized over its hidden dimension.
            _, _, _, dequantized = pseudo_RTN_quant(
                self.activation_bits,
                rows,
                group_size=rows.shape[-1],
            )
        else:
            # This matches the intended channel-wise path in quant.py:
            # each input channel is quantized over tokens in this invocation.
            channels = rows.transpose(0, 1).contiguous()
            _, _, _, dequantized = pseudo_RTN_quant(
                self.activation_bits,
                channels,
                group_size=channels.shape[-1],
            )
            dequantized = dequantized.transpose(0, 1)
        return dequantized.reshape(shape).to(dtype=x.dtype)

    def _quantize_mixnlq(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        rows = x.reshape(-1, shape[-1])
        if self.activation_granularity == "token":
            if self.mixnlq_exp_scale_normalized.numel() == 0:
                if self.activation_cache is not None:
                    cached = self.activation_cache.take(x)
                    if cached is not None:
                        return cached
                if (
                    rows.is_cuda
                    and dynamic_mixnlq_token_qdq is not None
                    and 2 <= self.activation_bits <= 6
                ):
                    dequantized = dynamic_mixnlq_token_qdq(
                        rows.contiguous(),
                        bits=self.activation_bits,
                        iterations=self.dynamic_token_iterations,
                    )
                else:
                    if self.activation_bits == 3:
                        dequantized, _ = pseudo_mix_quant_bias(
                            wbits=3,
                            W=rows.float(),
                            w_group_size=0,
                            iterations=self.dynamic_token_iterations,
                        )
                    else:
                        dequantized, _ = pseudo_mix_quant(
                            wbits=self.activation_bits,
                            W=rows.float(),
                            w_group_size=0,
                            iterations=self.dynamic_token_iterations,
                        )
                result = dequantized.reshape(shape).to(dtype=x.dtype)
                if self.activation_cache is not None:
                    self.activation_cache.store(x, result)
                return result
            else:
                rows = rows.float()
                if (
                    self.mixnlq_exp_scale_normalized.numel() != 1
                    or self.mixnlq_linear_scale_normalized.numel() != 1
                ):
                    raise RuntimeError(
                        "invalid calibrated per-token MixNLQ activation scales"
                    )
                token_absmax = rows.abs().amax(dim=1).clamp_min(1.0e-8)
                exp_scale = (
                    self.mixnlq_exp_scale_normalized.float() * token_absmax
                )
                linear_scale = (
                    self.mixnlq_linear_scale_normalized.float() * token_absmax
                )
                dequantized = pseudo_mix_quant_apply(
                    rows,
                    torch.sign(rows),
                    exp_scale,
                    linear_scale,
                    wbits=self.activation_bits,
                )
            return dequantized.reshape(shape).to(dtype=x.dtype)

        rows = rows.float()
        if self.mixnlq_exp_scale_normalized.numel() == 0:
            channels = rows.transpose(0, 1).contiguous()
            dequantized, _ = pseudo_mix_quant(
                wbits=self.activation_bits,
                W=channels,
                w_group_size=0,
                iterations=self.activation_iterations,
            )
            return dequantized.transpose(0, 1).reshape(shape).to(dtype=x.dtype)
        if (
            self.mixnlq_exp_scale_normalized.numel() != self.in_features
            or self.mixnlq_linear_scale_normalized.numel() != self.in_features
        ):
            raise RuntimeError("invalid calibrated MixNLQ activation scales")

        channels = rows.transpose(0, 1).contiguous()
        channel_absmax = channels.abs().amax(dim=1).clamp_min(1.0e-8)
        exp_scale = (
            self.mixnlq_exp_scale_normalized.float() * channel_absmax
        )
        linear_scale = (
            self.mixnlq_linear_scale_normalized.float() * channel_absmax
        )
        dequantized = pseudo_mix_quant_apply(
            channels,
            torch.sign(channels),
            exp_scale,
            linear_scale,
            wbits=self.activation_bits,
        )
        return (
            dequantized.transpose(0, 1)
            .reshape(shape)
            .to(dtype=x.dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_mode == "none":
            effective_x = x
        elif self.activation_mode == "rtn":
            effective_x = self._quantize_rtn(x)
        elif self.activation_mode == "mixnlq":
            effective_x = self._quantize_mixnlq(x)
        else:
            raise RuntimeError(
                f"unsupported activation mode {self.activation_mode!r}"
            )
        return F.linear(
            effective_x.to(self.weight.device),
            self.weight,
            self.bias,
        )


class ActivationSampleCollector:
    """Collect optional rows and quant.py-compatible online Hessian statistics."""

    def __init__(
        self,
        max_tokens: int,
        collect_samples: bool = True,
        collect_hessian: bool = False,
    ) -> None:
        self.max_tokens = max_tokens
        self.collect_samples = collect_samples
        self.collect_hessian = collect_hessian
        self.samples: list[torch.Tensor] = []
        self.collected_tokens = 0
        self.hessian_samples = 0
        self._hessian: torch.Tensor | None = None

    def add(self, x: torch.Tensor) -> None:
        detached = x.detach()
        rows = detached.reshape(-1, detached.shape[-1])

        if self.collect_hessian:
            sequence_count = (
                int(detached.shape[0]) if detached.ndim >= 3 else 1
            )
            updated_count = self.hessian_samples + sequence_count
            if self._hessian is None:
                self._hessian = torch.zeros(
                    (rows.shape[-1], rows.shape[-1]),
                    dtype=torch.float32,
                    device=rows.device,
                )
            else:
                self._hessian.mul_(
                    self.hessian_samples / updated_count
                )
            inputs = rows.transpose(0, 1).float()
            inputs.mul_((2.0 / updated_count) ** 0.5)
            self._hessian.addmm_(inputs, inputs.transpose(0, 1))
            self.hessian_samples = updated_count

        if (
            self.collect_samples
            and self.collected_tokens < self.max_tokens
        ):
            take = min(
                self.max_tokens - self.collected_tokens,
                rows.shape[0],
            )
            if take:
                self.samples.append(rows[:take].to(device="cpu"))
                self.collected_tokens += take

    def tensor(self, device: torch.device) -> torch.Tensor:
        if not self.samples:
            raise RuntimeError("no activation samples were collected")
        return torch.cat(self.samples, dim=0).to(device=device)

    def hessian(self, device: torch.device) -> torch.Tensor:
        if self._hessian is None or self.hessian_samples == 0:
            raise RuntimeError("no activation Hessian was collected")
        return self._hessian.to(device=device)


class _StopAfterFirstLayer(RuntimeError):
    pass


def _tree_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, tuple):
        return tuple(_tree_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_tree_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {
            key: _tree_to_device(item, device)
            for key, item in value.items()
        }
    return value


def _calibration_loader(
    name: str,
    nsamples: int,
    seed: int,
    seqlen: int,
    model: nn.Module,
    tokenizer,
    use_local: bool = False,
    local_wiki2_path: str = DEFAULT_LOCAL_WIKI2_PATH,
    local_c4_path: str = DEFAULT_LOCAL_C4_PATH,
) -> Iterable:
    if name == "wiki2":
        loader, _ = get_wikitext2(
            nsamples, seed, seqlen, model, tokenizer,
            use_local=use_local, local_wiki2_path=local_wiki2_path,
        )
    elif name == "c4":
        loader, _ = get_c4(
            nsamples, seed, seqlen, model, tokenizer,
            use_local=use_local, local_c4_path=local_c4_path,
        )
    elif name == "wiki2-c4":
        loader, _ = get_mixed_wiki2_c4_data(
            nsamples, seed, seqlen, model, tokenizer,
            use_local=use_local, local_wiki2_path=local_wiki2_path,
            local_c4_path=local_c4_path,
        )
    elif name == "wiki2-c4-arcc":
        loader, _ = get_mixed_wiki2_c4_arcc_data(
            nsamples, seed, seqlen, model, tokenizer,
            use_local=use_local, local_wiki2_path=local_wiki2_path,
            local_c4_path=local_c4_path,
        )
    elif name == "wiki2-c4-zeros":
        loader, _ = get_mixed_calibration_data(
            nsamples, seed, seqlen, model, tokenizer,
            use_local=use_local, local_wiki2_path=local_wiki2_path,
            local_c4_path=local_c4_path,
        )
    else:
        raise ValueError(f"unsupported calibration dataset {name!r}")
    return loader


@torch.inference_mode()
def _capture_first_layer_inputs(
    model: nn.Module,
    dataloader: Iterable,
    nsamples: int,
) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise TypeError(
            "calibrated activation MixNLQ currently requires a model with "
            "model.layers (for example Llama)"
        )
    layers = model.model.layers
    hidden_samples: list[torch.Tensor] = []
    layer_kwargs: list[dict[str, Any]] = []
    original_first_layer = layers[0]

    class Catcher(nn.Module):
        def __init__(self, wrapped: nn.Module) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, hidden_states: torch.Tensor, **kwargs):
            hidden_samples.append(hidden_states.detach())
            captured = {
                key: value
                for key, value in kwargs.items()
                if key != "past_key_value"
            }
            captured["use_cache"] = False
            layer_kwargs.append(captured)
            raise _StopAfterFirstLayer

    layers[0] = Catcher(original_first_layer)
    try:
        input_device = model.get_input_embeddings().weight.device
        for batch in dataloader:
            if len(hidden_samples) >= nsamples:
                break
            input_ids = batch[0] if isinstance(batch, (tuple, list)) else batch
            try:
                model(
                    input_ids.to(input_device),
                    use_cache=False,
                )
            except _StopAfterFirstLayer:
                pass
    finally:
        layers[0] = original_first_layer

    if len(hidden_samples) != nsamples:
        raise RuntimeError(
            f"requested {nsamples} calibration samples but captured "
            f"{len(hidden_samples)}"
        )
    return hidden_samples, layer_kwargs


def _linear_children(layer: nn.Module) -> list[tuple[str, nn.Linear]]:
    return [
        (name, module)
        for name, module in layer.named_modules()
        if name and isinstance(module, nn.Linear)
    ]


def _set_submodule(root: nn.Module, name: str, module: nn.Module) -> None:
    parent_name, _, child_name = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child_name, module)


def _supported_layer_kwargs(
    layer: nn.Module,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    parameters = inspect.signature(layer.forward).parameters
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in parameters}


def _extract_layer_hidden_states(output: Any) -> torch.Tensor:
    """Handle both old tuple and new Tensor decoder-layer return formats."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        if isinstance(output[0], torch.Tensor):
            return output[0]
    hidden_states = getattr(output, "last_hidden_state", None)
    if isinstance(hidden_states, torch.Tensor):
        return hidden_states
    raise TypeError(
        f"unsupported decoder-layer output type: {type(output).__name__}"
    )


@torch.inference_mode()
def _fit_mixnlq_activation_scales(
    samples: torch.Tensor,
    bits: int,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    channels = samples.transpose(0, 1).contiguous().float()
    _, parameters = pseudo_mix_quant(
        wbits=bits,
        W=channels,
        w_group_size=0,
        iterations=iterations,
    )
    _, _, exp_scale, linear_scale = parameters
    channel_absmax = channels.abs().amax(dim=1).clamp_min(1.0e-8)
    return exp_scale / channel_absmax, linear_scale / channel_absmax


@torch.inference_mode()
def quantize_with_calibration(
    model: nn.Module,
    tokenizer,
    weight_mode: WeightMode,
    weight_bits: int,
    weight_group_size: int,
    weight_iterations: int,
    activation_bits: int,
    activation_mode: ActivationMode,
    activation_granularity: str,
    activation_iterations: int,
    dynamic_token_iterations: int,
    usecalib: bool,
    calibration_data: str,
    calibration_samples: int,
    calibration_seqlen: int,
    calibration_tokens_per_linear: int,
    seed: int,
    use_local: bool = False,
    local_wiki2_path: str = DEFAULT_LOCAL_WIKI2_PATH,
    local_c4_path: str = DEFAULT_LOCAL_C4_PATH,
) -> int:
    dataloader = _calibration_loader(
        calibration_data,
        calibration_samples,
        seed,
        calibration_seqlen,
        model,
        tokenizer,
        use_local=use_local,
        local_wiki2_path=local_wiki2_path,
        local_c4_path=local_c4_path,
    )
    hidden_samples, saved_kwargs = _capture_first_layer_inputs(
        model,
        dataloader,
        calibration_samples,
    )
    replaced = 0

    for layer_index, layer in enumerate(
        tqdm(
            model.model.layers,
            desc="Layer-wise activation calibration",
            unit="layer",
        )
    ):
        linears = _linear_children(layer)
        collect_activation_samples = (
            activation_mode == "mixnlq"
            and activation_granularity == "channel"
        )
        collect_weight_hessian = usecalib or weight_mode == "gptq"
        collectors = {
            name: ActivationSampleCollector(
                calibration_tokens_per_linear,
                collect_samples=collect_activation_samples,
                collect_hessian=collect_weight_hessian,
            )
            for name, _ in linears
        }
        handles = []
        shared_activation_caches: dict[str, ActivationQDQCache] = {}
        for name, module in linears:
            collector = collectors[name]

            def collect_input(
                _module,
                inputs,
                collector=collector,
            ):
                collector.add(inputs[0])

            handles.append(module.register_forward_pre_hook(collect_input))

        layer_device = next(layer.parameters()).device
        next_hidden_samples = []
        for hidden, kwargs in zip(hidden_samples, saved_kwargs):
            moved_kwargs = _tree_to_device(kwargs, layer_device)
            output = layer(
                hidden.to(layer_device),
                **_supported_layer_kwargs(layer, moved_kwargs),
            )
            next_hidden_samples.append(
                _extract_layer_hidden_states(output).detach()
            )
        for handle in handles:
            handle.remove()

        for name, module in linears:
            sample_tensor = (
                collectors[name].tensor(module.weight.device)
                if collect_activation_samples
                else None
            )
            hessian = (
                collectors[name].hessian(module.weight.device)
                if collect_weight_hessian
                else None
            )
            normalized_exp = None
            normalized_linear = None
            if activation_mode == "mixnlq":
                if activation_granularity == "channel":
                    normalized_exp, normalized_linear = (
                        _fit_mixnlq_activation_scales(
                            sample_tensor,
                            activation_bits,
                            activation_iterations,
                        )
                    )
            pseudo_quantize_weight_(
                module,
                weight_mode,
                weight_bits,
                weight_group_size,
                weight_iterations,
                activation_hessian=hessian,
            )
            if activation_mode != "none":
                activation_cache = None
                if (
                    activation_mode == "mixnlq"
                    and activation_granularity == "token"
                    and normalized_exp is None
                ):
                    activation_cache = _shared_dynamic_activation_cache(
                        name,
                        shared_activation_caches,
                    )
                replacement = SimulatedQuantLinear(
                    source=module,
                    activation_mode=activation_mode,
                    activation_bits=activation_bits,
                    activation_granularity=activation_granularity,
                    activation_iterations=activation_iterations,
                    dynamic_token_iterations=dynamic_token_iterations,
                    activation_cache=activation_cache,
                    mixnlq_exp_scale_normalized=normalized_exp,
                    mixnlq_linear_scale_normalized=normalized_linear,
                )
                _set_submodule(layer, name, replacement)
            replaced += 1

        # quant.py propagates the outputs captured before quantizing the
        # current layer to the next calibration layer.
        hidden_samples = next_hidden_samples
        del collectors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return replaced


@torch.inference_mode()
def quantize_without_activation_calibration(
    model: nn.Module,
    weight_mode: WeightMode,
    weight_bits: int,
    weight_group_size: int,
    weight_iterations: int,
    activation_mode: ActivationMode,
    activation_bits: int,
    activation_granularity: str,
    activation_iterations: int,
    dynamic_token_iterations: int,
) -> int:
    names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name != "lm_head"
    ]
    shared_activation_caches: dict[str, ActivationQDQCache] = {}
    for name in tqdm(names, desc="Pseudo quantizing", unit="linear"):
        module = model.get_submodule(name)
        pseudo_quantize_weight_(
            module,
            weight_mode,
            weight_bits,
            weight_group_size,
            weight_iterations,
        )
        if activation_mode in {"rtn", "mixnlq"}:
            activation_cache = None
            if (
                activation_mode == "mixnlq"
                and activation_granularity == "token"
            ):
                activation_cache = _shared_dynamic_activation_cache(
                    name,
                    shared_activation_caches,
                )
            _set_submodule(
                model,
                name,
                SimulatedQuantLinear(
                    source=module,
                    activation_mode=activation_mode,
                    activation_bits=activation_bits,
                    activation_granularity=activation_granularity,
                    activation_iterations=activation_iterations,
                    dynamic_token_iterations=dynamic_token_iterations,
                    activation_cache=activation_cache,
                ),
            )
    return len(names)


def simulated_checkpoint(
    model: nn.Module,
    args: argparse.Namespace,
    evaluation: dict[str, float] | None = None,
) -> dict:
    checkpoint = {
        "format": SIM_CHECKPOINT_FORMAT,
        "base_model": args.model_path,
        "weight_quant_mode": args.weight_quant_mode,
        "weight_bits": args.weight_bits,
        "weight_group_size": args.weight_group_size,
        "activation_quant_mode": args.activation_quant_mode,
        "activation_bits": args.activation_bits,
        "activation_granularity": args.activation_granularity,
        "activation_iterations": args.activation_iterations,
        "dynamic_token_iterations": args.dynamic_token_iterations,
        "usecalib": args.usecalib,
        "calibration_use_local": getattr(args, "use_local", False),
        "calibration_local_paths": (
            {"wiki2": args.local_wiki2_path, "c4": args.local_c4_path}
            if getattr(args, "use_local", False) else None
        ),
        "weight_source_checkpoint": getattr(
            args,
            "weight_checkpoint",
            None,
        ),
        "model_state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
    }
    if evaluation is not None:
        checkpoint["evaluation"] = evaluation
    return checkpoint


def load_simulated_model(
    model_path: str,
    checkpoint_path: str,
    device: torch.device | None = None,
    attn_implementation: str = "sdpa",
    device_map: Any | None = None,
) -> nn.Module:
    """Reload a checkpoint produced by this script, including activation QDQ."""
    checkpoint = _load_checkpoint_file(checkpoint_path)
    if checkpoint.get("format") != SIM_CHECKPOINT_FORMAT:
        raise RuntimeError(
            f"unsupported simulated checkpoint format: "
            f"{checkpoint.get('format')!r}"
        )

    if device_map is None:
        if device is None:
            raise ValueError("either device or device_map must be supplied")
        device_map = {"": device}
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
    )
    activation_mode: ActivationMode = checkpoint["activation_quant_mode"]
    activation_bits = int(checkpoint["activation_bits"])
    activation_granularity = checkpoint["activation_granularity"]
    activation_iterations = int(
        checkpoint.get("activation_iterations", 100)
    )
    dynamic_token_iterations = int(
        checkpoint.get("dynamic_token_iterations", 2)
    )
    usecalib = bool(checkpoint.get("usecalib", False))
    if activation_mode != "none":
        names = [
            name
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear) and name != "lm_head"
        ]
        shared_activation_caches: dict[str, ActivationQDQCache] = {}
        for name in names:
            source = model.get_submodule(name)
            normalized_exp = None
            normalized_linear = None
            if (
                activation_mode == "mixnlq"
                and activation_granularity == "channel"
            ):
                normalized_exp = torch.zeros(
                    source.in_features,
                    dtype=torch.float32,
                    device=source.weight.device,
                )
                normalized_linear = torch.zeros_like(normalized_exp)
            activation_cache = None
            if (
                activation_mode == "mixnlq"
                and activation_granularity == "token"
            ):
                activation_cache = _shared_dynamic_activation_cache(
                    name,
                    shared_activation_caches,
                )
            _set_submodule(
                model,
                name,
                SimulatedQuantLinear(
                    source=source,
                    activation_mode=activation_mode,
                    activation_bits=activation_bits,
                    activation_granularity=activation_granularity,
                    activation_iterations=activation_iterations,
                    dynamic_token_iterations=dynamic_token_iterations,
                    activation_cache=activation_cache,
                    mixnlq_exp_scale_normalized=normalized_exp,
                    mixnlq_linear_scale_normalized=normalized_linear,
                ),
            )

    incompatible = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "simulated checkpoint structure mismatch: "
            f"missing={incompatible.missing_keys[:8]}, "
            f"unexpected={incompatible.unexpected_keys[:8]}"
        )
    model.quant_sim_config = {
        "format": checkpoint["format"],
        "weight_quant_mode": checkpoint["weight_quant_mode"],
        "weight_bits": int(checkpoint["weight_bits"]),
        "weight_group_size": int(checkpoint["weight_group_size"]),
        "activation_quant_mode": activation_mode,
        "activation_bits": activation_bits,
        "activation_granularity": activation_granularity,
        "activation_iterations": activation_iterations,
        "dynamic_token_iterations": dynamic_token_iterations,
        "usecalib": usecalib,
    }
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    if (
        args.weight_checkpoint is not None
        and args.activation_quant_mode == "none"
    ):
        args.activation_quant_mode = "mixnlq"
    if args.weight_bits < 2:
        raise ValueError("--weight-bits must be at least 2")
    if args.activation_bits < 2:
        raise ValueError("--activation-bits must be at least 2")
    if args.weight_iterations < 0 or args.activation_iterations < 0:
        raise ValueError("quantization iterations cannot be negative")
    if args.dynamic_token_iterations <= 0:
        raise ValueError("--dynamic-token-iterations must be positive")
    if args.calibration_samples <= 0:
        raise ValueError("--calibration-samples must be positive")
    if args.calibration_seqlen <= 0:
        raise ValueError("--calibration-seqlen must be positive")
    if args.calibration_tokens_per_linear <= 0:
        raise ValueError("--calibration-tokens-per-linear must be positive")
    if (
        args.wiki2_ppl_max_samples is not None
        and args.wiki2_ppl_max_samples <= 0
    ):
        raise ValueError("--wiki2-ppl-max-samples must be positive")
    if args.wiki2_ppl_max_length <= 1:
        raise ValueError("--wiki2-ppl-max-length must be greater than 1")
    if (
        args.wiki2_ppl_stride <= 0
        or args.wiki2_ppl_stride > args.wiki2_ppl_max_length
    ):
        raise ValueError(
            "--wiki2-ppl-stride must be in [1, --wiki2-ppl-max-length]"
        )
    if args.usecalib and args.weight_quant_mode != "mixnlq":
        raise ValueError(
            "--usecalib currently applies to activation-aware MixNLQ weight "
            "quantization and therefore requires "
            "--weight-quant-mode mixnlq"
        )
    if args.weight_checkpoint is not None:
        if args.usecalib:
            raise ValueError(
                "--weight-checkpoint reuses existing weights and requires "
                "dynamic activation fitting; do not pass --usecalib"
            )
        if not (
            args.activation_quant_mode == "mixnlq"
            and args.activation_granularity == "token"
        ):
            raise ValueError(
                "--weight-checkpoint requires dynamic per-token activations: "
                "--activation-quant-mode mixnlq "
                "--activation-granularity token"
            )

    physical_devices = _parse_visible_devices(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(physical_devices)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("quant_sim.py requires CUDA")
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
    print(f"Loading {args.model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map=(
            "balanced"
            if multi_gpu
            else {"": torch.device("cuda:0")}
        ),
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    loaded_weight_parameters = 0
    if args.weight_checkpoint is not None:
        loaded_weight_parameters, weight_metadata = load_existing_weights_(
            model,
            args.weight_checkpoint,
        )
        if weight_metadata:
            args.weight_quant_mode = weight_metadata.get(
                "weight_quant_mode",
                args.weight_quant_mode,
            )
            args.weight_bits = int(
                weight_metadata.get("weight_bits", args.weight_bits)
            )
            args.weight_group_size = int(
                weight_metadata.get(
                    "weight_group_size",
                    args.weight_group_size,
                )
            )
        print(
            f"Loaded {loaded_weight_parameters} parameter tensors from "
            f"{args.weight_checkpoint}; activation buffers were ignored."
        )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    model.config.use_cache = False

    start = time.perf_counter()
    effective_weight_mode: WeightMode = (
        "none"
        if args.weight_checkpoint is not None
        else args.weight_quant_mode
    )
    if (
        effective_weight_mode == "gptq"
        or args.usecalib
        or (
            args.activation_quant_mode == "mixnlq"
            and args.activation_granularity == "channel"
        )
    ):
        replaced = quantize_with_calibration(
            model=model,
            tokenizer=tokenizer,
            weight_mode=effective_weight_mode,
            weight_bits=args.weight_bits,
            weight_group_size=args.weight_group_size,
            weight_iterations=args.weight_iterations,
            activation_bits=args.activation_bits,
            activation_mode=args.activation_quant_mode,
            activation_granularity=args.activation_granularity,
            activation_iterations=args.activation_iterations,
            dynamic_token_iterations=args.dynamic_token_iterations,
            usecalib=args.usecalib,
            calibration_data=args.calibration_data,
            calibration_samples=args.calibration_samples,
            calibration_seqlen=args.calibration_seqlen,
            calibration_tokens_per_linear=args.calibration_tokens_per_linear,
            seed=args.seed,
            use_local=args.use_local,
            local_wiki2_path=args.local_wiki2_path,
            local_c4_path=args.local_c4_path,
        )
    else:
        replaced = quantize_without_activation_calibration(
            model=model,
            weight_mode=effective_weight_mode,
            weight_bits=args.weight_bits,
            weight_group_size=args.weight_group_size,
            weight_iterations=args.weight_iterations,
            activation_mode=args.activation_quant_mode,
            activation_bits=args.activation_bits,
            activation_granularity=args.activation_granularity,
            activation_iterations=args.activation_iterations,
            dynamic_token_iterations=args.dynamic_token_iterations,
        )
    elapsed = time.perf_counter() - start
    print(
        f"Prepared {replaced} Linear layers in {elapsed:.2f} s: "
        f"W={args.weight_quant_mode}{args.weight_bits}"
        f"{' (loaded)' if args.weight_checkpoint is not None else ''}, "
        f"A={args.activation_quant_mode}{args.activation_bits}, "
        f"dynamic_token_iterations={args.dynamic_token_iterations}, "
        f"weight_activation_calibration={args.usecalib}."
    )

    evaluation = {} if args.eval_wiki2_ppl or args.eval_c4_ppl else None
    if args.eval_wiki2_ppl:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\nEvaluating WikiText-2 perplexity before checkpoint export ...")
        wiki2_texts = load_wikitext_data(
            "test",
            args.wiki2_ppl_max_samples,
        )
        perplexity, average_loss = calculate_perplexity(
            model=model,
            tokenizer=tokenizer,
            texts=wiki2_texts,
            batch_size=1,
            max_length=args.wiki2_ppl_max_length,
            stride=args.wiki2_ppl_stride,
            print_info=False,
        )
        evaluation["wiki2_perplexity"] = float(perplexity.item())
        evaluation["wiki2_average_loss"] = float(average_loss.item())
        print(
            f"WikiText-2 perplexity={evaluation['wiki2_perplexity']:.4f}, "
            f"average_loss={evaluation['wiki2_average_loss']:.6f}"
        )

    if args.eval_c4_ppl:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        c4_result = evaluate_ppl_tasks(model, tokenizer, ["c4"], 1000)["c4"]
        evaluation["c4_perplexity"] = c4_result["perplexity"]
        evaluation["c4_average_loss"] = c4_result["average_loss"]

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            simulated_checkpoint(model, args, evaluation=evaluation),
            output,
        )
        print(f"Saved pseudo-quantized checkpoint to {output}")


if __name__ == "__main__":
    main()
