"""End-to-end benchmark for BF16, packed-INT4, and SpinQuant-equivalent models.

All variants use the same optional QKV/Gate-Up fusion, static KV cache, static
decode buffers, and separately compiled prefill/decode paths. No TinyGemm
dependency is used. The SpinQuant variant keeps offline R1/R2 out of the
forward, accounts for their checkpoint storage, and times the online R4.
"""

import argparse
import functools
import gc
import importlib
import json
import math
import os
import statistics
import inspect
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
import torch
from torch import nn
from transformers import AutoModelForCausalLM, StaticCache
from utils.llama_fusion import fuse_llama_projections
from utils.triton_quant_linear import (
    CustomBF16Linear,
    QuantMode,
    TritonInt4Linear,
)

try:
    from fast_hadamard_transform import (
        hadamard_transform as _fast_hadamard_transform,
    )
except (ImportError, OSError):
    _fast_hadamard_transform = None

if _fast_hadamard_transform is not None:
    @torch.library.custom_op(
        "mixnlq::spinquant_fast_hadamard",
        mutates_args=(),
    )
    def _spinquant_fast_hadamard(
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        return _fast_hadamard_transform(inputs)

    @_spinquant_fast_hadamard.register_fake
    def _spinquant_fast_hadamard_fake(
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        return torch.empty_like(inputs)
else:
    _spinquant_fast_hadamard = None


def patch_transformers_output_capturing() -> None:
    """Work around Transformers releases that use torch without importing it."""
    try:
        output_capturing = importlib.import_module(
            "transformers.utils.output_capturing"
        )
    except ImportError:
        return

    if "torch" not in output_capturing.__dict__:
        # The decorators have already been created, but their functions retain
        # this module dictionary as ``__globals__``. Populating it here is
        # therefore sufficient and does not require patching site-packages.
        output_capturing.torch = torch
        warnings.warn(
            "Patched transformers.utils.output_capturing: injected missing "
            "runtime torch import.",
            RuntimeWarning,
            stacklevel=2,
        )


@dataclass
class BenchmarkResult:
    name: str
    weight_storage_gib: float
    rotation_storage_gib: float
    loaded_model_gib: float
    prefill_ms: float
    prefill_tokens_per_second: float
    decode_ms_per_token: float
    decode_tokens_per_second: float
    e2e_ms: float
    generated_tokens_per_second: float
    peak_memory_gib: float
    prefill_tflop: float
    prefill_effective_tflops: float
    decode_gflop_per_token: float
    decode_effective_tflops: float
    e2e_effective_tflops: float
    decode_split_k: dict[str, int]
    speedup_vs_bf16: float = 1.0
    prefill_tflops_vs_bf16: float = 1.0
    decode_tflops_vs_bf16: float = 1.0
    e2e_tflops_vs_bf16: float = 1.0
    speedup_vs_lut4: float | None = None
    speedup_vs_linear: float | None = None
    speedup_vs_lut3: float | None = None
    speedup_vs_linear3: float | None = None


def _gf27_add(left: tuple[int, int, int], right: tuple[int, int, int]):
    return tuple((a + b) % 3 for a, b in zip(left, right))


def _gf27_neg(value: tuple[int, int, int]):
    return tuple((-item) % 3 for item in value)


def _gf27_mul(left: tuple[int, int, int], right: tuple[int, int, int]):
    """Multiply in GF(3)[t] / (t^3 + 2t + 1)."""
    product = [0] * 5
    for left_degree, left_value in enumerate(left):
        for right_degree, right_value in enumerate(right):
            product[left_degree + right_degree] += (
                left_value * right_value
            )
    product = [value % 3 for value in product]
    # t^3 = t + 2 in this field.
    for degree in (4, 3):
        coefficient = product[degree]
        if coefficient:
            product[degree] = 0
            product[degree - 3] = (
                product[degree - 3] + 2 * coefficient
            ) % 3
            product[degree - 2] = (
                product[degree - 2] + coefficient
            ) % 3
    return tuple(product[:3])


@functools.lru_cache(maxsize=1)
def _paley_hadamard_28() -> torch.Tensor:
    """Construct the order-28 Paley matrix used by the 14336-wide R4."""
    elements = [
        (constant, linear, quadratic)
        for quadratic in range(3)
        for linear in range(3)
        for constant in range(3)
    ]
    zero = (0, 0, 0)
    squares = {
        _gf27_mul(value, value)
        for value in elements
        if value != zero
    }
    q_matrix = torch.empty((27, 27), dtype=torch.float32)
    for row, left in enumerate(elements):
        for column, right in enumerate(elements):
            difference = _gf27_add(left, _gf27_neg(right))
            q_matrix[row, column] = (
                0.0
                if difference == zero
                else (1.0 if difference in squares else -1.0)
            )
    hadamard = torch.ones((28, 28), dtype=torch.float32)
    hadamard[1:, 1:] = q_matrix - torch.eye(27)
    gram = hadamard @ hadamard.T
    expected = 28.0 * torch.eye(28)
    if not torch.equal(gram, expected):
        raise RuntimeError("failed to construct the order-28 Hadamard matrix")
    return hadamard


class SpinQuantR4DownProjection(nn.Module):
    """SpinQuant-equivalent online R4 followed by the packed down projection."""

    def __init__(self, projection: nn.Module) -> None:
        super().__init__()
        if _fast_hadamard_transform is None:
            raise RuntimeError(
                "SpinQuant R4 benchmarking requires "
                "fast-hadamard-transform"
            )
        self.projection = projection
        self.in_features = int(projection.in_features)
        if self.in_features == 14336:
            self.had_k_order = 28
            self.fast_hadamard_size = 512
            had_k = _paley_hadamard_28()
        elif self.in_features > 0 and (
            self.in_features & (self.in_features - 1)
        ) == 0:
            self.had_k_order = 1
            self.fast_hadamard_size = self.in_features
            had_k = torch.ones((1, 1), dtype=torch.float32)
        else:
            from utils import hadamard_utils

            for order in (172, 156, 140, 108, 76, 60, 52, 44, 40, 36, 28, 20, 12):
                size, remainder = divmod(self.in_features, order)
                if remainder == 0 and size > 0 and (size & (size - 1)) == 0:
                    self.had_k_order = order
                    self.fast_hadamard_size = size
                    had_k = getattr(hadamard_utils, f"get_had{order}")()
                    break
            else:
                raise ValueError(
                    "Unsupported SpinQuant R4 input width "
                    f"{self.in_features}: no exact Hadamard decomposition"
                )
        reference = next(
            (
                tensor
                for tensor in (
                    list(projection.parameters())
                    + list(projection.buffers())
                )
                if tensor.is_floating_point()
            ),
            None,
        )
        if reference is None:
            raise ValueError("SpinQuant projection has no floating-point scale")
        self.register_buffer(
            "had_k",
            had_k.to(device=reference.device, dtype=reference.dtype),
        )
        self.inv_sqrt_width = 1.0 / math.sqrt(self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        rows = x.reshape(
            -1,
            self.had_k_order,
            self.fast_hadamard_size,
        ).contiguous()
        rows = _spinquant_fast_hadamard(rows)
        if self.had_k_order > 1:
            rows = torch.matmul(self.had_k, rows)
        rotated = (rows * self.inv_sqrt_width).reshape(input_shape)
        return self.projection(rotated)


def add_spinquant_r4_(model: nn.Module) -> int:
    """Wrap dense, routed-expert and shared-expert MLP down projections."""
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if name.rsplit(".", 1)[-1] == "down_proj"
    ]
    if not targets:
        raise ValueError(
            "SpinQuant R4 requires MLP down_proj modules; found none in "
            f"model_type={getattr(model.config, 'model_type', None)!r}"
        )
    # Validate every width before modifying the model.
    replacements = []
    for name, projection in targets:
        if isinstance(projection, SpinQuantR4DownProjection):
            continue
        try:
            replacements.append((name, SpinQuantR4DownProjection(projection)))
        except (ValueError, AttributeError) as exc:
            raise ValueError(f"Cannot apply SpinQuant R4 to {name}: {exc}") from exc
    for name, projection in replacements:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, projection)
    return len(replacements)


def spinquant_offline_rotation_storage_gib(model: nn.Module) -> float:
    """Estimate FP32 R1 plus one value-head R2 per layer (not a file measurement)."""
    config = model.config
    hidden_size = int(config.hidden_size)
    num_layers = int(config.num_hidden_layers)
    num_heads = int(config.num_attention_heads)
    head_dim = int(
        getattr(config, "v_head_dim", None)
        or getattr(config, "head_dim", None)
        or hidden_size // num_heads
    )
    elements = hidden_size * hidden_size + (
        num_layers * head_dim * head_dim
    )
    return elements * 4 / (1024**3)


def estimate_model_flops(
    model: nn.Module,
    batch_size: int,
    prompt_length: int,
    decode_steps: int,
    prefill_logits_to_keep_one: bool,
) -> tuple[float, float]:
    """Return prefill FLOPs and average FLOPs per generated Decode token.

    The estimate covers projection/MLP/LM-head matmuls and QK/AV attention.
    It deliberately excludes elementwise work and quantization decode logic,
    so INT4 results are comparable "effective model FLOP/s", not a count of
    native floating-point instructions executed by the custom kernels.
    """
    config = model.config
    hidden_size = int(config.hidden_size)
    intermediate_size = int(config.intermediate_size)
    num_layers = int(config.num_hidden_layers)
    num_heads = int(config.num_attention_heads)
    num_kv_heads = int(
        getattr(config, "num_key_value_heads", num_heads)
    )
    configured_head_dim = getattr(config, "head_dim", None)
    head_dim = int(configured_head_dim or (hidden_size // num_heads))
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim

    # A multiply-add counts as two FLOPs.
    attention_linear_flops = 2 * (
        hidden_size * (q_size + 2 * kv_size)
        + q_size * hidden_size
    )
    attention_pair_flops = 4 * q_size
    if getattr(config, "kv_lora_rank", None) is not None:
        qk_dim = int(config.qk_nope_head_dim + config.qk_rope_head_dim)
        value_dim = int(config.v_head_dim)
        kv_rank = int(config.kv_lora_rank)
        q_rank = getattr(config, "q_lora_rank", None)
        q_weights = hidden_size * num_heads * qk_dim if q_rank is None else (
            hidden_size * int(q_rank) + int(q_rank) * num_heads * qk_dim
        )
        attention_linear_flops = 2 * (
            q_weights
            + hidden_size * (kv_rank + int(config.qk_rope_head_dim))
            + kv_rank * num_heads * (int(config.qk_nope_head_dim) + value_dim)
            + num_heads * value_dim * hidden_size
        )
        attention_pair_flops = 2 * num_heads * (qk_dim + value_dim)

    transformer_linear_flops_per_token = num_layers * (
        attention_linear_flops + 6 * hidden_size * intermediate_size
    )
    if getattr(config, "n_routed_experts", None):
        mlp_flops = 0
        for layer in model.model.layers:
            mlp = layer.mlp
            if hasattr(mlp, "experts"):
                active_width = (
                    int(config.num_experts_per_tok)
                    * int(config.moe_intermediate_size)
                )
                shared = getattr(mlp, "shared_experts", None)
                if shared is not None:
                    active_width += int(shared.down_proj.in_features)
                mlp_flops += 6 * hidden_size * active_width
                mlp_flops += 2 * hidden_size * int(config.n_routed_experts)
            else:
                mlp_flops += 6 * hidden_size * int(mlp.down_proj.in_features)
        transformer_linear_flops_per_token = (
            num_layers * attention_linear_flops + mlp_flops
        )
    transformer_linear_prefill = (
        batch_size
        * prompt_length
        * transformer_linear_flops_per_token
    )
    # Causal attention has L(L+1)/2 query/key pairs; MLA has unequal QK/V widths.
    attention_prefill = (
        0.5 * attention_pair_flops
        * batch_size
        * num_layers
        * prompt_length
        * (prompt_length + 1)
    )
    lm_head_tokens = batch_size if prefill_logits_to_keep_one else (
        batch_size * prompt_length
    )
    lm_head_prefill = (
        2 * lm_head_tokens * hidden_size * int(config.vocab_size)
    )
    prefill_flops = (
        transformer_linear_prefill + attention_prefill + lm_head_prefill
    )

    if decode_steps <= 0:
        return float(prefill_flops), 0.0
    # The first Decode call appends position prompt_length and therefore
    # attends over prompt_length+1 keys. Each later call adds one key.
    average_context_length = prompt_length + (decode_steps + 1) / 2.0
    decode_transformer_per_token = transformer_linear_flops_per_token
    decode_attention_per_token = (
        attention_pair_flops * num_layers * average_context_length
    )
    decode_lm_head_per_token = (
        2 * hidden_size * int(config.vocab_size)
    )
    decode_flops_per_token = (
        decode_transformer_per_token
        + decode_attention_per_token
        + decode_lm_head_per_token
    )
    return float(prefill_flops), float(decode_flops_per_token)


def model_storage_gib(model: nn.Module) -> float:
    """Return unique persistent parameter/buffer storage owned by the model."""
    seen_storages = set()
    storage_bytes = 0
    tensors = list(model.parameters()) + list(model.buffers())
    for tensor in tensors:
        if tensor.numel() == 0:
            continue
        storage = tensor.untyped_storage()
        key = (tensor.device.type, tensor.device.index, storage.data_ptr())
        if key not in seen_storages:
            seen_storages.add(key)
            storage_bytes += storage.nbytes()
    return storage_bytes / (1024**3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare cuBLAS BF16, packed MixNLQ/linear INT4, and an optional "
            "SpinQuant-equivalent linear INT4 + online R4 path with identical "
            "model-level scheduling."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--model-path", required=True, help="Base Hugging Face model")
    parser.add_argument(
        "--bf16-checkpoint",
        default=None,
        help="Optional trained full-model state_dict loaded for all four variants",
    )
    parser.add_argument(
        "--mixnlq-checkpoint",
        default=None,
        help="Packed MixNLQ checkpoint exported by quant2.py",
    )
    parser.add_argument(
        "--mixnlq3-checkpoint",
        default=None,
        help=(
            "Optional packed 3-bit MixNLQ checkpoint exported by "
            "quant_real.py --quant-mode mixnlq3"
        ),
    )
    parser.add_argument(
        "--linear-checkpoint",
        default=None,
        help="Packed linear/RTN checkpoint exported by quant2.py",
    )
    parser.add_argument(
        "--linear3-checkpoint",
        default=None,
        help="Optional packed 3-bit linear/RTN checkpoint",
    )
    parser.add_argument(
        "--lut3-checkpoint",
        default=None,
        help="Optional packed LUT3-group checkpoint exported by quant_real.py",
    )
    parser.add_argument(
        "--spinquant3-checkpoint",
        default=None,
        help=(
            "Optional packed Linear3 checkpoint from rotated weights; "
            "enables the same online SpinQuant R4 path"
        ),
    )
    parser.add_argument(
        "--spinquant-checkpoint",
        default=None,
        help=(
            "Optional packed linear/RTN checkpoint exported from "
            "SpinQuant-rotated weights. Adds an equivalent SpinQuant variant "
            "with online R4 before every down_proj. For speed-only comparison, "
            "--linear-checkpoint may be supplied here as well"
        ),
    )
    parser.add_argument(
        "--lut4-checkpoint",
        "--lut-checkpoint",
        dest="lut4_checkpoint",
        default=None,
        help=(
            "Optional LUT4-group checkpoint exported with "
            "quant_real.py --quant-mode lut; --lut-checkpoint is a legacy alias"
        ),
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Single physical CUDA id; the benchmark uses logical cuda:0",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-length", type=int, default=2048)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument(
        "--input-mode",
        choices=("random", "wikitext2", "wiki2"),
        default="random",
        help="Use random token IDs or real tokenized WikiText-2 windows",
    )
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--benchmark-runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "eager", "flash_attention_2"),
        default="sdpa",
        help="HF attention backend; sdpa uses PyTorch CUDA attention by default",
    )
    parser.add_argument("--disable-compile", action="store_true")
    parser.add_argument(
        "--disable-split-k-autotune",
        action="store_true",
        help="Use the original native CUDA Split-K heuristic instead of GPU/shape tuning",
    )
    parser.add_argument("--disable-projection-fusion", action="store_true")
    parser.add_argument(
        "--mixnlq-decode-backend",
        "--int4-decode-backend",
        dest="mixnlq_decode_backend",
        choices=("auto", "cuda", "triton"),
        default="auto",
        help=(
            "Use the optional native CUDA MixNLQ/linear/group-LUT M=1 "
            "operators, force the existing Triton paths, or select CUDA when "
            "the extension is available"
        ),
    )
    parser.add_argument(
        "--mixnlq-cuda-kernel",
        choices=("auto", "scalar", "vec2"),
        default="auto",
        help=(
            "Select the native MixNLQ CUDA Decode kernel. auto uses vec2 for "
            "group sizes divisible by 16 (including per-channel mode) and "
            "otherwise falls back to the scalar-output kernel"
        ),
    )
    parser.add_argument(
        "--output-json", default=None, help="Optional path for machine-readable results"
    )
    return parser.parse_args()


def load_wikitext2_input(
    model_path: str,
    batch_size: int,
    prompt_length: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Tokenize WikiText-2 and select deterministic fixed-length windows."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    dataset_name = "Salesforce/wikitext/wikitext-2-raw-v1"

    encoded = tokenizer(
        "\n\n".join(dataset["text"]),
        return_tensors="pt",
    ).input_ids
    token_count = encoded.shape[1]
    if token_count < prompt_length:
        raise RuntimeError(
            f"WikiText-2 test split has only {token_count} tokens, but "
            f"--prompt-length is {prompt_length}"
        )

    generator = torch.Generator(device="cpu").manual_seed(seed)
    max_start = token_count - prompt_length
    starts = torch.randint(
        0,
        max_start + 1,
        (batch_size,),
        generator=generator,
    ).tolist()
    input_ids = torch.cat(
        [encoded[:, start : start + prompt_length] for start in starts],
        dim=0,
    )
    print(
        f"Loaded real inputs from {dataset_name} (test); "
        f"token_count={token_count}, window_starts={starts}."
    )
    return input_ids.to(device=device, non_blocking=True)


def load_checkpoint(path: str) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=True)


def replace_with_integer_linears(
    model: nn.Module,
    checkpoint: dict,
    expected_mode: QuantMode,
) -> int:
    supported_formats = {
        "packed-weight-only-int4-v2-kmajor",
        "packed-weight-only-int4-v3",
        "packed-weight-only-v4",
    }
    if checkpoint.get("format") not in supported_formats:
        raise RuntimeError("Unsupported checkpoint format; export it with quant2.py")
    quant_mode = checkpoint.get("quant_mode")
    if quant_mode != expected_mode:
        raise RuntimeError(
            f"Expected a {expected_mode} checkpoint, got {quant_mode!r}"
        )
    expected_bits = (
        3
        if quant_mode in {"mixnlq3", "linear3", "lut3"}
        else 4
    )
    if checkpoint.get("bits") != expected_bits:
        raise RuntimeError(
            f"{quant_mode} requires a packed {expected_bits}-bit checkpoint, "
            f"got bits={checkpoint.get('bits')!r}"
        )
    group_size = int(checkpoint["group_size"])
    entries = checkpoint.get("layers", {})
    layer_names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name != "lm_head"
    ]
    missing = [name for name in layer_names if name not in entries]
    if missing:
        raise RuntimeError(
            f"Packed checkpoint is missing {len(missing)} layers: {missing[:8]}"
        )
    for name in layer_names:
        layer = model.get_submodule(name)
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        entry = entries[name]
        if (
            int(entry["in_features"]) != layer.in_features
            or int(entry["out_features"]) != layer.out_features
        ):
            raise RuntimeError(f"Shape mismatch for packed layer {name}")
        integer_layer = TritonInt4Linear.from_checkpoint_entry(
            entry,
            quant_mode=quant_mode,
            group_size=group_size,
            device=layer.weight.device,
            dtype=layer.weight.dtype,
        )
        setattr(parent, child_name, integer_layer)
        del layer, integer_layer
    return len(layer_names)


def replace_with_custom_bf16_linears(model: nn.Module) -> int:
    layer_names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name != "lm_head"
    ]
    for name in layer_names:
        layer = model.get_submodule(name)
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, CustomBF16Linear(layer))
        del layer
    return len(layer_names)


def load_model(
    model_path: str,
    device: torch.device,
    bf16_checkpoint_path: str | None = None,
    checkpoint_path: str | None = None,
    quant_mode: QuantMode | None = None,
    custom_bf16: bool = False,
    fuse_projections: bool = True,
    attn_implementation: str = "sdpa",
    spinquant_r4: bool = False,
    device_map: str | dict | None = None,
) -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device_map if device_map is not None else {"": device},
        trust_remote_code=False,
        attn_implementation=attn_implementation,
    )
    if bf16_checkpoint_path is not None:
        trained_state = load_checkpoint(bf16_checkpoint_path)
        incompatible = model.load_state_dict(trained_state, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                f"Unexpected BF16 checkpoint keys: {incompatible.unexpected_keys[:8]}"
            )
        del trained_state

    if quant_mode is not None:
        if checkpoint_path is None:
            raise ValueError(f"{quant_mode} requires a packed checkpoint")
        checkpoint = load_checkpoint(checkpoint_path)
        replaced = replace_with_integer_linears(
            model,
            checkpoint=checkpoint,
            expected_mode=quant_mode,
        )
        group_size = int(checkpoint["group_size"])
        del checkpoint
        bits = (
            3
            if quant_mode in {"mixnlq3", "linear3", "lut3"}
            else 4
        )
        print(
            f"Loaded {replaced} packed {quant_mode} {bits}-bit layers "
            f"(group_size={group_size})."
        )
        if quant_mode in {
            "mixnlq", "mixnlq3", "linear", "linear3",
            "nf4_lut", "lut3",
        }:
            cuda_decode_layers = sum(
                isinstance(module, TritonInt4Linear)
                and module.use_cuda_decode
                for module in model.modules()
            )
            decode_backend = "native CUDA" if cuda_decode_layers else "Triton"
            print(
                f"{quant_mode} Decode backend: {decode_backend} "
                f"({cuda_decode_layers}/{replaced} CUDA layers before fusion)."
            )
    elif custom_bf16:
        replaced = replace_with_custom_bf16_linears(model)
        print(f"Replaced {replaced} Linear layers with custom BF16 controls.")

    if fuse_projections:
        qkv_count, mlp_count = fuse_llama_projections(model)
        if qkv_count:
            print(
                f"Fused {qkv_count} QKV projection groups and "
                f"{mlp_count} Gate/Up projection groups."
            )
        else:
            print("Projection fusion skipped: this architecture has no supported fusion groups.")

    if spinquant_r4:
        r4_count = add_spinquant_r4_(model)
        print(
            f"Inserted {r4_count} optimized SpinQuant R4 online rotations "
            "before down_proj."
        )

    model.eval()
    model.config.use_cache = True
    return model


class BenchmarkStaticCache(StaticCache):
    """StaticCache with separate value widths for DeepSeek MLA attention."""

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        layers = getattr(self, "layers", None)
        if layers is not None and key_states.shape[-1] != value_states.shape[-1]:
            layer = layers[layer_idx]
            if not layer.is_initialized:
                layer.lazy_initialization(key_states)
            if layer.values.shape[-1] != value_states.shape[-1]:
                # The bootstrap forward allocates this before graph capture.
                if torch.compiler.is_compiling():
                    raise RuntimeError("Initialize MLA KV cache before compilation")
                shape = (*layer.values.shape[:-1], value_states.shape[-1])
                layer.values = value_states.new_zeros(shape)
                torch._dynamo.mark_static_address(layer.values)
        return super().update(key_states, value_states, layer_idx, cache_kwargs)


class StaticCompiledRunner:
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
        self.max_length = prompt_length + new_tokens
        try:
            self.cache = BenchmarkStaticCache(
                config=model.config, max_cache_len=self.max_length
            )
        except TypeError:
            self.cache = BenchmarkStaticCache(
                config=model.config,
                max_batch_size=batch_size,
                max_cache_len=self.max_length,
                device=device,
                dtype=torch.bfloat16,
            )
        self.attention_mask = torch.ones(
            (batch_size, self.max_length), dtype=torch.long, device=device
        )
        self.prefill_position = torch.arange(prompt_length, device=device)
        self.decode_position = torch.empty(1, dtype=torch.long, device=device)
        self.token_buffer = torch.empty(
            (batch_size, 1), dtype=torch.long, device=device
        )
        mark_static_address = getattr(
            getattr(torch, "_dynamo", None), "mark_static_address", None
        )
        if mark_static_address is not None:
            for buffer in (
                self.attention_mask,
                self.prefill_position,
                self.decode_position,
                self.token_buffer,
            ):
                mark_static_address(buffer)
        self.cudagraph_mark_step_begin = getattr(
            getattr(torch, "compiler", None),
            "cudagraph_mark_step_begin",
            None,
        )
        self.logits_to_keep_supported = (
            "logits_to_keep" in inspect.signature(model.forward).parameters
        )

        def prefill_forward(input_ids, attention_mask, cache_position):
            kwargs = {"logits_to_keep": 1} if self.logits_to_keep_supported else {}
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=self.cache,
                cache_position=cache_position,
                use_cache=True,
                **kwargs,
            )
            return output.logits[:, -1:].argmax(dim=-1)

        def decode_step(input_ids, attention_mask, cache_position):
            kwargs = {"logits_to_keep": 1} if self.logits_to_keep_supported else {}
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
            # Allocate lazy KV/rotary buffers outside CUDA Graph's output pool.
            # Both compiled paths must mutate the same externally owned cache.
            with torch.inference_mode():
                bootstrap_input = torch.zeros(
                    (batch_size, prompt_length), dtype=torch.long, device=device
                )
                prefill_forward(
                    bootstrap_input, self.attention_mask, self.prefill_position
                )
                self.cache.reset()
                del bootstrap_input
            torch.cuda.synchronize(device)
            self.prefill_forward = torch.compile(
                prefill_forward, mode="max-autotune", fullgraph=False
            )
            self.decode_step = torch.compile(
                decode_step, mode="reduce-overhead", fullgraph=False
            )
        else:
            self.prefill_forward = prefill_forward
            self.decode_step = decode_step

    @torch.inference_mode()
    def timed_run(
        self, input_ids: torch.Tensor, new_tokens: int
    ) -> tuple[float, float]:
        self.cache.reset()
        start = torch.cuda.Event(enable_timing=True)
        prefill_done = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if self.cudagraph_mark_step_begin is not None:
            self.cudagraph_mark_step_begin()
        next_token = self.prefill_forward(
            input_ids, self.attention_mask, self.prefill_position
        )
        prefill_done.record()
        for index in range(new_tokens - 1):
            self.token_buffer.copy_(next_token)
            del next_token
            self.decode_position.fill_(input_ids.shape[1] + index)
            if self.cudagraph_mark_step_begin is not None:
                self.cudagraph_mark_step_begin()
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
    weight_storage_gib: float,
    rotation_storage_gib: float,
    loaded_model_gib: float,
    new_tokens: int,
    warmup_runs: int,
    benchmark_runs: int,
    compile_enabled: bool,
    autotune_split_k_enabled: bool = True,
) -> BenchmarkResult:
    split_k_settings = {}
    # Dense batched decode does not use GEMV; MoE experts may still receive M=1.
    if input_ids.shape[0] == 1 or getattr(model.config, "n_routed_experts", None):
        for layer_name, layer in model.named_modules():
            if isinstance(layer, TritonInt4Linear) and layer.use_cuda_decode:
                if autotune_split_k_enabled:
                    layer.autotune_decode_split_k()
                split_k_settings[layer_name] = layer.decode_split_k
    runner = StaticCompiledRunner(
        model=model,
        batch_size=input_ids.shape[0],
        prompt_length=input_ids.shape[1],
        new_tokens=new_tokens,
        device=input_ids.device,
        compile_enabled=compile_enabled,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    for _ in range(warmup_runs):
        runner.timed_run(input_ids, new_tokens)

    prefill_samples = []
    e2e_samples = []
    decode_samples = []
    for _ in range(benchmark_runs):
        prefill_ms, e2e_ms = runner.timed_run(input_ids, new_tokens)
        prefill_samples.append(prefill_ms)
        e2e_samples.append(e2e_ms)
        decode_samples.append(e2e_ms - prefill_ms)

    prefill_ms = statistics.median(prefill_samples)
    e2e_ms = statistics.median(e2e_samples)
    decode_ms = statistics.median(decode_samples)
    batch_size, prompt_length = input_ids.shape
    decode_steps = new_tokens - 1
    generated_tokens = batch_size * new_tokens
    prefill_flops, decode_flops_per_token = estimate_model_flops(
        model=model,
        batch_size=batch_size,
        prompt_length=prompt_length,
        decode_steps=decode_steps,
        prefill_logits_to_keep_one=runner.logits_to_keep_supported,
    )
    prefill_effective_tflops = prefill_flops / (prefill_ms * 1.0e9)
    decode_effective_tflops = (
        decode_flops_per_token
        * batch_size
        * decode_steps
        / (decode_ms * 1.0e9)
    )
    e2e_flops = (
        prefill_flops
        + decode_flops_per_token * batch_size * decode_steps
    )
    e2e_effective_tflops = e2e_flops / (e2e_ms * 1.0e9)

    return BenchmarkResult(
        name=name,
        weight_storage_gib=weight_storage_gib,
        rotation_storage_gib=rotation_storage_gib,
        loaded_model_gib=loaded_model_gib,
        prefill_ms=prefill_ms,
        prefill_tokens_per_second=batch_size * prompt_length * 1000.0 / prefill_ms,
        decode_ms_per_token=decode_ms / decode_steps,
        decode_tokens_per_second=batch_size * decode_steps * 1000.0 / decode_ms,
        e2e_ms=e2e_ms,
        generated_tokens_per_second=generated_tokens * 1000.0 / e2e_ms,
        peak_memory_gib=torch.cuda.max_memory_allocated() / (1024**3),
        prefill_tflop=prefill_flops / 1.0e12,
        prefill_effective_tflops=prefill_effective_tflops,
        decode_gflop_per_token=decode_flops_per_token / 1.0e9,
        decode_effective_tflops=decode_effective_tflops,
        e2e_effective_tflops=e2e_effective_tflops,
        decode_split_k=split_k_settings,
    )


def print_results(results: list[BenchmarkResult]) -> None:
    headers = (
        "model",
        "weight GiB",
        "rotation GiB",
        "total GiB",
        "loaded GiB",
        "prefill ms",
        "prefill tok/s",
        "decode ms/tok",
        "decode tok/s",
        "E2E ms",
        "speedup",
        "vs Linear",
        "vs Linear3",
        "vs LUT4",
        "vs LUT3",
        "peak GiB",
    )
    rows = [
        (
            result.name,
            f"{result.weight_storage_gib:.2f}",
            f"{result.rotation_storage_gib:.2f}",
            f"{result.weight_storage_gib + result.rotation_storage_gib:.2f}",
            f"{result.loaded_model_gib:.2f}",
            f"{result.prefill_ms:.2f}",
            f"{result.prefill_tokens_per_second:.1f}",
            f"{result.decode_ms_per_token:.3f}",
            f"{result.decode_tokens_per_second:.1f}",
            f"{result.e2e_ms:.2f}",
            f"{result.speedup_vs_bf16:.3f}x",
            (
                "N/A"
                if result.speedup_vs_linear is None
                else f"{result.speedup_vs_linear:.3f}x"
            ),
            (
                "N/A"
                if result.speedup_vs_linear3 is None
                else f"{result.speedup_vs_linear3:.3f}x"
            ),
            (
                "N/A"
                if result.speedup_vs_lut4 is None
                else f"{result.speedup_vs_lut4:.3f}x"
            ),
            (
                "N/A"
                if result.speedup_vs_lut3 is None
                else f"{result.speedup_vs_lut3:.3f}x"
            ),
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


def print_flops_results(results: list[BenchmarkResult]) -> None:
    headers = (
        "model",
        "prefill TFLOP",
        "prefill TFLOP/s",
        "vs BF16",
        "decode GFLOP/tok",
        "decode TFLOP/s",
        "vs BF16",
        "E2E TFLOP/s",
        "vs BF16",
    )
    rows = [
        (
            result.name,
            f"{result.prefill_tflop:.3f}",
            f"{result.prefill_effective_tflops:.2f}",
            f"{result.prefill_tflops_vs_bf16:.3f}x",
            f"{result.decode_gflop_per_token:.3f}",
            f"{result.decode_effective_tflops:.2f}",
            f"{result.decode_tflops_vs_bf16:.3f}x",
            f"{result.e2e_effective_tflops:.2f}",
            f"{result.e2e_tflops_vs_bf16:.3f}x",
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


def print_core_overheads(results: list[BenchmarkResult]) -> bool:
    linear = next(
        (result for result in results if result.name == "linear-int4"),
        None,
    )
    if linear is None:
        return False
    compared = [
        result
        for result in results
        if result.name in {
            "mixnlq-int4",
            "spinquant-linear-int4",
        }
    ]
    if not compared:
        return False
    print("\nIncremental core overhead versus packed Linear INT4")
    headers = (
        "model",
        "prefill delta ms",
        "prefill overhead",
        "decode delta ms/tok",
        "decode overhead",
        "E2E delta ms",
        "E2E overhead",
    )
    rows = []
    for result in compared:
        prefill_delta = result.prefill_ms - linear.prefill_ms
        decode_delta = (
            result.decode_ms_per_token - linear.decode_ms_per_token
        )
        e2e_delta = result.e2e_ms - linear.e2e_ms
        rows.append(
            (
                result.name,
                f"{prefill_delta:+.3f}",
                f"{prefill_delta / linear.prefill_ms:+.2%}",
                f"{decode_delta:+.4f}",
                (
                    f"{decode_delta / linear.decode_ms_per_token:+.2%}"
                ),
                f"{e2e_delta:+.3f}",
                f"{e2e_delta / linear.e2e_ms:+.2%}",
            )
        )
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    print("  ".join(value.ljust(widths[i]) for i, value in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))
    return True


def main() -> None:
    patch_transformers_output_capturing()
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.prompt_length <= 0:
        raise ValueError("--prompt-length must be positive")
    if args.new_tokens < 2:
        raise ValueError("--new-tokens must be at least 2 to measure cached decoding")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs cannot be negative")
    if args.benchmark_runs <= 0:
        raise ValueError("--benchmark-runs must be positive")

    physical_device = args.device.strip()
    if not physical_device or "," in physical_device:
        raise ValueError("--device must select exactly one physical CUDA device")
    os.environ["CUDA_VISIBLE_DEVICES"] = physical_device
    os.environ["MIXNLQ_DECODE_BACKEND"] = args.mixnlq_decode_backend
    os.environ["INT4_DECODE_BACKEND"] = args.mixnlq_decode_backend
    os.environ["MIXNLQ_CUDA_KERNEL"] = args.mixnlq_cuda_kernel
    if not torch.cuda.is_available():
        raise RuntimeError("This end-to-end benchmark requires CUDA")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("--device must expose exactly one GPU")

    # Let Inductor/cuBLAS use TF32 tensor cores for incidental FP32 matrix
    # multiplications. The model's main Linear operations remain BF16.
    torch.set_float32_matmul_precision("high")
    # Some PyTorch/Triton combinations emit this deprecation warning from
    # Inductor-generated helper code many times. It is not raised by our
    # kernels and does not affect benchmark correctness.
    warnings.filterwarnings(
        "ignore",
        message=(
            r"Logical operators 'and' and 'or' are deprecated for non-scalar "
            r"tensors.*"
        ),
        category=UserWarning,
        module=r"torch\._inductor\.runtime\.triton_helpers",
    )

    device = torch.device("cuda:0")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    quantized_variants = []
    if args.mixnlq_checkpoint is not None:
        quantized_variants.append(
            (
                "mixnlq-int4",
                args.mixnlq_checkpoint,
                "mixnlq",
                False,
                False,
            )
        )
    if args.mixnlq3_checkpoint is not None:
        quantized_variants.append(
            (
                "mixnlq-int3",
                args.mixnlq3_checkpoint,
                "mixnlq3",
                False,
                False,
            )
        )
    if args.lut3_checkpoint is not None:
        quantized_variants.append(
            (
                "LUT3-group",
                args.lut3_checkpoint,
                "lut3",
                False,
                False,
            )
        )
    if args.lut4_checkpoint is not None:
        quantized_variants.append(
            (
                "LUT4-group",
                args.lut4_checkpoint,
                "nf4_lut",
                False,
                False,
            )
        )
    if args.linear_checkpoint is not None:
        quantized_variants.append(
            (
                "linear-int4",
                args.linear_checkpoint,
                "linear",
                False,
                False,
            )
        )
    if args.linear3_checkpoint is not None:
        quantized_variants.append(
            (
                "linear-int3",
                args.linear3_checkpoint,
                "linear3",
                False,
                False,
            )
        )
    if args.spinquant_checkpoint is not None:
        quantized_variants.append(
            (
                "spinquant-linear-int4",
                args.spinquant_checkpoint,
                "linear",
                False,
                True,
            )
        )
    if args.spinquant3_checkpoint is not None:
        quantized_variants.append(
            (
                "spinquant-linear-int3",
                args.spinquant3_checkpoint,
                "linear3",
                False,
                True,
            )
        )
    variants = [
        ("bf16-cublas", None, None, False, False),
        *quantized_variants,
    ]
    results = []
    input_ids = (
        load_wikitext2_input(
            model_path=args.model_path,
            batch_size=args.batch_size,
            prompt_length=args.prompt_length,
            seed=args.seed,
            device=device,
        )
        if args.input_mode in ("wikitext2", "wiki2")
        else None
    )

    for (
        name,
        checkpoint,
        quant_mode,
        custom_bf16,
        spinquant_r4,
    ) in variants:
        # Each variant owns different static input buffers, but Transformers'
        # output-capturing wrapper shares its Dynamo cache across variants.
        # Drop stale tensor-identity guards so later variants are compiled
        # instead of hitting cache_size_limit and falling back to eager.
        dynamo = getattr(torch, "_dynamo", None)
        reset_dynamo = getattr(dynamo, "reset", None)
        if reset_dynamo is not None:
            reset_dynamo()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        allocated_before_model = torch.cuda.memory_allocated()
        print(f"\nLoading {name} model ...")
        model = load_model(
            args.model_path,
            device,
            bf16_checkpoint_path=args.bf16_checkpoint,
            checkpoint_path=checkpoint,
            quant_mode=quant_mode,
            custom_bf16=custom_bf16,
            fuse_projections=not args.disable_projection_fusion,
            attn_implementation=args.attn_implementation,
            spinquant_r4=spinquant_r4,
        )
        torch.cuda.synchronize()
        weight_storage_gib = model_storage_gib(model)
        rotation_storage_gib = (
            spinquant_offline_rotation_storage_gib(model)
            if spinquant_r4
            else 0.0
        )
        loaded_model_gib = (
            torch.cuda.memory_allocated() - allocated_before_model
        ) / (1024**3)
        print(
            f"{name}: persistent weight/buffer storage="
            f"{weight_storage_gib:.2f} GiB, offline R1/R2 artifact="
            f"{rotation_storage_gib:.2f} GiB, loaded CUDA allocation="
            f"{loaded_model_gib:.2f} GiB."
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
            name,
            model,
            input_ids,
            weight_storage_gib,
            rotation_storage_gib,
            loaded_model_gib,
            args.new_tokens,
            args.warmup_runs,
            args.benchmark_runs,
            compile_enabled=not args.disable_compile,
            autotune_split_k_enabled=not args.disable_split_k_autotune,
        )
        results.append(result)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    bf16_e2e_ms = results[0].e2e_ms
    bf16_prefill_tflops = results[0].prefill_effective_tflops
    bf16_decode_tflops = results[0].decode_effective_tflops
    bf16_e2e_tflops = results[0].e2e_effective_tflops
    for result in results:
        result.speedup_vs_bf16 = bf16_e2e_ms / result.e2e_ms
        result.prefill_tflops_vs_bf16 = (
            result.prefill_effective_tflops / bf16_prefill_tflops
        )
        result.decode_tflops_vs_bf16 = (
            result.decode_effective_tflops / bf16_decode_tflops
        )
        result.e2e_tflops_vs_bf16 = (
            result.e2e_effective_tflops / bf16_e2e_tflops
        )
    lut4_result = next(
        (result for result in results if result.name == "LUT4-group"),
        None,
    )
    if lut4_result is not None:
        for result in results:
            result.speedup_vs_lut4 = lut4_result.e2e_ms / result.e2e_ms
    linear_result = next(
        (result for result in results if result.name == "linear-int4"),
        None,
    )
    if linear_result is not None:
        for result in results:
            result.speedup_vs_linear = (
                linear_result.e2e_ms / result.e2e_ms
            )
    lut3_result = next(
        (result for result in results if result.name == "LUT3-group"),
        None,
    )
    if lut3_result is not None:
        for result in results:
            result.speedup_vs_lut3 = lut3_result.e2e_ms / result.e2e_ms
    linear3_result = next(
        (result for result in results if result.name == "linear-int3"),
        None,
    )
    if linear3_result is not None:
        for result in results:
            result.speedup_vs_linear3 = (
                linear3_result.e2e_ms / result.e2e_ms
            )

    print("\nMedian end-to-end results")
    print_results(results)
    print("\nEquivalent model FLOP throughput")
    print_flops_results(results)
    print_core_overheads(results)
    print(
        "\nFLOP scope: projection/MLP/LM-head matmuls plus QK/AV attention; "
        "elementwise operations and packed-weight unpack/LUT work are excluded. "
        "SpinQuant R4 work is also excluded from the numerator but included "
        "in measured time. "
        "Low-bit values therefore report equivalent model TFLOP/s."
    )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump([asdict(result) for result in results], handle, indent=2)
        print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
