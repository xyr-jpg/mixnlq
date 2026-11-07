from __future__ import annotations

import importlib
import os
import statistics
import warnings

import torch


_EXTENSION_LOADED: bool | None = None
_EXTENSION_ERROR: Exception | None = None
_FAKE_REGISTERED = False
_FALLBACK_WARNED = False
_VEC2_GROUP_FALLBACK_WARNED = False
_SPLIT_K_CACHE: dict[tuple, int] = {}
_EXTENSION_MAX_SPLIT_K = 64


def _split_k_candidates(k: int, group_size: int, scalar: bool) -> tuple[int, ...]:
    chunk_size = group_size if scalar else 128
    if not scalar:
        while group_size % chunk_size:
            chunk_size //= 2
    work_units = k // chunk_size
    return tuple(
        split for split in (1, 2, 4, 8, 16, 32, 64)
        if split <= work_units
    )


def _benchmark_decode_call(call) -> float:
    """Measure the complete operator, including reduction, without host launch gaps."""
    for _ in range(2):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    calls_per_graph = 8
    with torch.cuda.graph(graph):
        for _ in range(calls_per_graph):
            output = call()
    # Keep the final captured output alive until all replays finish.
    graph.replay()
    graph.replay()
    samples = []
    for _ in range(3):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(8):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / (8 * calls_per_graph))
    del output
    return statistics.median(samples)


@torch.inference_mode()
def autotune_cuda_decode_split_k(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scale_0: torch.Tensor,
    scale_1: torch.Tensor,
    scale_2: torch.Tensor,
    group_lut: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    quant_mode: str,
) -> int:
    """Tune once per GPU/format/shape in this process, before compile or capture."""
    if not _load_extension():
        raise RuntimeError(f"mixnlq_cuda_ops is unavailable: {_EXTENSION_ERROR}")
    if _EXTENSION_MAX_SPLIT_K < 64:
        raise RuntimeError(
            "The loaded mixnlq_cuda_ops extension does not support Split-K up to 64. "
            "Rebuild it with `python setup_cuda.py build_ext --inplace` "
            "and restart Python."
        )
    with torch.cuda.device(x.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Split-K tuning must run before CUDA Graph capture")
        ops = torch.ops.mixnlq_cuda
        variant = _mixnlq_kernel_variant() if quant_mode == "mixnlq" else 0
        scalar = quant_mode == "mixnlq" and (
            variant == 1 or not cuda_tiled_group_supported(group_size)
        )
        if scalar:
            variant = 1
        if quant_mode == "mixnlq":
            op = ops.decode
            args = (x, packed_weight, scale_0, scale_1, bias, group_size)
            suffix = (variant,)
            n = scale_0.shape[1]
        elif quant_mode == "mixnlq3":
            op = ops.decode3
            args = (x, packed_weight, scale_0, scale_1, scale_2, bias, group_size)
            suffix = ()
            n = scale_0.shape[1]
        elif quant_mode in {"linear", "linear3"}:
            op = ops.linear_decode if quant_mode == "linear" else ops.linear_decode3
            args = (x, packed_weight, scale_0, scale_1, bias, group_size)
            suffix = ()
            n = scale_0.shape[1]
        elif quant_mode in {"nf4_lut", "lut3"}:
            op = ops.lut_decode if quant_mode == "nf4_lut" else ops.lut_decode3
            args = (x, packed_weight, group_lut, bias, group_size)
            suffix = ()
            n = group_lut.shape[1 if quant_mode == "nf4_lut" else 2]
        else:
            raise ValueError(f"No native CUDA Split-K tuning for {quant_mode!r}")

        properties = torch.cuda.get_device_properties(x.device)
        key = (
            torch.cuda.current_device(),
            str(getattr(properties, "uuid", "")),
            properties.name, properties.major, properties.minor,
            properties.multi_processor_count,
            torch.__version__, torch.version.cuda,
            quant_mode, tuple(x.shape), str(x.dtype), tuple(packed_weight.shape),
            group_size, n, bias is not None, variant,
        )
        if key not in _SPLIT_K_CACHE:
            timings = {}
            for split in _split_k_candidates(x.shape[1], group_size, scalar):
                timings[split] = _benchmark_decode_call(
                    lambda split=split: op(*args, split, *suffix)
                )
            best = min(timings, key=timings.get)
            _SPLIT_K_CACHE[key] = best
            print(
                f"CUDA Split-K tuned: {properties.name}, {quant_mode}, "
                f"K={x.shape[1]}, N={n}, group_size={group_size}, "
                f"split_k={best} ({timings[best] * 1000:.2f} us)"
            )
        return _SPLIT_K_CACHE[key]


def cuda_tiled_group_supported(group_size: int) -> bool:
    """Whether the native tiled kernels can preserve quant-group boundaries."""
    return group_size >= 16 and group_size % 16 == 0


def _mixnlq_kernel_variant() -> int:
    variant = os.environ.get("MIXNLQ_CUDA_KERNEL", "auto").lower()
    aliases = {
        "auto": 0,
        "scalar": 1,
        "legacy": 1,
        "vec2": 2,
    }
    if variant not in aliases:
        raise ValueError(
            "MIXNLQ_CUDA_KERNEL must be auto, scalar, legacy, or vec2"
        )
    return aliases[variant]


def _register_fake() -> None:
    global _FAKE_REGISTERED
    if _FAKE_REGISTERED:
        return
    register_fake = getattr(torch.library, "register_fake", None)
    if register_fake is None:
        return

    try:
        @register_fake("mixnlq_cuda::decode")
        def _mixnlq_decode_fake(
            x: torch.Tensor,
            packed_weight: torch.Tensor,
            linear_scale: torch.Tensor,
            exp_scale: torch.Tensor,
            bias: torch.Tensor | None,
            group_size: int,
            split_k: int = 0,
            kernel_variant: int = 0,
        ) -> torch.Tensor:
            n = linear_scale.shape[1]
            del (
                exp_scale,
                bias,
                group_size,
                split_k,
                kernel_variant,
            )
            return x.new_empty((1, n))

        @register_fake("mixnlq_cuda::decode3")
        def _mixnlq3_decode_fake(
            x: torch.Tensor,
            packed_weight: torch.Tensor,
            linear_scale: torch.Tensor,
            exp_scale: torch.Tensor,
            quant_bias: torch.Tensor,
            bias: torch.Tensor | None,
            group_size: int,
            split_k: int = 0,
        ) -> torch.Tensor:
            n = linear_scale.shape[1]
            del (
                packed_weight,
                exp_scale,
                quant_bias,
                bias,
                group_size,
                split_k,
            )
            return x.new_empty((1, n))

        @register_fake("mixnlq_cuda::linear_decode")
        def _linear_decode_fake(
            x: torch.Tensor,
            packed_weight: torch.Tensor,
            scale: torch.Tensor,
            zero_point: torch.Tensor,
            bias: torch.Tensor | None,
            group_size: int,
            split_k: int = 0,
        ) -> torch.Tensor:
            n = scale.shape[1]
            del zero_point, bias, group_size, split_k
            return x.new_empty((1, n))

        @register_fake("mixnlq_cuda::linear_decode3")
        def _linear3_decode_fake(
            x: torch.Tensor,
            packed_weight: torch.Tensor,
            scale: torch.Tensor,
            zero_point: torch.Tensor,
            bias: torch.Tensor | None,
            group_size: int,
            split_k: int = 0,
        ) -> torch.Tensor:
            n = scale.shape[1]
            del packed_weight, zero_point, bias, group_size, split_k
            return x.new_empty((1, n))

        @register_fake("mixnlq_cuda::lut_decode3")
        def _lut3_decode_fake(
            x: torch.Tensor,
            packed_weight: torch.Tensor,
            group_lut: torch.Tensor,
            bias: torch.Tensor | None,
            group_size: int,
            split_k: int = 0,
        ) -> torch.Tensor:
            n = group_lut.shape[2]
            del packed_weight, bias, group_size, split_k
            return x.new_empty((1, n))

        @register_fake("mixnlq_cuda::lut_decode")
        def _lut_decode_fake(
            x: torch.Tensor,
            packed_weight: torch.Tensor,
            group_lut: torch.Tensor,
            bias: torch.Tensor | None,
            group_size: int,
            split_k: int = 0,
        ) -> torch.Tensor:
            n = group_lut.shape[1]
            del bias, group_size, split_k
            return x.new_empty((1, n))
    except RuntimeError as exc:
        if "already has" not in str(exc) and "already registered" not in str(exc):
            raise
    _FAKE_REGISTERED = True


def _load_extension() -> bool:
    global _EXTENSION_LOADED, _EXTENSION_ERROR, _EXTENSION_MAX_SPLIT_K
    if _EXTENSION_LOADED is not None:
        return _EXTENSION_LOADED
    try:
        extension = importlib.import_module("mixnlq_cuda_ops")
        _EXTENSION_MAX_SPLIT_K = int(getattr(extension, "MAX_SPLIT_K", 16))
        _register_fake()
        _EXTENSION_LOADED = True
    except (ImportError, OSError, RuntimeError) as exc:
        _EXTENSION_ERROR = exc
        _EXTENSION_LOADED = False
    return _EXTENSION_LOADED


def cuda_decode_enabled() -> bool:
    """Resolve the packed-weight Decode backend once layers are built."""
    global _FALLBACK_WARNED
    backend = os.environ.get(
        "INT4_DECODE_BACKEND",
        os.environ.get("MIXNLQ_DECODE_BACKEND", "auto"),
    ).lower()
    if backend not in {"auto", "cuda", "triton"}:
        raise ValueError(
            "INT4_DECODE_BACKEND must be auto, cuda, or triton"
        )
    if backend == "triton":
        return False
    loaded = _load_extension()
    if backend == "cuda" and not loaded:
        raise RuntimeError(
            "MIXNLQ CUDA Decode was requested but mixnlq_cuda_ops could not "
            f"be loaded: {_EXTENSION_ERROR}. Build it with "
            "`python setup_cuda.py build_ext --inplace`."
        )
    if backend == "auto" and not loaded and not _FALLBACK_WARNED:
        warnings.warn(
            "mixnlq_cuda_ops is unavailable; falling back to Triton "
            "MixNLQ/linear/LUT packed-weight Decode. Build it with "
            "`python setup_cuda.py build_ext --inplace`.",
            RuntimeWarning,
            stacklevel=2,
        )
        _FALLBACK_WARNED = True
    return loaded


def mixnlq_cuda_decode(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    linear_scale: torch.Tensor,
    exp_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    split_k: int = 0,
    kernel_variant: int | None = None,
) -> torch.Tensor:
    global _VEC2_GROUP_FALLBACK_WARNED
    if not _load_extension():
        raise RuntimeError(f"mixnlq_cuda_ops is unavailable: {_EXTENSION_ERROR}")
    if kernel_variant is None:
        kernel_variant = _mixnlq_kernel_variant()
    if kernel_variant == 2 and not cuda_tiled_group_supported(group_size):
        kernel_variant = 1
        if not _VEC2_GROUP_FALLBACK_WARNED:
            warnings.warn(
                "MixNLQ vec2 Decode requires group_size divisible by 16; "
                "falling back "
                f"to scalar CUDA for effective group_size={group_size}.",
                RuntimeWarning,
                stacklevel=2,
            )
            _VEC2_GROUP_FALLBACK_WARNED = True
    return torch.ops.mixnlq_cuda.decode(
        x,
        packed_weight,
        linear_scale,
        exp_scale,
        bias,
        group_size,
        split_k,
        kernel_variant,
    )


def mixnlq3_cuda_decode(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    linear_scale: torch.Tensor,
    exp_scale: torch.Tensor,
    quant_bias: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    split_k: int = 0,
) -> torch.Tensor:
    if not _load_extension():
        raise RuntimeError(f"mixnlq_cuda_ops is unavailable: {_EXTENSION_ERROR}")
    return torch.ops.mixnlq_cuda.decode3(
        x,
        packed_weight,
        linear_scale,
        exp_scale,
        quant_bias,
        bias,
        group_size,
        split_k,
    )


def linear_cuda_decode(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    split_k: int = 0,
) -> torch.Tensor:
    if not _load_extension():
        raise RuntimeError(f"mixnlq_cuda_ops is unavailable: {_EXTENSION_ERROR}")
    return torch.ops.mixnlq_cuda.linear_decode(
        x,
        packed_weight,
        scale,
        zero_point,
        bias,
        group_size,
        split_k,
    )

def linear3_cuda_decode(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    split_k: int = 0,
) -> torch.Tensor:
    if not _load_extension():
        raise RuntimeError(f"mixnlq_cuda_ops is unavailable: {_EXTENSION_ERROR}")
    return torch.ops.mixnlq_cuda.linear_decode3(
        x, packed_weight, scale, zero_point, bias, group_size, split_k
    )


def lut3_cuda_decode(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    group_lut: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    split_k: int = 0,
) -> torch.Tensor:
    if not _load_extension():
        raise RuntimeError(f"mixnlq_cuda_ops is unavailable: {_EXTENSION_ERROR}")
    return torch.ops.mixnlq_cuda.lut_decode3(
        x, packed_weight, group_lut, bias, group_size, split_k
    )


def lut_cuda_decode(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    group_lut: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    split_k: int = 0,
) -> torch.Tensor:
    if not _load_extension():
        raise RuntimeError(f"mixnlq_cuda_ops is unavailable: {_EXTENSION_ERROR}")
    return torch.ops.mixnlq_cuda.lut_decode(
        x,
        packed_weight,
        group_lut,
        bias,
        group_size,
        split_k,
    )
