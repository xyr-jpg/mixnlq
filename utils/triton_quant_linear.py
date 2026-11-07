from typing import Literal

import torch
from torch import nn
import triton
import triton.language as tl
from utils.cuda_mixnlq import (
    autotune_cuda_decode_split_k,
    cuda_tiled_group_supported,
    cuda_decode_enabled,
    linear_cuda_decode,
    linear3_cuda_decode,
    lut_cuda_decode,
    lut3_cuda_decode,
    mixnlq3_cuda_decode,
    mixnlq_cuda_decode,
)

try:
    from torch.library import triton_op, wrap_triton
except ImportError as exc:
    raise ImportError(
        "This implementation requires a PyTorch release that provides "
        "torch.library.triton_op and torch.library.wrap_triton. Upgrade PyTorch "
        "or use a matching recent nightly build."
    ) from exc


QuantMode = Literal[
    "mixnlq", "mixnlq3", "linear", "linear3",
    "nf4", "nf4_lut", "lut3",
]
DECODE_SPLIT_K = 8

NF4_CODEBOOK_VALUES = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)
NF3_CODEBOOK_VALUES = (
    -1.0,
    -0.5350227355957031,
    -0.24693140387535095,
    0.0,
    0.1833375245332718,
    0.38199394941329956,
    0.6229856610298157,
    1.0,
)

MIXNLQ_N_TILE = 64
MIXNLQ_PACKED_K_TILE = 8
TILED_QUANT_MODES = frozenset({"mixnlq", "linear", "nf4_lut"})
TILED_PACKED_LAYOUT = "n64-kp8"
INT3_TILED_PACKED_LAYOUT = "int3-n256-k128-segment-v2"
INT3_THREAD_PACKED_LAYOUT = "int3-n256-k128-v1"
INT3_LEGACY_PACKED_LAYOUT = "bitplane-k8-plane-major"


def _resolve_group_size(group_size: int, in_features: int) -> int:
    """Map checkpoint group_size=-1 to one K-wide group per output channel."""
    if group_size == -1:
        return in_features
    if group_size <= 0 or group_size % 32:
        raise ValueError(
            "group_size must be -1 or a positive multiple of 32"
        )
    if in_features % group_size:
        raise ValueError(
            f"{in_features=} must be divisible by {group_size=}"
        )
    return group_size


@triton.jit
def _mixnlq_tiled_packed_offsets(
    offs_k,
    offs_n,
    N: tl.constexpr,
):
    packed_k = offs_k // 2
    # Keep these literals inside the JIT function. Older Triton releases
    # reject ordinary Python globals referenced from a @triton.jit function.
    n_tiles = N // 64
    packed_k_tile = packed_k // 8
    packed_k_inner = packed_k % 8
    n_tile = offs_n // 64
    n_inner = offs_n % 64
    lane = n_inner // 2
    pair = n_inner % 2
    return (
        (
            (
                (packed_k_tile * n_tiles + n_tile) * 32 + lane
            )
            * 8
            + packed_k_inner
        )
        * 2
        + pair
    )


@triton.jit
def _int3_tiled_packed_offsets(
    offs_k,
    offs_n,
    PLANE: tl.constexpr,
    N: tl.constexpr,
):
    packed_k = offs_k // 8
    packed_k_tile = packed_k // 16
    packed_k_inner = packed_k % 16
    packed_k_half = packed_k_inner // 8
    packed_k_half_inner = packed_k_inner % 8
    n_tile = offs_n // 256
    n_inner = offs_n % 256
    output_pair = n_inner // 2
    pair_lane = n_inner % 2
    n_tiles = N // 256
    return (
        (
            (
                (
                    (
                        (
                            packed_k_tile * n_tiles + n_tile
                        ) * 3 + PLANE
                    ) * 2 + packed_k_half
                ) * 128 + output_pair
            ) * 16 + packed_k_half_inner * 2
        ) + pair_lane
    )


@triton.jit
def _mixnlq_dequantize_kmajor(
    packed_ptr,
    linear_scale_ptr,
    exp_scale_ptr,
    output_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    PER_CHANNEL: tl.constexpr,
    BLOCK_PACKED_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    if PER_CHANNEL:
        group_id = 0
        group_packed_begin = tl.program_id(0) * BLOCK_PACKED_K
        group_packed_end = K // 2
    else:
        group_id = tl.program_id(0)
        group_packed_begin = group_id * (GROUP_SIZE // 2)
        group_packed_end = group_packed_begin + GROUP_SIZE // 2
    offs_packed_k = group_packed_begin + tl.arange(0, BLOCK_PACKED_K)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    valid_packed_k = (
        (offs_packed_k < group_packed_end)
        & (offs_packed_k < K // 2)
    )
    mask = valid_packed_k[:, None] & (offs_n[None, :] < N)
    # Pass the even logical K coordinate to the tiled-offset helper. Both
    # nibbles are then decoded from this single packed-byte load.
    offs_k_low = offs_packed_k * 2
    packed = tl.load(
        packed_ptr
        + _mixnlq_tiled_packed_offsets(
            offs_k_low[:, None], offs_n[None, :], N
        ),
        mask=mask,
        other=0,
    )
    # One scale vector is shared by every packed byte in this quantization
    # group instead of being loaded once for every logical K element.
    scale_offsets = group_id * N + offs_n
    linear_scale = tl.load(
        linear_scale_ptr + scale_offsets,
        mask=offs_n < N,
        other=0.0,
    ).to(tl.float32)
    exp_scale = tl.load(
        exp_scale_ptr + scale_offsets,
        mask=offs_n < N,
        other=0.0,
    ).to(tl.float32)

    nibble_low = (packed & 0x0F).to(tl.int32)
    index_low = nibble_low & 0x07
    sign_low = tl.where((nibble_low & 0x08) != 0, -1.0, 1.0)
    weight_low = sign_low * (
        linear_scale[None, :] * index_low.to(tl.float32)
        + exp_scale[None, :] * (1 << index_low).to(tl.float32)
    )
    tl.store(
        output_ptr + offs_k_low[:, None] * N + offs_n[None, :],
        weight_low,
        mask=mask,
    )

    nibble_high = ((packed >> 4) & 0x0F).to(tl.int32)
    index_high = nibble_high & 0x07
    sign_high = tl.where((nibble_high & 0x08) != 0, -1.0, 1.0)
    weight_high = sign_high * (
        linear_scale[None, :] * index_high.to(tl.float32)
        + exp_scale[None, :] * (1 << index_high).to(tl.float32)
    )
    tl.store(
        output_ptr + (offs_k_low[:, None] + 1) * N + offs_n[None, :],
        weight_high,
        mask=mask,
    )


@triton.jit
def _mixnlq3_dequantize(
    packed_ptr,
    linear_scale_ptr,
    exp_scale_ptr,
    quant_bias_ptr,
    output_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_PACKED_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs_packed_k = (
        tl.program_id(0) * BLOCK_PACKED_K
        + tl.arange(0, BLOCK_PACKED_K)
    )
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_packed_k[:, None] < K // 8) & (
        offs_n[None, :] < N
    )
    logical_k = offs_packed_k[:, None] * 8
    index_low = tl.load(
        packed_ptr + _int3_tiled_packed_offsets(
            logical_k, offs_n[None, :], 0, N
        ),
        mask=mask, other=0
    ).to(tl.int32)
    index_high = tl.load(
        packed_ptr + _int3_tiled_packed_offsets(
            logical_k, offs_n[None, :], 1, N
        ),
        mask=mask, other=0
    ).to(tl.int32)
    signs = tl.load(
        packed_ptr + _int3_tiled_packed_offsets(
            logical_k, offs_n[None, :], 2, N
        ),
        mask=mask, other=0
    ).to(tl.int32)
    group = (offs_packed_k[:, None] * 8) // GROUP_SIZE
    scale_offsets = group * N + offs_n[None, :]
    linear_scale = tl.load(
        linear_scale_ptr + scale_offsets, mask=mask, other=0.0
    ).to(tl.float32)
    exp_scale = tl.load(
        exp_scale_ptr + scale_offsets, mask=mask, other=0.0
    ).to(tl.float32)
    quant_bias = tl.load(
        quant_bias_ptr + scale_offsets, mask=mask, other=0.0
    ).to(tl.float32)
    # One packed/scale load reconstructs eight consecutive K values.
    for packed_bit in range(8):
        index = (
            (index_low >> packed_bit) & 1
        ) | (((index_high >> packed_bit) & 1) << 1)
        sign = tl.where(
            ((signs >> packed_bit) & 1) != 0,
            -1.0,
            1.0,
        )
        index_f32 = index.to(tl.float32)
        weight = sign * (
            linear_scale * index_f32
            + exp_scale * (1 << index).to(tl.float32)
            + quant_bias
        )
        offs_k = offs_packed_k[:, None] * 8 + packed_bit
        tl.store(
            output_ptr + offs_k * N + offs_n[None, :],
            weight,
            mask=mask,
        )

@triton.jit
def _linear3_dequantize(
    packed_ptr,
    scale_ptr,
    zero_point_ptr,
    output_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_PACKED_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs_packed_k = (
        tl.program_id(0) * BLOCK_PACKED_K
        + tl.arange(0, BLOCK_PACKED_K)
    )
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_packed_k[:, None] < K // 8) & (
        offs_n[None, :] < N
    )
    logical_k = offs_packed_k[:, None] * 8
    bit0 = tl.load(
        packed_ptr + _int3_tiled_packed_offsets(
            logical_k, offs_n[None, :], 0, N
        ), mask=mask, other=0
    ).to(tl.int32)
    bit1 = tl.load(
        packed_ptr + _int3_tiled_packed_offsets(
            logical_k, offs_n[None, :], 1, N
        ), mask=mask, other=0
    ).to(tl.int32)
    bit2 = tl.load(
        packed_ptr + _int3_tiled_packed_offsets(
            logical_k, offs_n[None, :], 2, N
        ), mask=mask, other=0
    ).to(tl.int32)
    group = (offs_packed_k[:, None] * 8) // GROUP_SIZE
    parameter_offsets = group * N + offs_n[None, :]
    scale = tl.load(
        scale_ptr + parameter_offsets, mask=mask, other=0.0
    ).to(tl.float32)
    zero_point = tl.load(
        zero_point_ptr + parameter_offsets, mask=mask, other=0.0
    ).to(tl.float32)
    for packed_bit in range(8):
        code = (
            ((bit0 >> packed_bit) & 1)
            | (((bit1 >> packed_bit) & 1) << 1)
            | (((bit2 >> packed_bit) & 1) << 2)
        ).to(tl.float32)
        offs_k = offs_packed_k[:, None] * 8 + packed_bit
        tl.store(
            output_ptr + offs_k * N + offs_n[None, :],
            scale * (code - zero_point),
            mask=mask,
        )


@triton.jit
def _lut3_dequantize(
    packed_ptr,
    group_lut_ptr,
    output_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_PACKED_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs_packed_k = (
        tl.program_id(0) * BLOCK_PACKED_K
        + tl.arange(0, BLOCK_PACKED_K)
    )
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_packed_k[:, None] < K // 8) & (
        offs_n[None, :] < N
    )
    logical_k = offs_packed_k[:, None] * 8
    bit0 = tl.load(
        packed_ptr + _int3_tiled_packed_offsets(
            logical_k, offs_n[None, :], 0, N
        ), mask=mask, other=0
    ).to(tl.int32)
    bit1 = tl.load(
        packed_ptr + _int3_tiled_packed_offsets(
            logical_k, offs_n[None, :], 1, N
        ), mask=mask, other=0
    ).to(tl.int32)
    bit2 = tl.load(
        packed_ptr + _int3_tiled_packed_offsets(
            logical_k, offs_n[None, :], 2, N
        ), mask=mask, other=0
    ).to(tl.int32)
    group = (offs_packed_k[:, None] * 8) // GROUP_SIZE
    for packed_bit in range(8):
        code = (
            ((bit0 >> packed_bit) & 1)
            | (((bit1 >> packed_bit) & 1) << 1)
            | (((bit2 >> packed_bit) & 1) << 2)
        )
        lut_offsets = (group * 8 + code) * N + offs_n[None, :]
        weight = tl.load(
            group_lut_ptr + lut_offsets, mask=mask, other=0.0
        )
        offs_k = offs_packed_k[:, None] * 8 + packed_bit
        tl.store(
            output_ptr + offs_k * N + offs_n[None, :],
            weight,
            mask=mask,
        )


@triton.jit
def _linear_dequantize_kmajor(
    packed_ptr,
    scale_ptr,
    zero_point_ptr,
    output_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    PER_CHANNEL: tl.constexpr,
    BLOCK_PACKED_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    if PER_CHANNEL:
        group_id = 0
        group_packed_begin = tl.program_id(0) * BLOCK_PACKED_K
        group_packed_end = K // 2
    else:
        group_id = tl.program_id(0)
        group_packed_begin = group_id * (GROUP_SIZE // 2)
        group_packed_end = group_packed_begin + GROUP_SIZE // 2
    offs_packed_k = group_packed_begin + tl.arange(0, BLOCK_PACKED_K)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    valid_packed_k = (
        (offs_packed_k < group_packed_end)
        & (offs_packed_k < K // 2)
    )
    mask = valid_packed_k[:, None] & (offs_n[None, :] < N)
    offs_k_low = offs_packed_k * 2
    packed = tl.load(
        packed_ptr
        + _mixnlq_tiled_packed_offsets(
            offs_k_low[:, None], offs_n[None, :], N
        ),
        mask=mask,
        other=0,
    )
    scale_offsets = group_id * N + offs_n
    scale = tl.load(
        scale_ptr + scale_offsets, mask=offs_n < N, other=0.0
    ).to(tl.float32)
    zero_point = tl.load(
        zero_point_ptr + scale_offsets, mask=offs_n < N, other=0.0
    ).to(tl.float32)

    q_low = (packed & 0x0F).to(tl.float32)
    weight_low = scale[None, :] * (q_low - zero_point[None, :])
    tl.store(
        output_ptr + offs_k_low[:, None] * N + offs_n[None, :],
        weight_low,
        mask=mask,
    )

    q_high = ((packed >> 4) & 0x0F).to(tl.float32)
    weight_high = scale[None, :] * (q_high - zero_point[None, :])
    tl.store(
        output_ptr + (offs_k_low[:, None] + 1) * N + offs_n[None, :],
        weight_high,
        mask=mask,
    )


@triton.jit
def _nf4_group_lut_dequantize(
    packed_ptr,
    group_lut_ptr,
    output_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    PER_CHANNEL: tl.constexpr,
    BLOCK_PACKED_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    if PER_CHANNEL:
        group_id = 0
        group_packed_begin = tl.program_id(0) * BLOCK_PACKED_K
        group_packed_end = K // 2
    else:
        group_id = tl.program_id(0)
        group_packed_begin = group_id * (GROUP_SIZE // 2)
        group_packed_end = group_packed_begin + GROUP_SIZE // 2
    offs_packed_k = group_packed_begin + tl.arange(0, BLOCK_PACKED_K)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    valid_packed_k = (
        (offs_packed_k < group_packed_end)
        & (offs_packed_k < K // 2)
    )
    mask = valid_packed_k[:, None] & (offs_n[None, :] < N)
    offs_k_low = offs_packed_k * 2
    packed = tl.load(
        packed_ptr
        + _mixnlq_tiled_packed_offsets(
            offs_k_low[:, None], offs_n[None, :], N
        ),
        mask=mask,
        other=0,
    )
    lut_base = (group_id * N + offs_n[None, :]) * 16

    code_low = (packed & 0x0F).to(tl.int32)
    weight_low = tl.load(
        group_lut_ptr + lut_base + code_low,
        mask=mask,
        other=0.0,
    )
    tl.store(
        output_ptr + offs_k_low[:, None] * N + offs_n[None, :],
        weight_low,
        mask=mask,
    )

    code_high = ((packed >> 4) & 0x0F).to(tl.int32)
    weight_high = tl.load(
        group_lut_ptr + lut_base + code_high,
        mask=mask,
        other=0.0,
    )
    tl.store(
        output_ptr + (offs_k_low[:, None] + 1) * N + offs_n[None, :],
        weight_high,
        mask=mask,
    )


_PREFILL_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
        num_warps=8,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
        num_warps=8,
        num_stages=4,
    ),
]


@triton.autotune(
    configs=_PREFILL_CONFIGS,
    key=["M", "N", "K", "GROUP_SIZE"],
)
@triton.jit
def _mixnlq_int4_gemm(
    x_ptr,
    packed_ptr,
    linear_scale_ptr,
    exp_scale_ptr,
    bias_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    GROUP_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        valid_k = offs_k < K
        weight_mask = valid_k[:, None] & (offs_n[None, :] < N)
        x = tl.load(
            x_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=(offs_m[:, None] < M) & valid_k[None, :],
            other=0.0,
        )
        packed = tl.load(
            packed_ptr
            + _mixnlq_tiled_packed_offsets(
                offs_k[:, None], offs_n[None, :], N
            ),
            mask=weight_mask,
            other=0,
        )
        nibble = tl.where(
            (offs_k[:, None] & 1) == 0,
            packed & 0x0F,
            (packed >> 4) & 0x0F,
        ).to(tl.int32)
        scale_offsets = (k_start // GROUP_SIZE) * N + offs_n
        linear_scale = tl.load(
            linear_scale_ptr + scale_offsets, mask=offs_n < N, other=0.0
        ).to(tl.float32)
        exp_scale = tl.load(
            exp_scale_ptr + scale_offsets, mask=offs_n < N, other=0.0
        ).to(tl.float32)
        index = nibble & 0x07
        sign = tl.where((nibble & 0x08) != 0, -1.0, 1.0)
        weight = sign * (
            linear_scale[None, :] * index.to(tl.float32)
            + exp_scale[None, :] * (1 << index).to(tl.float32)
        )
        accumulator += tl.dot(x, weight.to(tl.bfloat16))
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        accumulator += bias[None, :]
    tl.store(
        output_ptr + offs_m[:, None] * N + offs_n[None, :],
        accumulator,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.autotune(
    configs=_PREFILL_CONFIGS,
    key=["M", "N", "K", "GROUP_SIZE"],
)
@triton.jit
def _linear_int4_gemm(
    x_ptr,
    packed_ptr,
    scale_ptr,
    zero_point_ptr,
    bias_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    GROUP_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        valid_k = offs_k < K
        weight_mask = valid_k[:, None] & (offs_n[None, :] < N)
        x = tl.load(
            x_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=(offs_m[:, None] < M) & valid_k[None, :],
            other=0.0,
        )
        packed = tl.load(
            packed_ptr
            + _mixnlq_tiled_packed_offsets(
                offs_k[:, None], offs_n[None, :], N
            ),
            mask=weight_mask,
            other=0,
        )
        q = tl.where(
            (offs_k[:, None] & 1) == 0,
            packed & 0x0F,
            (packed >> 4) & 0x0F,
        ).to(tl.float32)
        scale_offsets = (k_start // GROUP_SIZE) * N + offs_n
        scale = tl.load(
            scale_ptr + scale_offsets, mask=offs_n < N, other=0.0
        ).to(tl.float32)
        zero_point = tl.load(
            zero_point_ptr + scale_offsets, mask=offs_n < N, other=0.0
        ).to(tl.float32)
        weight = scale[None, :] * (q - zero_point[None, :])
        accumulator += tl.dot(x, weight.to(tl.bfloat16))
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        accumulator += bias[None, :]
    tl.store(
        output_ptr + offs_m[:, None] * N + offs_n[None, :],
        accumulator,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.autotune(
    configs=_PREFILL_CONFIGS,
    key=["M", "N", "K", "GROUP_SIZE"],
)
@triton.jit
def _nf4_int4_gemm(
    x_ptr,
    packed_ptr,
    absmax_ptr,
    codebook_ptr,
    bias_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    GROUP_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        valid_k = offs_k < K
        weight_mask = valid_k[:, None] & (offs_n[None, :] < N)
        x = tl.load(
            x_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=(offs_m[:, None] < M) & valid_k[None, :],
            other=0.0,
        )
        packed = tl.load(
            packed_ptr + (offs_k[:, None] // 2) * N + offs_n[None, :],
            mask=weight_mask,
            other=0,
        )
        code = tl.where(
            (offs_k[:, None] & 1) == 0,
            packed & 0x0F,
            (packed >> 4) & 0x0F,
        ).to(tl.int32)
        level = tl.load(codebook_ptr + code, mask=weight_mask, other=0.0)
        scale_offsets = (k_start // GROUP_SIZE) * N + offs_n
        absmax = tl.load(
            absmax_ptr + scale_offsets, mask=offs_n < N, other=0.0
        )
        weight = level.to(tl.float32) * absmax[None, :].to(tl.float32)
        accumulator += tl.dot(x, weight.to(tl.bfloat16))
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        accumulator += bias[None, :]
    tl.store(
        output_ptr + offs_m[:, None] * N + offs_n[None, :],
        accumulator,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.autotune(
    configs=_PREFILL_CONFIGS,
    key=["M", "N", "K", "GROUP_SIZE"],
)
@triton.jit
def _nf4_group_lut_gemm(
    x_ptr,
    packed_ptr,
    group_lut_ptr,
    bias_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    GROUP_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        valid_k = offs_k < K
        weight_mask = valid_k[:, None] & (offs_n[None, :] < N)
        x = tl.load(
            x_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=(offs_m[:, None] < M) & valid_k[None, :],
            other=0.0,
        )
        packed = tl.load(
            packed_ptr
            + _mixnlq_tiled_packed_offsets(
                offs_k[:, None], offs_n[None, :], N
            ),
            mask=weight_mask,
            other=0,
        )
        code = tl.where(
            (offs_k[:, None] & 1) == 0,
            packed & 0x0F,
            (packed >> 4) & 0x0F,
        ).to(tl.int32)
        lut_base = (
            ((k_start // GROUP_SIZE) * N + offs_n[None, :]) * 16
        )
        weight = tl.load(
            group_lut_ptr + lut_base + code,
            mask=weight_mask,
            other=0.0,
        )
        accumulator += tl.dot(x, weight.to(tl.bfloat16))
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        accumulator += bias[None, :]
    tl.store(
        output_ptr + offs_m[:, None] * N + offs_n[None, :],
        accumulator,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 16}, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
    ],
    key=["N", "K", "GROUP_SIZE", "SPLIT_K"],
)
@triton.jit
def _mixnlq_int4_gemv(
    x_ptr,
    packed_ptr,
    linear_scale_ptr,
    exp_scale_ptr,
    partial_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    split_id = tl.program_id(1)
    k_per_split: tl.constexpr = tl.cdiv(K, SPLIT_K)
    k_begin = split_id * k_per_split
    accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_offset in range(0, k_per_split, BLOCK_K):
        offs_k = k_begin + k_offset + tl.arange(0, BLOCK_K)
        mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        x = tl.load(x_ptr + offs_k, mask=offs_k < K, other=0.0).to(tl.float32)
        packed = tl.load(
            packed_ptr
            + _mixnlq_tiled_packed_offsets(
                offs_k[:, None], offs_n[None, :], N
            ),
            mask=mask,
            other=0,
        )
        nibble = tl.where(
            (offs_k[:, None] & 1) == 0,
            packed & 0x0F,
            (packed >> 4) & 0x0F,
        ).to(tl.int32)
        scale_offsets = (offs_k[:, None] // GROUP_SIZE) * N + offs_n[None, :]
        linear_scale = tl.load(
            linear_scale_ptr + scale_offsets, mask=mask, other=0.0
        ).to(tl.float32)
        exp_scale = tl.load(
            exp_scale_ptr + scale_offsets, mask=mask, other=0.0
        ).to(tl.float32)
        index = nibble & 0x07
        sign = tl.where((nibble & 0x08) != 0, -1.0, 1.0)
        weight = sign * (
            linear_scale * index.to(tl.float32)
            + exp_scale * (1 << index).to(tl.float32)
        )
        accumulator += tl.sum(x[:, None] * weight, axis=0)

    tl.store(
        partial_ptr + split_id * N + offs_n,
        accumulator,
        mask=offs_n < N,
    )

@triton.jit
def _linear3_gemv(
    x_ptr,
    packed_ptr,
    scale_ptr,
    zero_point_ptr,
    partial_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    split_id = tl.program_id(1)
    k_per_split: tl.constexpr = tl.cdiv(K, SPLIT_K)
    k_begin = split_id * k_per_split
    accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k_offset in range(0, k_per_split, BLOCK_K):
        offs_k = k_begin + k_offset + tl.arange(0, BLOCK_K)
        mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        x = tl.load(x_ptr + offs_k, mask=offs_k < K, other=0.0).to(tl.float32)
        packed_bit = offs_k[:, None] % 8
        bit0 = tl.load(
            packed_ptr + _int3_tiled_packed_offsets(
                offs_k[:, None], offs_n[None, :], 0, N
            ), mask=mask, other=0
        ).to(tl.int32)
        bit1 = tl.load(
            packed_ptr + _int3_tiled_packed_offsets(
                offs_k[:, None], offs_n[None, :], 1, N
            ), mask=mask, other=0
        ).to(tl.int32)
        bit2 = tl.load(
            packed_ptr + _int3_tiled_packed_offsets(
                offs_k[:, None], offs_n[None, :], 2, N
            ), mask=mask, other=0
        ).to(tl.int32)
        code = (
            ((bit0 >> packed_bit) & 1)
            | (((bit1 >> packed_bit) & 1) << 1)
            | (((bit2 >> packed_bit) & 1) << 2)
        ).to(tl.float32)
        parameter_offsets = (
            offs_k[:, None] // GROUP_SIZE
        ) * N + offs_n[None, :]
        scale = tl.load(
            scale_ptr + parameter_offsets, mask=mask, other=0.0
        ).to(tl.float32)
        zero_point = tl.load(
            zero_point_ptr + parameter_offsets, mask=mask, other=0.0
        ).to(tl.float32)
        accumulator += tl.sum(
            x[:, None] * scale * (code - zero_point),
            axis=0,
        )
    tl.store(
        partial_ptr + split_id * N + offs_n,
        accumulator,
        mask=offs_n < N,
    )


@triton.jit
def _lut3_gemv(
    x_ptr,
    packed_ptr,
    group_lut_ptr,
    partial_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    split_id = tl.program_id(1)
    k_per_split: tl.constexpr = tl.cdiv(K, SPLIT_K)
    k_begin = split_id * k_per_split
    accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k_offset in range(0, k_per_split, BLOCK_K):
        offs_k = k_begin + k_offset + tl.arange(0, BLOCK_K)
        mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        x = tl.load(x_ptr + offs_k, mask=offs_k < K, other=0.0).to(tl.float32)
        packed_bit = offs_k[:, None] % 8
        bit0 = tl.load(
            packed_ptr + _int3_tiled_packed_offsets(
                offs_k[:, None], offs_n[None, :], 0, N
            ), mask=mask, other=0
        ).to(tl.int32)
        bit1 = tl.load(
            packed_ptr + _int3_tiled_packed_offsets(
                offs_k[:, None], offs_n[None, :], 1, N
            ), mask=mask, other=0
        ).to(tl.int32)
        bit2 = tl.load(
            packed_ptr + _int3_tiled_packed_offsets(
                offs_k[:, None], offs_n[None, :], 2, N
            ), mask=mask, other=0
        ).to(tl.int32)
        code = (
            ((bit0 >> packed_bit) & 1)
            | (((bit1 >> packed_bit) & 1) << 1)
            | (((bit2 >> packed_bit) & 1) << 2)
        )
        group = offs_k[:, None] // GROUP_SIZE
        lut_offsets = (group * 8 + code) * N + offs_n[None, :]
        weight = tl.load(
            group_lut_ptr + lut_offsets, mask=mask, other=0.0
        ).to(tl.float32)
        accumulator += tl.sum(x[:, None] * weight, axis=0)
    tl.store(
        partial_ptr + split_id * N + offs_n,
        accumulator,
        mask=offs_n < N,
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4),
    ],
    key=["N", "K", "GROUP_SIZE", "SPLIT_K"],
)
@triton.jit
def _mixnlq3_gemv(
    x_ptr,
    packed_ptr,
    linear_scale_ptr,
    exp_scale_ptr,
    quant_bias_ptr,
    partial_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    split_id = tl.program_id(1)
    k_per_split: tl.constexpr = tl.cdiv(K, SPLIT_K)
    k_begin = split_id * k_per_split
    accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_offset in range(0, k_per_split, BLOCK_K):
        offs_k = k_begin + k_offset + tl.arange(0, BLOCK_K)
        mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        x = tl.load(
            x_ptr + offs_k,
            mask=offs_k < K,
            other=0.0,
        ).to(tl.float32)
        packed_bit = offs_k[:, None] % 8
        index_low = tl.load(
            packed_ptr + _int3_tiled_packed_offsets(
                offs_k[:, None], offs_n[None, :], 0, N
            ), mask=mask, other=0
        ).to(tl.int32)
        index_high = tl.load(
            packed_ptr + _int3_tiled_packed_offsets(
                offs_k[:, None], offs_n[None, :], 1, N
            ), mask=mask, other=0
        ).to(tl.int32)
        signs = tl.load(
            packed_ptr + _int3_tiled_packed_offsets(
                offs_k[:, None], offs_n[None, :], 2, N
            ), mask=mask, other=0
        ).to(tl.int32)
        index = (
            (index_low >> packed_bit) & 1
        ) | (((index_high >> packed_bit) & 1) << 1)
        sign = tl.where(
            ((signs >> packed_bit) & 1) != 0,
            -1.0,
            1.0,
        )
        scale_offsets = (
            offs_k[:, None] // GROUP_SIZE
        ) * N + offs_n[None, :]
        linear_scale = tl.load(
            linear_scale_ptr + scale_offsets, mask=mask, other=0.0
        ).to(tl.float32)
        exp_scale = tl.load(
            exp_scale_ptr + scale_offsets, mask=mask, other=0.0
        ).to(tl.float32)
        quant_bias = tl.load(
            quant_bias_ptr + scale_offsets, mask=mask, other=0.0
        ).to(tl.float32)
        index_f32 = index.to(tl.float32)
        weight = sign * (
            linear_scale * index_f32
            + exp_scale * (1 << index).to(tl.float32)
            + quant_bias
        )
        accumulator += tl.sum(x[:, None] * weight, axis=0)

    tl.store(
        partial_ptr + split_id * N + offs_n,
        accumulator,
        mask=offs_n < N,
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 16}, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
    ],
    key=["N", "K", "GROUP_SIZE", "SPLIT_K"],
)
@triton.jit
def _linear_int4_gemv(
    x_ptr,
    packed_ptr,
    scale_ptr,
    zero_point_ptr,
    partial_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    split_id = tl.program_id(1)
    k_per_split: tl.constexpr = tl.cdiv(K, SPLIT_K)
    k_begin = split_id * k_per_split
    accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_offset in range(0, k_per_split, BLOCK_K):
        offs_k = k_begin + k_offset + tl.arange(0, BLOCK_K)
        mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        x = tl.load(x_ptr + offs_k, mask=offs_k < K, other=0.0).to(tl.float32)
        packed = tl.load(
            packed_ptr
            + _mixnlq_tiled_packed_offsets(
                offs_k[:, None], offs_n[None, :], N
            ),
            mask=mask,
            other=0,
        )
        q = tl.where(
            (offs_k[:, None] & 1) == 0,
            packed & 0x0F,
            (packed >> 4) & 0x0F,
        ).to(tl.float32)
        scale_offsets = (offs_k[:, None] // GROUP_SIZE) * N + offs_n[None, :]
        scale = tl.load(scale_ptr + scale_offsets, mask=mask, other=0.0)
        zero_point = tl.load(
            zero_point_ptr + scale_offsets, mask=mask, other=0.0
        )
        weight = scale.to(tl.float32) * (q - zero_point.to(tl.float32))
        accumulator += tl.sum(x[:, None] * weight, axis=0)

    tl.store(
        partial_ptr + split_id * N + offs_n,
        accumulator,
        mask=offs_n < N,
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
    ],
    key=["N", "K", "GROUP_SIZE"],
)
@triton.jit
def _nf4_int4_gemv(
    x_ptr,
    packed_ptr,
    absmax_ptr,
    codebook_ptr,
    bias_ptr,
    output_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k_offset in range(0, K, BLOCK_K):
        offs_k = k_offset + tl.arange(0, BLOCK_K)
        valid_k = offs_k < K
        mask = valid_k[:, None] & (offs_n[None, :] < N)
        x = tl.load(x_ptr + offs_k, mask=valid_k, other=0.0).to(tl.float32)
        packed = tl.load(
            packed_ptr + (offs_k[:, None] // 2) * N + offs_n[None, :],
            mask=mask,
            other=0,
        )
        code = tl.where(
            (offs_k[:, None] & 1) == 0,
            packed & 0x0F,
            (packed >> 4) & 0x0F,
        ).to(tl.int32)
        level = tl.load(codebook_ptr + code, mask=mask, other=0.0)
        scale_offsets = (k_offset // GROUP_SIZE) * N + offs_n
        absmax = tl.load(
            absmax_ptr + scale_offsets, mask=offs_n < N, other=0.0
        )
        weight = level.to(tl.float32) * absmax[None, :].to(tl.float32)
        weight = weight.to(tl.bfloat16).to(tl.float32)
        accumulator += tl.sum(x[:, None] * weight, axis=0)
    if HAS_BIAS:
        accumulator += tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    tl.store(output_ptr + offs_n, accumulator, mask=offs_n < N)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
    ],
    key=["N", "K", "GROUP_SIZE"],
)
@triton.jit
def _nf4_group_lut_gemv(
    x_ptr,
    packed_ptr,
    group_lut_ptr,
    bias_ptr,
    output_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k_offset in range(0, K, BLOCK_K):
        offs_k = k_offset + tl.arange(0, BLOCK_K)
        valid_k = offs_k < K
        mask = valid_k[:, None] & (offs_n[None, :] < N)
        x = tl.load(x_ptr + offs_k, mask=valid_k, other=0.0).to(tl.float32)
        packed = tl.load(
            packed_ptr
            + _mixnlq_tiled_packed_offsets(
                offs_k[:, None], offs_n[None, :], N
            ),
            mask=mask,
            other=0,
        )
        code = tl.where(
            (offs_k[:, None] & 1) == 0,
            packed & 0x0F,
            (packed >> 4) & 0x0F,
        ).to(tl.int32)
        lut_base = (
            (
                (k_offset // GROUP_SIZE) * N
                + offs_n[None, :]
            )
            * 16
        )
        weight = tl.load(
            group_lut_ptr + lut_base + code,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sum(x[:, None] * weight, axis=0)
    if HAS_BIAS:
        accumulator += tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    tl.store(output_ptr + offs_n, accumulator, mask=offs_n < N)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
    ],
    key=["N", "K", "SPLIT_K"],
)
@triton.jit
def _bf16_gemv(
    x_ptr,
    weight_kn_ptr,
    partial_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    split_id = tl.program_id(1)
    k_per_split = (K + SPLIT_K - 1) // SPLIT_K
    k_begin = split_id * k_per_split
    accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k_offset in range(0, k_per_split, BLOCK_K):
        offs_k = k_begin + k_offset + tl.arange(0, BLOCK_K)
        valid_k = (offs_k < K) & (offs_k < k_begin + k_per_split)
        x = tl.load(x_ptr + offs_k, mask=valid_k, other=0.0).to(tl.float32)
        weight = tl.load(
            weight_kn_ptr + offs_k[:, None] * N + offs_n[None, :],
            mask=valid_k[:, None] & (offs_n[None, :] < N),
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sum(x[:, None] * weight, axis=0)
    tl.store(
        partial_ptr + split_id * N + offs_n,
        accumulator,
        mask=offs_n < N,
    )


def _decode_grid(n: int):
    return lambda meta: (triton.cdiv(n, meta["BLOCK_N"]),)


def _decode_splitk_grid(n: int):
    return lambda meta: (
        triton.cdiv(n, meta["BLOCK_N"]),
        DECODE_SPLIT_K,
    )


def _prefill_grid(m: int, n: int):
    return lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )


@triton_op("mixnlq::mixnlq_int4_linear", mutates_args={})
def mixnlq_int4_linear_op(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    linear_scale: torch.Tensor,
    exp_scale: torch.Tensor,
    bias: torch.Tensor,
    group_size: int,
    has_bias: bool,
) -> torch.Tensor:
    m, k = x.shape
    n = linear_scale.shape[1]
    if m == 1:
        partial = torch.empty(
            (DECODE_SPLIT_K, n), dtype=torch.float32, device=x.device
        )
        wrap_triton(_mixnlq_int4_gemv)[_decode_splitk_grid(n)](
            x, packed_weight, linear_scale, exp_scale, partial,
            N=n, K=k, GROUP_SIZE=group_size, SPLIT_K=DECODE_SPLIT_K,
        )
        output = partial.sum(dim=0, keepdim=True).to(dtype=x.dtype)
        return output + bias if has_bias else output
    dequantized = torch.empty((k, n), dtype=x.dtype, device=x.device)
    per_channel = group_size == k
    block_packed_k = (
        64
        if per_channel
        else triton.next_power_of_2(group_size // 2)
    )
    packed_k_programs = (
        triton.cdiv(k // 2, block_packed_k)
        if per_channel
        else triton.cdiv(k, group_size)
    )
    wrap_triton(_mixnlq_dequantize_kmajor)[
        (packed_k_programs, triton.cdiv(n, 64))
    ](
        packed_weight, linear_scale, exp_scale, dequantized,
        N=n, K=k, GROUP_SIZE=group_size,
        PER_CHANNEL=per_channel,
        BLOCK_PACKED_K=block_packed_k, BLOCK_N=64,
    )
    output = torch.mm(x, dequantized)
    return output + bias if has_bias else output


@triton_op("mixnlq::mixnlq3_linear", mutates_args={})
def mixnlq3_linear_op(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    linear_scale: torch.Tensor,
    exp_scale: torch.Tensor,
    quant_bias: torch.Tensor,
    bias: torch.Tensor,
    group_size: int,
    has_bias: bool,
) -> torch.Tensor:
    m, k = x.shape
    n = linear_scale.shape[1]
    if m == 1:
        partial = torch.empty(
            (DECODE_SPLIT_K, n), dtype=torch.float32, device=x.device
        )
        wrap_triton(_mixnlq3_gemv)[_decode_splitk_grid(n)](
            x,
            packed_weight,
            linear_scale,
            exp_scale,
            quant_bias,
            partial,
            N=n,
            K=k,
            GROUP_SIZE=group_size,
            SPLIT_K=DECODE_SPLIT_K,
        )
        output = partial.sum(dim=0, keepdim=True).to(dtype=x.dtype)
        return output + bias if has_bias else output

    dequantized = torch.empty((k, n), dtype=x.dtype, device=x.device)
    wrap_triton(_mixnlq3_dequantize)[
        (triton.cdiv(k // 8, 8), triton.cdiv(n, 64))
    ](
        packed_weight,
        linear_scale,
        exp_scale,
        quant_bias,
        dequantized,
        N=n,
        K=k,
        GROUP_SIZE=group_size,
        BLOCK_PACKED_K=8,
        BLOCK_N=64,
    )
    output = torch.mm(x, dequantized)
    return output + bias if has_bias else output

@triton_op("mixnlq::linear3_linear", mutates_args={})
def linear3_linear_op(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    bias: torch.Tensor,
    group_size: int,
    has_bias: bool,
) -> torch.Tensor:
    m, k = x.shape
    n = scale.shape[1]
    if m == 1:
        partial = torch.empty(
            (DECODE_SPLIT_K, n), dtype=torch.float32, device=x.device
        )
        wrap_triton(_linear3_gemv)[_decode_splitk_grid(n)](
            x, packed_weight, scale, zero_point, partial,
            N=n, K=k, GROUP_SIZE=group_size, SPLIT_K=DECODE_SPLIT_K,
            BLOCK_N=64, BLOCK_K=32,
        )
        output = partial.sum(dim=0, keepdim=True).to(dtype=x.dtype)
        return output + bias if has_bias else output
    dequantized = torch.empty((k, n), dtype=x.dtype, device=x.device)
    wrap_triton(_linear3_dequantize)[
        (triton.cdiv(k // 8, 8), triton.cdiv(n, 64))
    ](
        packed_weight, scale, zero_point, dequantized,
        N=n, K=k, GROUP_SIZE=group_size,
        BLOCK_PACKED_K=8, BLOCK_N=64,
    )
    output = torch.mm(x, dequantized)
    return output + bias if has_bias else output


@triton_op("mixnlq::lut3_linear", mutates_args={})
def lut3_linear_op(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    group_lut: torch.Tensor,
    bias: torch.Tensor,
    group_size: int,
    has_bias: bool,
) -> torch.Tensor:
    m, k = x.shape
    n = group_lut.shape[2]
    if m == 1:
        partial = torch.empty(
            (DECODE_SPLIT_K, n), dtype=torch.float32, device=x.device
        )
        wrap_triton(_lut3_gemv)[_decode_splitk_grid(n)](
            x, packed_weight, group_lut, partial,
            N=n, K=k, GROUP_SIZE=group_size, SPLIT_K=DECODE_SPLIT_K,
            BLOCK_N=64, BLOCK_K=32,
        )
        output = partial.sum(dim=0, keepdim=True).to(dtype=x.dtype)
        return output + bias if has_bias else output
    dequantized = torch.empty((k, n), dtype=x.dtype, device=x.device)
    wrap_triton(_lut3_dequantize)[
        (triton.cdiv(k // 8, 8), triton.cdiv(n, 64))
    ](
        packed_weight, group_lut, dequantized,
        N=n, K=k, GROUP_SIZE=group_size,
        BLOCK_PACKED_K=8, BLOCK_N=64,
    )
    output = torch.mm(x, dequantized)
    return output + bias if has_bias else output


@triton_op("mixnlq::linear_int4_linear", mutates_args={})
def linear_int4_linear_op(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    bias: torch.Tensor,
    group_size: int,
    has_bias: bool,
) -> torch.Tensor:
    m, k = x.shape
    n = scale.shape[1]
    if m == 1:
        partial = torch.empty(
            (DECODE_SPLIT_K, n), dtype=torch.float32, device=x.device
        )
        wrap_triton(_linear_int4_gemv)[_decode_splitk_grid(n)](
            x, packed_weight, scale, zero_point, partial,
            N=n, K=k, GROUP_SIZE=group_size, SPLIT_K=DECODE_SPLIT_K,
        )
        output = partial.sum(dim=0, keepdim=True).to(dtype=x.dtype)
        return output + bias if has_bias else output
    dequantized = torch.empty((k, n), dtype=x.dtype, device=x.device)
    per_channel = group_size == k
    block_packed_k = (
        64
        if per_channel
        else triton.next_power_of_2(group_size // 2)
    )
    packed_k_programs = (
        triton.cdiv(k // 2, block_packed_k)
        if per_channel
        else triton.cdiv(k, group_size)
    )
    wrap_triton(_linear_dequantize_kmajor)[
        (packed_k_programs, triton.cdiv(n, 64))
    ](
        packed_weight, scale, zero_point, dequantized,
        N=n, K=k, GROUP_SIZE=group_size,
        PER_CHANNEL=per_channel,
        BLOCK_PACKED_K=block_packed_k, BLOCK_N=64,
    )
    output = torch.mm(x, dequantized)
    return output + bias if has_bias else output


@triton_op("mixnlq::nf4_int4_linear", mutates_args={})
def nf4_int4_linear_op(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    absmax: torch.Tensor,
    codebook: torch.Tensor,
    bias: torch.Tensor,
    group_size: int,
    has_bias: bool,
) -> torch.Tensor:
    m, k = x.shape
    n = packed_weight.shape[1]
    output = torch.empty((m, n), dtype=x.dtype, device=x.device)
    if m == 1:
        wrap_triton(_nf4_int4_gemv)[_decode_grid(n)](
            x, packed_weight, absmax, codebook, bias, output,
            N=n, K=k, GROUP_SIZE=group_size,
            HAS_BIAS=has_bias,
        )
        return output
    wrap_triton(_nf4_int4_gemm)[_prefill_grid(m, n)](
        x, packed_weight, absmax, codebook, bias, output,
        M=m, N=n, K=k, GROUP_SIZE=group_size, HAS_BIAS=has_bias, GROUP_M=8,
    )
    return output


@triton_op("mixnlq::nf4_group_lut_linear", mutates_args={})
def nf4_group_lut_linear_op(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    group_lut: torch.Tensor,
    bias: torch.Tensor,
    group_size: int,
    has_bias: bool,
) -> torch.Tensor:
    m, k = x.shape
    n = group_lut.shape[1]
    if m == 1:
        output = torch.empty((m, n), dtype=x.dtype, device=x.device)
        wrap_triton(_nf4_group_lut_gemv)[_decode_grid(n)](
            x, packed_weight, group_lut, bias, output,
            N=n, K=k, GROUP_SIZE=group_size,
            HAS_BIAS=has_bias,
        )
        return output
    dequantized = torch.empty((k, n), dtype=x.dtype, device=x.device)
    per_channel = group_size == k
    block_packed_k = (
        64
        if per_channel
        else triton.next_power_of_2(group_size // 2)
    )
    packed_k_programs = (
        triton.cdiv(k // 2, block_packed_k)
        if per_channel
        else triton.cdiv(k, group_size)
    )
    wrap_triton(_nf4_group_lut_dequantize)[
        (packed_k_programs, triton.cdiv(n, 64))
    ](
        packed_weight, group_lut, dequantized,
        N=n, K=k, GROUP_SIZE=group_size,
        PER_CHANNEL=per_channel,
        BLOCK_PACKED_K=block_packed_k, BLOCK_N=64,
    )
    output = torch.mm(x, dequantized)
    return output + bias if has_bias else output


@triton_op("mixnlq::custom_bf16_linear", mutates_args={})
def custom_bf16_linear_op(
    x: torch.Tensor,
    weight_kn: torch.Tensor,
    bias: torch.Tensor,
    has_bias: bool,
) -> torch.Tensor:
    m, k = x.shape
    n = weight_kn.shape[1]
    if m != 1:
        output = torch.mm(x, weight_kn)
        return output + bias if has_bias else output
    partial = torch.empty(
        (DECODE_SPLIT_K, n), dtype=torch.float32, device=x.device
    )
    wrap_triton(_bf16_gemv)[_decode_splitk_grid(n)](
        x, weight_kn, partial,
        N=n, K=k, SPLIT_K=DECODE_SPLIT_K,
    )
    output = partial.sum(dim=0, keepdim=True).to(dtype=x.dtype)
    return output + bias if has_bias else output


def _pack_nibbles_k_major(codes: torch.Tensor) -> torch.Tensor:
    if codes.shape[-1] % 2:
        raise ValueError("INT4 packing requires an even in_features dimension")
    codes = codes.to(torch.uint8).reshape(codes.shape[0], -1)
    packed_nk = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed_nk.transpose(0, 1).contiguous()


def _tile_pack_kmajor(packed_kmajor: torch.Tensor) -> torch.Tensor:
    if packed_kmajor.ndim != 2:
        raise ValueError("K-major packed weight must have shape (K/2, N)")
    packed_k, n = packed_kmajor.shape
    if packed_k % MIXNLQ_PACKED_K_TILE:
        raise ValueError(
            f"packed K={packed_k} must be divisible by "
            f"{MIXNLQ_PACKED_K_TILE}"
        )
    if n % MIXNLQ_N_TILE:
        raise ValueError(
            f"N={n} must be divisible by MixNLQ N tile {MIXNLQ_N_TILE}"
        )
    return (
        packed_kmajor.reshape(
            packed_k // MIXNLQ_PACKED_K_TILE,
            MIXNLQ_PACKED_K_TILE,
            n // MIXNLQ_N_TILE,
            MIXNLQ_N_TILE // 2,
            2,
        )
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )


def _pack_nibbles_mixnlq_tiled(codes: torch.Tensor) -> torch.Tensor:
    return _tile_pack_kmajor(_pack_nibbles_k_major(codes))


def _pack_mixnlq3_bitplanes(codes: torch.Tensor) -> torch.Tensor:
    """Pack eight generic 3-bit codes into three plane-major bit planes."""
    if codes.ndim != 2 or codes.shape[1] % 8:
        raise ValueError("INT3 codes must have shape (N, K), K divisible by 8")
    if bool(torch.any((codes < 0) | (codes > 7)).item()):
        raise ValueError("INT3 codes must lie in the inclusive range [0, 7]")
    grouped = codes.to(torch.uint8).reshape(codes.shape[0], -1, 8)
    shifts = torch.arange(8, dtype=torch.int64, device=codes.device)
    index = grouped & 0x03
    signs = (grouped >> 2) & 0x01
    low = torch.sum(
        ((index & 1).to(torch.int64) << shifts),
        dim=-1,
    ).to(torch.uint8)
    high = torch.sum(
        (((index >> 1) & 1).to(torch.int64) << shifts),
        dim=-1,
    ).to(torch.uint8)
    sign = torch.sum(
        (signs.to(torch.int64) << shifts),
        dim=-1,
    ).to(torch.uint8)
    plane_major = (
        torch.stack((low, high, sign), dim=1)
        .permute(2, 1, 0)
        .contiguous()
    )
    return _tile_pack_int3_plane_major(plane_major)


def _tile_pack_int3_plane_major(
    plane_major: torch.Tensor,
) -> torch.Tensor:
    """Reorder [K/8, 3, N] into warp-coalesced K128/N256 segments.

    The result is [Ktile, Ntile, plane, Khalf, output_pair, payload16].
    For a fixed plane/Khalf, adjacent CUDA lanes therefore load adjacent
    16-byte payloads instead of striding by one thread's complete 96 bytes.
    """
    if plane_major.ndim != 3 or plane_major.shape[1] != 3:
        raise ValueError("legacy INT3 weights must have shape (K/8, 3, N)")
    packed_k, _, n = plane_major.shape
    if packed_k % 16 or n % 256:
        raise ValueError(
            "INT3 tiled weights require K divisible by 128 and "
            "N divisible by 256"
        )
    return (
        plane_major.reshape(
            packed_k // 16,
            2,
            8,
            3,
            n // 256,
            128,
            2,
        )
        .permute(0, 4, 3, 1, 5, 2, 6)
        .contiguous()
        .reshape(
            packed_k // 16,
            n // 256,
            3,
            2,
            128,
            16,
        )
    )


def _retile_int3_thread_major(
    thread_major: torch.Tensor,
) -> torch.Tensor:
    """Convert the v1 [Kt,Nt,pair,plane,Kp16,lane] layout to v2."""
    if (
        thread_major.ndim != 6
        or tuple(thread_major.shape[2:]) != (128, 3, 16, 2)
    ):
        raise ValueError(
            "INT3 v1 tiled weights must have shape "
            "(K/128, N/256, 128, 3, 16, 2)"
        )
    k_tiles, n_tiles = thread_major.shape[:2]
    plane_major = (
        thread_major.permute(0, 4, 3, 1, 2, 5)
        .contiguous()
        .reshape(k_tiles * 16, 3, n_tiles * 256)
    )
    return _tile_pack_int3_plane_major(plane_major)


# Backward-compatible helper name used by existing validation scripts.
_tile_pack_mixnlq_kmajor = _tile_pack_kmajor


def _quantize_nf4(
    weight: torch.Tensor,
    group_size: int,
    chunk_groups: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Nearest-codebook NF4 PTQ without constructing a full W x 16 tensor."""
    grouped = weight.float().reshape(-1, group_size)
    absmax = grouped.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    codebook = torch.tensor(
        NF4_CODEBOOK_VALUES,
        dtype=torch.float32,
        device=weight.device,
    )
    codes = torch.empty(grouped.shape, dtype=torch.uint8, device=weight.device)
    for start in range(0, grouped.shape[0], chunk_groups):
        end = min(start + chunk_groups, grouped.shape[0])
        normalized = grouped[start:end] / absmax[start:end]
        distances = (normalized.unsqueeze(-1) - codebook).abs()
        codes[start:end] = distances.argmin(dim=-1).to(torch.uint8)
    return codes.reshape(weight.shape), absmax.squeeze(1), codebook

def _quantize_nf3(
    weight: torch.Tensor,
    group_size: int,
    chunk_groups: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Nearest 8-entry normal-codebook PTQ for the LUT3 comparison."""
    grouped = weight.float().reshape(-1, group_size)
    absmax = grouped.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    codebook = torch.tensor(
        NF3_CODEBOOK_VALUES,
        dtype=torch.float32,
        device=weight.device,
    )
    codes = torch.empty(grouped.shape, dtype=torch.uint8, device=weight.device)
    for start in range(0, grouped.shape[0], chunk_groups):
        end = min(start + chunk_groups, grouped.shape[0])
        normalized = grouped[start:end] / absmax[start:end]
        distances = (normalized.unsqueeze(-1) - codebook).abs()
        codes[start:end] = distances.argmin(dim=-1).to(torch.uint8)
    return codes.reshape(weight.shape), absmax.squeeze(1), codebook


class TritonInt4Linear(nn.Module):
    """Packed low-bit layer with separate prefill and M=1 decode paths."""

    def __init__(
        self,
        packed_weight: torch.Tensor,
        scale_0: torch.Tensor,
        scale_1: torch.Tensor,
        bias: torch.Tensor | None,
        in_features: int,
        out_features: int,
        group_size: int,
        quant_mode: QuantMode,
        codebook: torch.Tensor | None = None,
        group_lut: torch.Tensor | None = None,
        scale_2: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        group_size = _resolve_group_size(group_size, in_features)
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.quant_mode = quant_mode
        self.decode_split_k = 0
        self.use_cuda_decode = (
            quant_mode in {
                "mixnlq", "mixnlq3", "linear", "linear3",
                "nf4_lut", "lut3",
            }
            and cuda_tiled_group_supported(group_size)
            and cuda_decode_enabled()
        )
        expected_kmajor_shape = (in_features // 2, out_features)
        expected_mixnlq_shape = (
            in_features // (2 * MIXNLQ_PACKED_K_TILE),
            out_features // MIXNLQ_N_TILE,
            MIXNLQ_N_TILE // 2,
            MIXNLQ_PACKED_K_TILE,
            2,
        )
        expected_scale_shape = (in_features // group_size, out_features)
        if quant_mode in {"mixnlq3", "linear3", "lut3"}:
            if in_features % 128 or out_features % 256:
                raise ValueError(
                    f"{quant_mode} tiled weights require K divisible by 128 "
                    "and N divisible by 256"
                )
            expected_weight_shape = (
                in_features // 128,
                out_features // 256,
                3,
                2,
                128,
                16,
            )
            self.packed_layout = INT3_TILED_PACKED_LAYOUT
        elif quant_mode in TILED_QUANT_MODES:
            if (
                in_features % (2 * MIXNLQ_PACKED_K_TILE)
                or out_features % MIXNLQ_N_TILE
            ):
                raise ValueError(
                    f"{quant_mode} tiled weights require K divisible by 16 and "
                    "N divisible by 64"
                )
            expected_weight_shape = expected_mixnlq_shape
            self.packed_layout = TILED_PACKED_LAYOUT
        else:
            expected_weight_shape = expected_kmajor_shape
            self.packed_layout = "k-major"
        if tuple(packed_weight.shape) != expected_weight_shape:
            raise ValueError(
                f"packed_weight shape {tuple(packed_weight.shape)} != "
                f"{expected_weight_shape}"
            )
        if quant_mode == "lut3":
            if scale_0.numel() != 0 or scale_1.numel() != 0:
                raise ValueError("LUT3 stores no separate scale tensors")
            expected_lut_shape = (
                expected_scale_shape[0],
                8,
                expected_scale_shape[1],
            )
            if group_lut is None or tuple(group_lut.shape) != expected_lut_shape:
                raise ValueError(
                    f"group_lut shape must be {expected_lut_shape}, got "
                    f"{None if group_lut is None else tuple(group_lut.shape)}"
                )
        elif quant_mode == "nf4_lut":
            if scale_0.numel() != 0 or scale_1.numel() != 0:
                raise ValueError("NF4 group-LUT stores no separate scale tensors")
            expected_lut_shape = (*expected_scale_shape, 16)
            if group_lut is None or tuple(group_lut.shape) != expected_lut_shape:
                raise ValueError(
                    f"group_lut shape must be {expected_lut_shape}, got "
                    f"{None if group_lut is None else tuple(group_lut.shape)}"
                )
        elif tuple(scale_0.shape) != expected_scale_shape:
            raise ValueError(f"scale_0 shape mismatch: {tuple(scale_0.shape)}")
        if quant_mode == "nf4":
            if scale_1.numel() != 0:
                raise ValueError("NF4 uses one absmax tensor; scale_1 must be empty")
            if codebook is None or tuple(codebook.shape) != (16,):
                raise ValueError("NF4 requires a 16-entry codebook")
        elif quant_mode not in {"nf4_lut", "lut3"} and tuple(scale_1.shape) != expected_scale_shape:
            raise ValueError(f"scale_1 shape mismatch: {tuple(scale_1.shape)}")
        if quant_mode == "mixnlq3":
            if scale_2 is None or tuple(scale_2.shape) != expected_scale_shape:
                raise ValueError(
                    "mixnlq3 requires scale_2 quantization-bias tensor with "
                    f"shape {expected_scale_shape}"
                )
        elif scale_2 is not None and scale_2.numel() != 0:
            raise ValueError(f"{quant_mode} must not provide scale_2")
        self.register_buffer("packed_weight", packed_weight)
        self.register_buffer("scale_0", scale_0)
        self.register_buffer("scale_1", scale_1)
        self.register_buffer(
            "scale_2",
            (
                torch.empty(0, dtype=scale_0.dtype, device=scale_0.device)
                if scale_2 is None
                else scale_2
            ),
        )
        self.register_buffer(
            "codebook",
            (
                torch.empty(0, dtype=scale_0.dtype, device=scale_0.device)
                if codebook is None
                else codebook.detach().to(
                    dtype=scale_0.dtype, device=scale_0.device
                )
            ),
        )
        self.register_buffer(
            "group_lut",
            (
                torch.empty(0, dtype=scale_0.dtype, device=scale_0.device)
                if group_lut is None
                else group_lut.detach().to(
                    dtype=scale_0.dtype, device=scale_0.device
                )
            ),
        )
        self.has_bias = bias is not None
        self.register_buffer(
            "bias",
            (
                torch.empty(0, dtype=scale_0.dtype, device=scale_0.device)
                if bias is None
                else bias.detach().clone()
            ),
        )

    @classmethod
    @torch.no_grad()
    def from_float(
        cls,
        layer: nn.Linear,
        quant_mode: QuantMode,
        group_size: int,
        mixnlq_iterations: int = 0,
        mixnlq_activation_samples: torch.Tensor | None = None,
        mixnlq_activation_hessian: torch.Tensor | None = None,
    ) -> "TritonInt4Linear":
        weight = layer.weight.detach()
        out_features, in_features = weight.shape
        group_size = _resolve_group_size(group_size, in_features)

        codebook = None
        group_lut = None
        scale_2 = None
        if quant_mode == "mixnlq":
            if (
                mixnlq_activation_samples is None
                and mixnlq_activation_hessian is None
            ):
                from utils.quant_utils import pseudo_mix_quant

                _, params = pseudo_mix_quant(
                    wbits=4,
                    W=weight.float(),
                    w_group_size=group_size,
                    iterations=mixnlq_iterations,
                )
                signs, indices, exp_scales, linear_scales = params
            else:
                from utils.quant_utils import activation_aware_mixnlq_quant

                _, signs, indices, params = (
                    activation_aware_mixnlq_quant(
                        W=weight,
                        activation_samples=mixnlq_activation_samples,
                        hessian=mixnlq_activation_hessian,
                        wbits=4,
                        group_size=group_size,
                        iterations=mixnlq_iterations,
                    )
                )
                _, _, exp_scales, linear_scales = params
            codes = indices.to(torch.uint8).reshape(out_features, in_features)
            codes |= (signs.reshape(out_features, in_features) < 0).to(torch.uint8) << 3
            scale_0 = linear_scales.reshape(out_features, -1).T.contiguous()
            scale_1 = exp_scales.reshape(out_features, -1).T.contiguous()
        elif quant_mode == "mixnlq3":
            if (
                mixnlq_activation_samples is None
                and mixnlq_activation_hessian is None
            ):
                from utils.quant_utils import pseudo_mix_quant_bias

                _, params = pseudo_mix_quant_bias(
                    wbits=3,
                    W=weight.float(),
                    w_group_size=group_size,
                    iterations=mixnlq_iterations,
                )
                signs, indices, exp_scales, linear_scales, quant_bias = params
            else:
                from utils.quant_utils import activation_aware_mixnlq_quant

                _, signs, indices, params = activation_aware_mixnlq_quant(
                    W=weight,
                    activation_samples=mixnlq_activation_samples,
                    hessian=mixnlq_activation_hessian,
                    wbits=3,
                    group_size=group_size,
                    iterations=mixnlq_iterations,
                )
                _, _, exp_scales, linear_scales, quant_bias = params
            codes = indices.to(torch.uint8).reshape(out_features, in_features)
            codes |= (
                (signs.reshape(out_features, in_features) < 0).to(torch.uint8)
                << 2
            )
            scale_0 = linear_scales.reshape(out_features, -1).T.contiguous()
            scale_1 = exp_scales.reshape(out_features, -1).T.contiguous()
            scale_2 = quant_bias.reshape(out_features, -1).T.contiguous()
        elif quant_mode == "linear":
            from utils.quant_utils import pseudo_RTN_quant

            q, scales, zero_points, _ = pseudo_RTN_quant(
                wbits=4, W=weight.float(), group_size=group_size
            )
            codes = q.to(torch.uint8).reshape(out_features, in_features)
            scale_0 = scales.reshape(out_features, -1).T.contiguous()
            scale_1 = zero_points.reshape(out_features, -1).T.contiguous()
        elif quant_mode == "linear3":
            from utils.quant_utils import pseudo_RTN_quant

            q, scales, zero_points, _ = pseudo_RTN_quant(
                wbits=3, W=weight.float(), group_size=group_size
            )
            codes = q.to(torch.uint8).reshape(out_features, in_features)
            scale_0 = scales.reshape(out_features, -1).T.contiguous()
            scale_1 = zero_points.reshape(out_features, -1).T.contiguous()
        elif quant_mode == "nf4":
            codes, absmax, codebook = _quantize_nf4(weight, group_size)
            scale_0 = absmax.reshape(out_features, -1).T.contiguous()
            scale_1 = torch.empty(
                0, dtype=weight.dtype, device=weight.device
            )
        elif quant_mode == "nf4_lut":
            codes, absmax, nf4_codebook = _quantize_nf4(weight, group_size)
            grouped_absmax = (
                absmax.reshape(out_features, -1).T.contiguous().to(weight.dtype)
            )
            rounded_codebook = nf4_codebook.to(weight.dtype)
            group_lut = (
                grouped_absmax[:, :, None].float()
                * rounded_codebook[None, None, :].float()
            ).to(weight.dtype).contiguous()
            scale_0 = torch.empty(
                0, dtype=weight.dtype, device=weight.device
            )
            scale_1 = torch.empty(
                0, dtype=weight.dtype, device=weight.device
            )
        elif quant_mode == "lut3":
            codes, absmax, nf3_codebook = _quantize_nf3(
                weight, group_size
            )
            grouped_absmax = (
                absmax.reshape(out_features, -1).T.contiguous().to(weight.dtype)
            )
            rounded_codebook = nf3_codebook.to(weight.dtype)
            group_lut = (
                grouped_absmax[:, None, :].float()
                * rounded_codebook[None, :, None].float()
            ).to(weight.dtype).contiguous()
            scale_0 = torch.empty(
                0, dtype=weight.dtype, device=weight.device
            )
            scale_1 = torch.empty(
                0, dtype=weight.dtype, device=weight.device
            )
        else:
            raise ValueError(f"Unsupported quant_mode: {quant_mode}")

        if quant_mode in {"mixnlq3", "linear3", "lut3"}:
            packed_weight = _pack_mixnlq3_bitplanes(codes)
        elif quant_mode in TILED_QUANT_MODES:
            packed_weight = _pack_nibbles_mixnlq_tiled(codes)
        else:
            packed_weight = _pack_nibbles_k_major(codes)
        return cls(
            packed_weight=packed_weight,
            scale_0=scale_0.to(weight.dtype),
            scale_1=scale_1.to(weight.dtype),
            bias=layer.bias,
            in_features=in_features,
            out_features=out_features,
            group_size=group_size,
            quant_mode=quant_mode,
            codebook=codebook,
            group_lut=group_lut,
            scale_2=(
                None if scale_2 is None else scale_2.to(weight.dtype)
            ),
        )

    @classmethod
    def from_checkpoint_entry(
        cls,
        entry: dict,
        quant_mode: QuantMode,
        group_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "TritonInt4Linear":
        bias = entry.get("bias")
        if bias is not None and bias.numel() == 0:
            bias = None
        codebook = entry.get("codebook")
        group_lut = entry.get("group_lut")
        scale_2 = entry.get("scale_2")
        if quant_mode == "nf4":
            # NF4 has only one per-group absmax tensor. Some exporters retain
            # a legacy/placeholder scale_1 field; it is not part of NF4
            # reconstruction and must not be interpreted as another scale.
            legacy_scale_1 = entry.get("scale_1")
            scale_1 = torch.empty(0, dtype=dtype)
            if codebook is None or codebook.numel() == 0:
                if legacy_scale_1 is not None and legacy_scale_1.numel() != 0:
                    raise ValueError(
                        "This checkpoint is labeled NF4 but has a non-empty "
                        "scale_1 and no NF4 codebook. It was likely exported by "
                        "an older implementation that encoded linear INT4; "
                        "re-export it with the current quant2.py."
                    )
                codebook = torch.tensor(NF4_CODEBOOK_VALUES, dtype=dtype)
            elif codebook.numel() != 16:
                raise ValueError(
                    f"NF4 checkpoint codebook must contain 16 values, got "
                    f"{codebook.numel()}"
                )
            codebook = codebook.reshape(16)
        elif quant_mode in {"nf4_lut", "lut3"}:
            scale_1 = torch.empty(0, dtype=dtype)
            if group_lut is None:
                raise ValueError(
                    f"{quant_mode} checkpoint is missing group_lut; "
                    f"re-export it with quant_real.py --quant-mode {quant_mode}"
                )
        else:
            scale_1 = entry.get("scale_1")
            if scale_1 is None:
                raise ValueError(
                    f"{quant_mode} checkpoint is missing required scale_1"
                )
        in_features = int(entry["in_features"])
        out_features = int(entry["out_features"])
        packed_weight = entry["packed_weight"]
        packed_layout = entry.get("packed_layout", "k-major")
        if quant_mode in {"mixnlq3", "linear3", "lut3"}:
            if packed_layout == INT3_LEGACY_PACKED_LAYOUT:
                packed_weight = _tile_pack_int3_plane_major(packed_weight)
            elif packed_layout == INT3_THREAD_PACKED_LAYOUT:
                packed_weight = _retile_int3_thread_major(packed_weight)
            elif packed_layout != INT3_TILED_PACKED_LAYOUT:
                raise ValueError(
                    f"{quant_mode} requires tiled INT3 weights, got "
                    f"{packed_layout!r}; re-export the checkpoint"
                )
            if (
                quant_mode == "mixnlq3"
                and (scale_2 is None or scale_2.numel() == 0)
            ):
                raise ValueError(
                    "mixnlq3 checkpoint is missing required scale_2 "
                    "(quantization bias)"
                )
        elif quant_mode in TILED_QUANT_MODES:
            if packed_layout == "k-major":
                packed_weight = _tile_pack_kmajor(packed_weight)
            elif packed_layout not in {
                TILED_PACKED_LAYOUT,
                "mixnlq-n64-kp8",
            }:
                raise ValueError(
                    f"Unsupported tiled packed layout for {quant_mode}: "
                    f"{packed_layout!r}"
                )
        elif packed_layout != "k-major":
            raise ValueError(
                f"{quant_mode} requires k-major packed weights, got "
                f"{packed_layout!r}"
            )
        return cls(
            packed_weight=packed_weight.to(device=device),
            scale_0=entry["scale_0"].to(device=device, dtype=dtype),
            scale_1=scale_1.to(device=device, dtype=dtype),
            bias=None if bias is None else bias.to(device=device, dtype=dtype),
            in_features=in_features,
            out_features=out_features,
            group_size=group_size,
            quant_mode=quant_mode,
            codebook=(
                None
                if codebook is None
                else codebook.to(device=device, dtype=dtype)
            ),
            group_lut=(
                None
                if group_lut is None
                else group_lut.to(device=device, dtype=dtype)
            ),
            scale_2=(
                None
                if scale_2 is None
                else scale_2.to(device=device, dtype=dtype)
            ),
        )

    def checkpoint_entry(self) -> dict:
        return {
            "packed_weight": self.packed_weight.detach().cpu(),
            "packed_layout": self.packed_layout,
            "scale_0": self.scale_0.detach().cpu(),
            "scale_1": self.scale_1.detach().cpu(),
            "scale_2": self.scale_2.detach().cpu(),
            "codebook": self.codebook.detach().cpu(),
            "group_lut": self.group_lut.detach().cpu(),
            "bias": self.bias.detach().cpu(),
            "in_features": self.in_features,
            "out_features": self.out_features,
        }

    def autotune_decode_split_k(self) -> int:
        if self.use_cuda_decode:
            x = torch.ones(
                (1, self.in_features),
                dtype=self.scale_0.dtype,
                device=self.packed_weight.device,
            )
            self.decode_split_k = autotune_cuda_decode_split_k(
                x, self.packed_weight, self.scale_0, self.scale_1,
                self.scale_2, self.group_lut,
                self.bias if self.has_bias else None,
                self.group_size, self.quant_mode,
            )
        return self.decode_split_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        x_2d = x.reshape(-1, self.in_features).contiguous()
        if self.use_cuda_decode and x_2d.shape[0] == 1:
            if self.quant_mode == "mixnlq3":
                output = mixnlq3_cuda_decode(
                    x_2d,
                    self.packed_weight,
                    self.scale_0,
                    self.scale_1,
                    self.scale_2,
                    self.bias if self.has_bias else None,
                    self.group_size,
                    split_k=self.decode_split_k,
                )
            elif self.quant_mode == "linear3":
                output = linear3_cuda_decode(
                    x_2d,
                    self.packed_weight,
                    self.scale_0,
                    self.scale_1,
                    self.bias if self.has_bias else None,
                    self.group_size,
                    split_k=self.decode_split_k,
                )
            elif self.quant_mode == "lut3":
                output = lut3_cuda_decode(
                    x_2d,
                    self.packed_weight,
                    self.group_lut,
                    self.bias if self.has_bias else None,
                    self.group_size,
                    split_k=self.decode_split_k,
                )
            elif self.quant_mode == "mixnlq":
                output = mixnlq_cuda_decode(
                    x_2d,
                    self.packed_weight,
                    self.scale_0,
                    self.scale_1,
                    self.bias if self.has_bias else None,
                    self.group_size,
                    split_k=self.decode_split_k,
                )
            elif self.quant_mode == "linear":
                output = linear_cuda_decode(
                    x_2d,
                    self.packed_weight,
                    self.scale_0,
                    self.scale_1,
                    self.bias if self.has_bias else None,
                    self.group_size,
                    split_k=self.decode_split_k,
                )
            else:
                output = lut_cuda_decode(
                    x_2d,
                    self.packed_weight,
                    self.group_lut,
                    self.bias if self.has_bias else None,
                    self.group_size,
                    split_k=self.decode_split_k,
                )
        elif self.quant_mode == "lut3":
            output = lut3_linear_op(
                x_2d,
                self.packed_weight,
                self.group_lut,
                self.bias,
                self.group_size,
                self.has_bias,
            )
        elif self.quant_mode == "nf4_lut":
            output = nf4_group_lut_linear_op(
                x_2d,
                self.packed_weight,
                self.group_lut,
                self.bias,
                self.group_size,
                self.has_bias,
            )
        elif self.quant_mode == "nf4":
            output = nf4_int4_linear_op(
                x_2d,
                self.packed_weight,
                self.scale_0,
                self.codebook,
                self.bias,
                self.group_size,
                self.has_bias,
            )
        else:
            if self.quant_mode == "mixnlq3":
                output = mixnlq3_linear_op(
                    x_2d,
                    self.packed_weight,
                    self.scale_0,
                    self.scale_1,
                    self.scale_2,
                    self.bias,
                    self.group_size,
                    self.has_bias,
                )
            elif self.quant_mode == "linear3":
                output = linear3_linear_op(
                    x_2d,
                    self.packed_weight,
                    self.scale_0,
                    self.scale_1,
                    self.bias,
                    self.group_size,
                    self.has_bias,
                )
            else:
                op = (
                    mixnlq_int4_linear_op
                    if self.quant_mode == "mixnlq"
                    else linear_int4_linear_op
                )
                output = op(
                    x_2d,
                    self.packed_weight,
                    self.scale_0,
                    self.scale_1,
                    self.bias,
                    self.group_size,
                    self.has_bias,
                )
        return output.reshape(*input_shape[:-1], self.out_features)


class CustomBF16Linear(nn.Module):
    """BF16 control using the same prefill/decode scheduling as INT4."""

    def __init__(self, layer: nn.Linear) -> None:
        super().__init__()
        self.in_features = layer.in_features
        self.out_features = layer.out_features
        self.register_buffer(
            "weight_kn", layer.weight.detach().transpose(0, 1).contiguous()
        )
        self.has_bias = layer.bias is not None
        self.register_buffer(
            "bias",
            (
                torch.empty(
                    0, dtype=layer.weight.dtype, device=layer.weight.device
                )
                if layer.bias is None
                else layer.bias.detach().clone()
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        x_2d = x.reshape(-1, self.in_features).contiguous()
        output = custom_bf16_linear_op(
            x_2d, self.weight_kn, self.bias, self.has_bias
        )
        return output.reshape(*input_shape[:-1], self.out_features)
