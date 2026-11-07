#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Optional.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kDecodeThreads = 128;
constexpr int kReduceThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kDecodeWarps = kDecodeThreads / kWarpSize;
constexpr int kVec2OutputsPerThread = 2;
constexpr int kVec2OutputsPerBlock =
    kDecodeThreads * kVec2OutputsPerThread;
constexpr int kMixNtile = 64;
constexpr int kMixPackedKtile = 8;

__device__ __forceinline__ int mixnlq_tiled_offset(
    const int packed_k,
    const int output_index,
    const int n) {
  const int n_tiles = n / kMixNtile;
  const int packed_k_tile = packed_k / kMixPackedKtile;
  const int packed_k_inner = packed_k % kMixPackedKtile;
  const int n_tile = output_index / kMixNtile;
  const int n_inner = output_index % kMixNtile;
  const int lane = n_inner / 2;
  const int pair = n_inner % 2;
  const int tile = packed_k_tile * n_tiles + n_tile;
  return (((tile * (kMixNtile / 2) + lane) * kMixPackedKtile
           + packed_k_inner) * 2 + pair);
}

__device__ __forceinline__ int int3_tiled_offset(
    const int packed_k,
    const int output_index,
    const int plane,
    const int n) {
  const int n_tiles = n / 256;
  const int packed_k_tile = packed_k / 16;
  const int packed_k_inner = packed_k % 16;
  const int packed_k_half = packed_k_inner / 8;
  const int packed_k_half_inner = packed_k_inner % 8;
  const int n_tile = output_index / 256;
  const int output_pair = (output_index % 256) / 2;
  const int pair_lane = output_index % 2;
  return ((((((packed_k_tile * n_tiles + n_tile) * 3 + plane) * 2
              + packed_k_half) * 128 + output_pair) * 16
            + packed_k_half_inner * 2) + pair_lane);
}

__device__ __forceinline__ void accumulate_mixnlq_code(
    const uint32_t code,
    const float x,
    float& linear_sum,
    float& exponential_sum) {
  const uint32_t index = code & 0x7U;
  const uint32_t sign_bit = (code & 0x8U) << 28;
  const float signed_x =
      __uint_as_float(__float_as_uint(x) ^ sign_bit);
  const float linear_coefficient = static_cast<float>(index);
  const float exponential_coefficient =
      __uint_as_float((index + 127U) << 23);
  linear_sum = fmaf(signed_x, linear_coefficient, linear_sum);
  exponential_sum = fmaf(
      signed_x, exponential_coefficient, exponential_sum);
}

template <int SplitK, bool DirectOutput>
__global__ void mixnlq_decode_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ linear_scale,
    const __nv_bfloat16* __restrict__ exp_scale,
    const __nv_bfloat16* __restrict__ bias,
    float* __restrict__ partial,
    __nv_bfloat16* __restrict__ output,
    const int k,
    const int n,
    const int group_size,
    const int groups) {
  extern __shared__ __nv_bfloat16 shared_x[];

  const int output_index =
      static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int split_id = static_cast<int>(blockIdx.y);
  const int groups_per_split = (groups + SplitK - 1) / SplitK;
  const int first_group = split_id * groups_per_split;
  const int last_group = min(first_group + groups_per_split, groups);
  float accumulator = 0.0F;

  for (int group = first_group; group < last_group; ++group) {
    const int k_begin = group * group_size;
    for (int index = threadIdx.x; index < group_size; index += blockDim.x) {
      shared_x[index] = x[k_begin + index];
    }
    __syncthreads();

    if (output_index < n) {
      const int scale_offset = group * n + output_index;
      const float linear =
          __bfloat162float(linear_scale[scale_offset]);
      const float exponential =
          __bfloat162float(exp_scale[scale_offset]);
      const int packed_count = group_size / 2;
      float linear_sum = 0.0F;
      float exponential_sum = 0.0F;

      for (int packed_index = 0; packed_index < packed_count; ++packed_index) {
        const uint8_t packed =
            packed_weight[mixnlq_tiled_offset(
                k_begin / 2 + packed_index, output_index, n)];
        const int x_index = packed_index * 2;
        const float x_even = __bfloat162float(shared_x[x_index]);
        const float x_odd = __bfloat162float(shared_x[x_index + 1]);
        const unsigned int even_code =
            static_cast<unsigned int>(packed & 0x0FU);
        const unsigned int odd_code =
            static_cast<unsigned int>((packed >> 4) & 0x0FU);
        accumulate_mixnlq_code(
            even_code, x_even, linear_sum, exponential_sum);
        accumulate_mixnlq_code(
            odd_code, x_odd, linear_sum, exponential_sum);
      }
      accumulator = fmaf(linear, linear_sum, accumulator);
      accumulator = fmaf(exponential, exponential_sum, accumulator);
    }
    __syncthreads();
  }

  if (output_index >= n) {
    return;
  }
  if constexpr (DirectOutput) {
    if (bias != nullptr) {
      accumulator += __bfloat162float(bias[output_index]);
    }
    output[output_index] = __float2bfloat16_rn(accumulator);
  } else {
    partial[split_id * n + output_index] = accumulator;
  }
}

// MixNLQ 3-bit layout: every eight logical K values use three bit-plane
// bytes [index_bit_0, index_bit_1, sign_bit], stored as [K/8, 3, N].
// The bias below is the per-quantization-group magnitude offset, distinct
// from the optional Linear layer output bias.
template <int SplitK, bool DirectOutput>
__global__ void mixnlq3_decode_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ linear_scale,
    const __nv_bfloat16* __restrict__ exp_scale,
    const __nv_bfloat16* __restrict__ quant_bias,
    const __nv_bfloat16* __restrict__ output_bias,
    float* __restrict__ partial,
    __nv_bfloat16* __restrict__ output,
    const int k,
    const int n,
    const int group_size,
    const int groups) {
  extern __shared__ __nv_bfloat16 shared_x[];

  const int output_index =
      static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int split_id = static_cast<int>(blockIdx.y);
  const int groups_per_split = (groups + SplitK - 1) / SplitK;
  const int first_group = split_id * groups_per_split;
  const int last_group = min(first_group + groups_per_split, groups);
  float accumulator = 0.0F;

  for (int group = first_group; group < last_group; ++group) {
    const int k_begin = group * group_size;
    for (int index = threadIdx.x; index < group_size; index += blockDim.x) {
      shared_x[index] = x[k_begin + index];
    }
    __syncthreads();

    if (output_index < n) {
      const int scale_offset = group * n + output_index;
      const float linear =
          __bfloat162float(linear_scale[scale_offset]);
      const float exponential =
          __bfloat162float(exp_scale[scale_offset]);
      const float magnitude_bias =
          __bfloat162float(quant_bias[scale_offset]);
      float linear_sum = 0.0F;
      float exponential_sum = 0.0F;
      float sign_sum = 0.0F;

      const int packed_begin = k_begin / 8;
      const int packed_count = group_size / 8;
      for (int packed_index = 0;
           packed_index < packed_count;
           ++packed_index) {
        const int packed_offset =
            (packed_begin + packed_index) * (3 * n) + output_index;
        const uint8_t index_low = packed_weight[packed_offset];
        const uint8_t index_high = packed_weight[packed_offset + n];
        const uint8_t signs = packed_weight[packed_offset + 2 * n];
#pragma unroll
        for (int bit = 0; bit < 8; ++bit) {
          const uint32_t index =
              ((index_low >> bit) & 1U)
              | (((index_high >> bit) & 1U) << 1);
          const uint32_t sign_bit =
              static_cast<uint32_t>((signs >> bit) & 1U) << 31;
          const float value =
              __bfloat162float(shared_x[packed_index * 8 + bit]);
          const float signed_x = __uint_as_float(
              __float_as_uint(value) ^ sign_bit);
          linear_sum = fmaf(
              signed_x, static_cast<float>(index), linear_sum);
          const float exponential_coefficient =
              __uint_as_float((index + 127U) << 23);
          exponential_sum = fmaf(
              signed_x, exponential_coefficient, exponential_sum);
          sign_sum += signed_x;
        }
      }
      accumulator = fmaf(linear, linear_sum, accumulator);
      accumulator = fmaf(exponential, exponential_sum, accumulator);
      accumulator = fmaf(magnitude_bias, sign_sum, accumulator);
    }
    __syncthreads();
  }

  if (output_index >= n) {
    return;
  }
  if constexpr (DirectOutput) {
    if (output_bias != nullptr) {
      accumulator += __bfloat162float(output_bias[output_index]);
    }
    output[output_index] = __float2bfloat16_rn(accumulator);
  } else {
    partial[split_id * n + output_index] = accumulator;
  }
}

__device__ __forceinline__ float select_lut3(
    const uint32_t code,
    const float v0,
    const float v1,
    const float v2,
    const float v3,
    const float v4,
    const float v5,
    const float v6,
    const float v7) {
  return code == 0 ? v0
      : code == 1 ? v1
      : code == 2 ? v2
      : code == 3 ? v3
      : code == 4 ? v4
      : code == 5 ? v5
      : code == 6 ? v6
      : v7;
}

template <int SplitK, bool DirectOutput>
__global__ void linear3_decode_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ scale,
    const __nv_bfloat16* __restrict__ zero_point,
    const __nv_bfloat16* __restrict__ bias,
    float* __restrict__ partial,
    __nv_bfloat16* __restrict__ output,
    const int n,
    const int group_size,
    const int groups) {
  extern __shared__ __nv_bfloat16 shared_x[];
  const int output_index =
      static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int split_id = static_cast<int>(blockIdx.y);
  const int groups_per_split = (groups + SplitK - 1) / SplitK;
  const int first_group = split_id * groups_per_split;
  const int last_group = min(first_group + groups_per_split, groups);
  float accumulator = 0.0F;

  for (int group = first_group; group < last_group; ++group) {
    const int k_begin = group * group_size;
    for (int index = threadIdx.x; index < group_size; index += blockDim.x) {
      shared_x[index] = x[k_begin + index];
    }
    __syncthreads();
    if (output_index < n) {
      const int parameter_offset = group * n + output_index;
      const float group_scale = __bfloat162float(scale[parameter_offset]);
      const float group_zero =
          __bfloat162float(zero_point[parameter_offset]);
      float q_sum = 0.0F;
      float x_sum = 0.0F;
      const int packed_begin = k_begin / 8;
      const int packed_count = group_size / 8;
      for (int packed_index = 0; packed_index < packed_count; ++packed_index) {
        const int packed_offset =
            (packed_begin + packed_index) * (3 * n) + output_index;
        const uint8_t bit0 = packed_weight[packed_offset];
        const uint8_t bit1 = packed_weight[packed_offset + n];
        const uint8_t bit2 = packed_weight[packed_offset + 2 * n];
#pragma unroll
        for (int bit = 0; bit < 8; ++bit) {
          const uint32_t code =
              ((bit0 >> bit) & 1U)
              | (((bit1 >> bit) & 1U) << 1)
              | (((bit2 >> bit) & 1U) << 2);
          const float value =
              __bfloat162float(shared_x[packed_index * 8 + bit]);
          q_sum = fmaf(value, static_cast<float>(code), q_sum);
          x_sum += value;
        }
      }
      accumulator = fmaf(
          group_scale, q_sum - group_zero * x_sum, accumulator);
    }
    __syncthreads();
  }
  if (output_index >= n) {
    return;
  }
  if constexpr (DirectOutput) {
    if (bias != nullptr) {
      accumulator += __bfloat162float(bias[output_index]);
    }
    output[output_index] = __float2bfloat16_rn(accumulator);
  } else {
    partial[split_id * n + output_index] = accumulator;
  }
}

template <int SplitK, bool DirectOutput>
__global__ void lut3_decode_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ group_lut,
    const __nv_bfloat16* __restrict__ bias,
    float* __restrict__ partial,
    __nv_bfloat16* __restrict__ output,
    const int n,
    const int group_size,
    const int groups) {
  extern __shared__ __nv_bfloat16 shared_x[];
  const int output_index =
      static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int split_id = static_cast<int>(blockIdx.y);
  const int groups_per_split = (groups + SplitK - 1) / SplitK;
  const int first_group = split_id * groups_per_split;
  const int last_group = min(first_group + groups_per_split, groups);
  float accumulator = 0.0F;

  for (int group = first_group; group < last_group; ++group) {
    const int k_begin = group * group_size;
    for (int index = threadIdx.x; index < group_size; index += blockDim.x) {
      shared_x[index] = x[k_begin + index];
    }
    __syncthreads();
    if (output_index < n) {
      const int lut_base = group * (8 * n) + output_index;
      const float v0 = __bfloat162float(group_lut[lut_base]);
      const float v1 = __bfloat162float(group_lut[lut_base + n]);
      const float v2 = __bfloat162float(group_lut[lut_base + 2 * n]);
      const float v3 = __bfloat162float(group_lut[lut_base + 3 * n]);
      const float v4 = __bfloat162float(group_lut[lut_base + 4 * n]);
      const float v5 = __bfloat162float(group_lut[lut_base + 5 * n]);
      const float v6 = __bfloat162float(group_lut[lut_base + 6 * n]);
      const float v7 = __bfloat162float(group_lut[lut_base + 7 * n]);
      const int packed_begin = k_begin / 8;
      const int packed_count = group_size / 8;
      for (int packed_index = 0; packed_index < packed_count; ++packed_index) {
        const int packed_offset =
            (packed_begin + packed_index) * (3 * n) + output_index;
        const uint8_t bit0 = packed_weight[packed_offset];
        const uint8_t bit1 = packed_weight[packed_offset + n];
        const uint8_t bit2 = packed_weight[packed_offset + 2 * n];
#pragma unroll
        for (int bit = 0; bit < 8; ++bit) {
          const uint32_t code =
              ((bit0 >> bit) & 1U)
              | (((bit1 >> bit) & 1U) << 1)
              | (((bit2 >> bit) & 1U) << 2);
          const float value =
              __bfloat162float(shared_x[packed_index * 8 + bit]);
          accumulator = fmaf(
              value,
              select_lut3(code, v0, v1, v2, v3, v4, v5, v6, v7),
              accumulator);
        }
      }
    }
    __syncthreads();
  }
  if (output_index >= n) {
    return;
  }
  if constexpr (DirectOutput) {
    if (bias != nullptr) {
      accumulator += __bfloat162float(bias[output_index]);
    }
    output[output_index] = __float2bfloat16_rn(accumulator);
  } else {
    partial[split_id * n + output_index] = accumulator;
  }
}

__device__ __forceinline__ void accumulate_mixnlq3_vec2_word(
    const uint32_t low_word,
    const uint32_t high_word,
    const uint32_t sign_word,
    const __nv_bfloat16* __restrict__ cached_x,
    const int x_index,
    float& linear_sum_0,
    float& linear_sum_1,
    float& exp_sum_0,
    float& exp_sum_1,
    float& sign_sum_0,
    float& sign_sum_1) {
#pragma unroll
  for (int packed_pair = 0; packed_pair < 2; ++packed_pair) {
    const uint16_t low =
        static_cast<uint16_t>(low_word >> (packed_pair * 16));
    const uint16_t high =
        static_cast<uint16_t>(high_word >> (packed_pair * 16));
    const uint16_t signs =
        static_cast<uint16_t>(sign_word >> (packed_pair * 16));
    const uint8_t low_0 = static_cast<uint8_t>(low);
    const uint8_t low_1 = static_cast<uint8_t>(low >> 8);
    const uint8_t high_0 = static_cast<uint8_t>(high);
    const uint8_t high_1 = static_cast<uint8_t>(high >> 8);
    const uint8_t sign_0 = static_cast<uint8_t>(signs);
    const uint8_t sign_1 = static_cast<uint8_t>(signs >> 8);
#pragma unroll
    for (int bit = 0; bit < 8; ++bit) {
      const uint32_t index_0 =
          ((low_0 >> bit) & 1U) | (((high_0 >> bit) & 1U) << 1);
      const uint32_t index_1 =
          ((low_1 >> bit) & 1U) | (((high_1 >> bit) & 1U) << 1);
      const float value = __bfloat162float(
          cached_x[x_index + packed_pair * 8 + bit]);
      const float signed_x_0 = __uint_as_float(
          __float_as_uint(value)
          ^ (static_cast<uint32_t>((sign_0 >> bit) & 1U) << 31));
      const float signed_x_1 = __uint_as_float(
          __float_as_uint(value)
          ^ (static_cast<uint32_t>((sign_1 >> bit) & 1U) << 31));
      linear_sum_0 = fmaf(
          signed_x_0, static_cast<float>(index_0), linear_sum_0);
      linear_sum_1 = fmaf(
          signed_x_1, static_cast<float>(index_1), linear_sum_1);
      exp_sum_0 = fmaf(
          signed_x_0,
          __uint_as_float((index_0 + 127U) << 23),
          exp_sum_0);
      exp_sum_1 = fmaf(
          signed_x_1,
          __uint_as_float((index_1 + 127U) << 23),
          exp_sum_1);
      sign_sum_0 += signed_x_0;
      sign_sum_1 += signed_x_1;
    }
  }
}

__device__ __forceinline__ void accumulate_linear3_vec2_word(
    const uint32_t bit0_word,
    const uint32_t bit1_word,
    const uint32_t bit2_word,
    const __nv_bfloat16* __restrict__ cached_x,
    const int x_index,
    float& q_sum_0,
    float& q_sum_1,
    float& x_sum) {
#pragma unroll
  for (int packed_pair = 0; packed_pair < 2; ++packed_pair) {
    const uint16_t bit0 =
        static_cast<uint16_t>(bit0_word >> (packed_pair * 16));
    const uint16_t bit1 =
        static_cast<uint16_t>(bit1_word >> (packed_pair * 16));
    const uint16_t bit2 =
        static_cast<uint16_t>(bit2_word >> (packed_pair * 16));
#pragma unroll
    for (int bit = 0; bit < 8; ++bit) {
      const uint32_t code_0 =
          ((static_cast<uint8_t>(bit0) >> bit) & 1U)
          | (((static_cast<uint8_t>(bit1) >> bit) & 1U) << 1)
          | (((static_cast<uint8_t>(bit2) >> bit) & 1U) << 2);
      const uint32_t code_1 =
          ((static_cast<uint8_t>(bit0 >> 8) >> bit) & 1U)
          | (((static_cast<uint8_t>(bit1 >> 8) >> bit) & 1U) << 1)
          | (((static_cast<uint8_t>(bit2 >> 8) >> bit) & 1U) << 2);
      const float value = __bfloat162float(
          cached_x[x_index + packed_pair * 8 + bit]);
      q_sum_0 = fmaf(value, static_cast<float>(code_0), q_sum_0);
      q_sum_1 = fmaf(value, static_cast<float>(code_1), q_sum_1);
      x_sum += value;
    }
  }
}

__device__ __forceinline__ void accumulate_lut3_vec2_word(
    const uint32_t bit0_word,
    const uint32_t bit1_word,
    const uint32_t bit2_word,
    const __nv_bfloat16* __restrict__ cached_x,
    const int x_index,
    const __nv_bfloat16* __restrict__ shared_lut,
    const int local_output,
    float& accumulator_0,
    float& accumulator_1) {
#pragma unroll
  for (int packed_pair = 0; packed_pair < 2; ++packed_pair) {
    const uint16_t bit0 =
        static_cast<uint16_t>(bit0_word >> (packed_pair * 16));
    const uint16_t bit1 =
        static_cast<uint16_t>(bit1_word >> (packed_pair * 16));
    const uint16_t bit2 =
        static_cast<uint16_t>(bit2_word >> (packed_pair * 16));
#pragma unroll
    for (int bit = 0; bit < 8; ++bit) {
      const uint32_t code_0 =
          ((static_cast<uint8_t>(bit0) >> bit) & 1U)
          | (((static_cast<uint8_t>(bit1) >> bit) & 1U) << 1)
          | (((static_cast<uint8_t>(bit2) >> bit) & 1U) << 2);
      const uint32_t code_1 =
          ((static_cast<uint8_t>(bit0 >> 8) >> bit) & 1U)
          | (((static_cast<uint8_t>(bit1 >> 8) >> bit) & 1U) << 1)
          | (((static_cast<uint8_t>(bit2 >> 8) >> bit) & 1U) << 2);
      const float value = __bfloat162float(
          cached_x[x_index + packed_pair * 8 + bit]);
      accumulator_0 = fmaf(
          value,
          __bfloat162float(
              shared_lut[code_0 * kVec2OutputsPerBlock + local_output]),
          accumulator_0);
      accumulator_1 = fmaf(
          value,
          __bfloat162float(
              shared_lut[
                  code_1 * kVec2OutputsPerBlock + local_output + 1]),
          accumulator_1);
    }
  }
}

// Vec2 INT3 Decode. Offline K128/N256 segment-major tiling stores one
// 16-byte payload per output pair and 64-K half. The 32 lanes of a warp load
// adjacent payloads while retaining one uint4 load per plane/half.
template <int SplitK, bool DirectOutput, int StaticGroupSize>
__global__ void mixnlq3_decode_vec2_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ linear_scale,
    const __nv_bfloat16* __restrict__ exp_scale,
    const __nv_bfloat16* __restrict__ quant_bias,
    const __nv_bfloat16* __restrict__ output_bias,
    float* __restrict__ partial,
    __nv_bfloat16* __restrict__ output,
    const int n,
    const int group_size,
    const int work_units,
    const int chunk_size) {
  constexpr int MaxChunkSize = 128;
  constexpr int StaticChunkSize =
      StaticGroupSize == 0
      ? 0
      : (StaticGroupSize < MaxChunkSize
             ? StaticGroupSize
             : MaxChunkSize);
  const int active_group_size =
      StaticGroupSize == 0 ? group_size : StaticGroupSize;
  const int active_chunk_size =
      StaticGroupSize == 0 ? chunk_size : StaticChunkSize;
  extern __shared__ __nv_bfloat16 shared_storage[];
  const int warp_id = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;
  __nv_bfloat16* warp_x = shared_storage + warp_id * MaxChunkSize;
  const int output_index =
      static_cast<int>(blockIdx.x) * kVec2OutputsPerBlock
      + static_cast<int>(threadIdx.x) * 2;
  const int split_id = static_cast<int>(blockIdx.y);
  const int units_per_split = (work_units + SplitK - 1) / SplitK;
  const int first_unit = split_id * units_per_split;
  const int last_unit = min(first_unit + units_per_split, work_units);
  float accumulator_0 = 0.0F;
  float accumulator_1 = 0.0F;

  for (int unit = first_unit; unit < last_unit; ++unit) {
    const int k_begin = unit * active_chunk_size;
    int group;
    if constexpr (StaticGroupSize == 0) {
      group = k_begin / active_group_size;
    } else {
      group = unit / (StaticGroupSize / StaticChunkSize);
    }
#pragma unroll
    for (int index = lane_id;
         index < active_chunk_size;
         index += kWarpSize) {
      warp_x[index] = x[k_begin + index];
    }
    __syncwarp();
    if (output_index < n) {
      const int parameter_offset = group * n + output_index;
      const float2 linear = __bfloat1622float2(
          *reinterpret_cast<const __nv_bfloat162*>(
              linear_scale + parameter_offset));
      const float2 exponential = __bfloat1622float2(
          *reinterpret_cast<const __nv_bfloat162*>(
              exp_scale + parameter_offset));
      const float2 magnitude_bias = __bfloat1622float2(
          *reinterpret_cast<const __nv_bfloat162*>(
              quant_bias + parameter_offset));
      float linear_sum_0 = 0.0F;
      float linear_sum_1 = 0.0F;
      float exp_sum_0 = 0.0F;
      float exp_sum_1 = 0.0F;
      float sign_sum_0 = 0.0F;
      float sign_sum_1 = 0.0F;
#pragma unroll
      for (int block64 = 0;
           block64 < active_chunk_size / 64;
           ++block64) {
        const int packed_k = k_begin / 8 + block64 * 8;
        const int low_offset =
            int3_tiled_offset(packed_k, output_index, 0, n);
        const int high_offset =
            int3_tiled_offset(packed_k, output_index, 1, n);
        const int sign_offset =
            int3_tiled_offset(packed_k, output_index, 2, n);
        const uint4 low = *reinterpret_cast<const uint4*>(
            packed_weight + low_offset);
        const uint4 high = *reinterpret_cast<const uint4*>(
            packed_weight + high_offset);
        const uint4 signs = *reinterpret_cast<const uint4*>(
            packed_weight + sign_offset);
        const int x_index = block64 * 64;
        accumulate_mixnlq3_vec2_word(
            low.x, high.x, signs.x, warp_x, x_index,
            linear_sum_0, linear_sum_1, exp_sum_0, exp_sum_1,
            sign_sum_0, sign_sum_1);
        accumulate_mixnlq3_vec2_word(
            low.y, high.y, signs.y, warp_x, x_index + 16,
            linear_sum_0, linear_sum_1, exp_sum_0, exp_sum_1,
            sign_sum_0, sign_sum_1);
        accumulate_mixnlq3_vec2_word(
            low.z, high.z, signs.z, warp_x, x_index + 32,
            linear_sum_0, linear_sum_1, exp_sum_0, exp_sum_1,
            sign_sum_0, sign_sum_1);
        accumulate_mixnlq3_vec2_word(
            low.w, high.w, signs.w, warp_x, x_index + 48,
            linear_sum_0, linear_sum_1, exp_sum_0, exp_sum_1,
            sign_sum_0, sign_sum_1);
      }
      if (active_chunk_size % 64 != 0) {
        const int packed_k =
            k_begin / 8 + (active_chunk_size / 64) * 8;
        const uint2 low = *reinterpret_cast<const uint2*>(
            packed_weight
            + int3_tiled_offset(packed_k, output_index, 0, n));
        const uint2 high = *reinterpret_cast<const uint2*>(
            packed_weight
            + int3_tiled_offset(packed_k, output_index, 1, n));
        const uint2 signs = *reinterpret_cast<const uint2*>(
            packed_weight
            + int3_tiled_offset(packed_k, output_index, 2, n));
        const int x_index = (active_chunk_size / 64) * 64;
        accumulate_mixnlq3_vec2_word(
            low.x, high.x, signs.x, warp_x, x_index,
            linear_sum_0, linear_sum_1, exp_sum_0, exp_sum_1,
            sign_sum_0, sign_sum_1);
        accumulate_mixnlq3_vec2_word(
            low.y, high.y, signs.y, warp_x, x_index + 16,
            linear_sum_0, linear_sum_1, exp_sum_0, exp_sum_1,
            sign_sum_0, sign_sum_1);
      }
      accumulator_0 = fmaf(linear.x, linear_sum_0, accumulator_0);
      accumulator_0 = fmaf(exponential.x, exp_sum_0, accumulator_0);
      accumulator_0 = fmaf(magnitude_bias.x, sign_sum_0, accumulator_0);
      accumulator_1 = fmaf(linear.y, linear_sum_1, accumulator_1);
      accumulator_1 = fmaf(exponential.y, exp_sum_1, accumulator_1);
      accumulator_1 = fmaf(magnitude_bias.y, sign_sum_1, accumulator_1);
    }
    __syncwarp();
  }
  if (output_index >= n) {
    return;
  }
  if constexpr (DirectOutput) {
    if (output_bias != nullptr) {
      const float2 bias_pair = __bfloat1622float2(
          *reinterpret_cast<const __nv_bfloat162*>(
              output_bias + output_index));
      accumulator_0 += bias_pair.x;
      accumulator_1 += bias_pair.y;
    }
    output[output_index] = __float2bfloat16_rn(accumulator_0);
    output[output_index + 1] = __float2bfloat16_rn(accumulator_1);
  } else {
    partial[split_id * n + output_index] = accumulator_0;
    partial[split_id * n + output_index + 1] = accumulator_1;
  }
}

template <int SplitK, bool DirectOutput, int StaticGroupSize>
__global__ void linear3_decode_vec2_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ scale,
    const __nv_bfloat16* __restrict__ zero_point,
    const __nv_bfloat16* __restrict__ bias,
    float* __restrict__ partial,
    __nv_bfloat16* __restrict__ output,
    const int n,
    const int group_size,
    const int work_units,
    const int chunk_size) {
  constexpr int MaxChunkSize = 128;
  constexpr int StaticChunkSize =
      StaticGroupSize == 0
      ? 0
      : (StaticGroupSize < MaxChunkSize
             ? StaticGroupSize
             : MaxChunkSize);
  const int active_group_size =
      StaticGroupSize == 0 ? group_size : StaticGroupSize;
  const int active_chunk_size =
      StaticGroupSize == 0 ? chunk_size : StaticChunkSize;
  extern __shared__ __nv_bfloat16 shared_storage[];
  const int warp_id = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;
  __nv_bfloat16* warp_x = shared_storage + warp_id * MaxChunkSize;
  const int output_index =
      static_cast<int>(blockIdx.x) * kVec2OutputsPerBlock
      + static_cast<int>(threadIdx.x) * 2;
  const int split_id = static_cast<int>(blockIdx.y);
  const int units_per_split = (work_units + SplitK - 1) / SplitK;
  const int first_unit = split_id * units_per_split;
  const int last_unit = min(first_unit + units_per_split, work_units);
  float accumulator_0 = 0.0F;
  float accumulator_1 = 0.0F;
  for (int unit = first_unit; unit < last_unit; ++unit) {
    const int k_begin = unit * active_chunk_size;
    int group;
    if constexpr (StaticGroupSize == 0) {
      group = k_begin / active_group_size;
    } else {
      group = unit / (StaticGroupSize / StaticChunkSize);
    }
#pragma unroll
    for (int index = lane_id;
         index < active_chunk_size;
         index += kWarpSize) {
      warp_x[index] = x[k_begin + index];
    }
    __syncwarp();
    if (output_index < n) {
      const int parameter_offset = group * n + output_index;
      const float2 scale_pair = __bfloat1622float2(
          *reinterpret_cast<const __nv_bfloat162*>(
              scale + parameter_offset));
      const float2 zero_pair = __bfloat1622float2(
          *reinterpret_cast<const __nv_bfloat162*>(
              zero_point + parameter_offset));
      float q_sum_0 = 0.0F;
      float q_sum_1 = 0.0F;
      float x_sum = 0.0F;
#pragma unroll
      for (int block64 = 0;
           block64 < active_chunk_size / 64;
           ++block64) {
        const int packed_k = k_begin / 8 + block64 * 8;
        const uint4 bit0 = *reinterpret_cast<const uint4*>(
            packed_weight
            + int3_tiled_offset(packed_k, output_index, 0, n));
        const uint4 bit1 = *reinterpret_cast<const uint4*>(
            packed_weight
            + int3_tiled_offset(packed_k, output_index, 1, n));
        const uint4 bit2 = *reinterpret_cast<const uint4*>(
            packed_weight
            + int3_tiled_offset(packed_k, output_index, 2, n));
        const int x_index = block64 * 64;
        accumulate_linear3_vec2_word(
            bit0.x, bit1.x, bit2.x, warp_x, x_index,
            q_sum_0, q_sum_1, x_sum);
        accumulate_linear3_vec2_word(
            bit0.y, bit1.y, bit2.y, warp_x, x_index + 16,
            q_sum_0, q_sum_1, x_sum);
        accumulate_linear3_vec2_word(
            bit0.z, bit1.z, bit2.z, warp_x, x_index + 32,
            q_sum_0, q_sum_1, x_sum);
        accumulate_linear3_vec2_word(
            bit0.w, bit1.w, bit2.w, warp_x, x_index + 48,
            q_sum_0, q_sum_1, x_sum);
      }
      if (active_chunk_size % 64 != 0) {
        const int packed_k =
            k_begin / 8 + (active_chunk_size / 64) * 8;
        const uint2 bit0 = *reinterpret_cast<const uint2*>(
            packed_weight
            + int3_tiled_offset(packed_k, output_index, 0, n));
        const uint2 bit1 = *reinterpret_cast<const uint2*>(
            packed_weight
            + int3_tiled_offset(packed_k, output_index, 1, n));
        const uint2 bit2 = *reinterpret_cast<const uint2*>(
            packed_weight
            + int3_tiled_offset(packed_k, output_index, 2, n));
        const int x_index = (active_chunk_size / 64) * 64;
        accumulate_linear3_vec2_word(
            bit0.x, bit1.x, bit2.x, warp_x, x_index,
            q_sum_0, q_sum_1, x_sum);
        accumulate_linear3_vec2_word(
            bit0.y, bit1.y, bit2.y, warp_x, x_index + 16,
            q_sum_0, q_sum_1, x_sum);
      }
      accumulator_0 = fmaf(
          scale_pair.x,
          q_sum_0 - zero_pair.x * x_sum,
          accumulator_0);
      accumulator_1 = fmaf(
          scale_pair.y,
          q_sum_1 - zero_pair.y * x_sum,
          accumulator_1);
    }
    __syncwarp();
  }
  if (output_index >= n) {
    return;
  }
  if constexpr (DirectOutput) {
    if (bias != nullptr) {
      const float2 bias_pair = __bfloat1622float2(
          *reinterpret_cast<const __nv_bfloat162*>(bias + output_index));
      accumulator_0 += bias_pair.x;
      accumulator_1 += bias_pair.y;
    }
    output[output_index] = __float2bfloat16_rn(accumulator_0);
    output[output_index + 1] = __float2bfloat16_rn(accumulator_1);
  } else {
    partial[split_id * n + output_index] = accumulator_0;
    partial[split_id * n + output_index + 1] = accumulator_1;
  }
}

template <int SplitK, bool DirectOutput, int StaticGroupSize>
__global__ void lut3_decode_vec2_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ group_lut,
    const __nv_bfloat16* __restrict__ bias,
    float* __restrict__ partial,
    __nv_bfloat16* __restrict__ output,
    const int n,
    const int group_size,
    const int work_units,
    const int chunk_size) {
  constexpr int MaxChunkSize = 128;
  constexpr int LutEntries = 8;
  constexpr int StaticChunkSize =
      StaticGroupSize == 0
      ? 0
      : (StaticGroupSize < MaxChunkSize
             ? StaticGroupSize
             : MaxChunkSize);
  const int active_group_size =
      StaticGroupSize == 0 ? group_size : StaticGroupSize;
  const int active_chunk_size =
      StaticGroupSize == 0 ? chunk_size : StaticChunkSize;
  extern __shared__ __nv_bfloat16 shared_storage[];
  const int warp_id = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;
  __nv_bfloat16* warp_x = shared_storage + warp_id * MaxChunkSize;
  __nv_bfloat16* shared_lut =
      shared_storage + kDecodeWarps * MaxChunkSize;
  const int output_index =
      static_cast<int>(blockIdx.x) * kVec2OutputsPerBlock
      + static_cast<int>(threadIdx.x) * 2;
  const int local_output = static_cast<int>(threadIdx.x) * 2;
  const int split_id = static_cast<int>(blockIdx.y);
  const int units_per_split = (work_units + SplitK - 1) / SplitK;
  const int first_unit = split_id * units_per_split;
  const int last_unit = min(first_unit + units_per_split, work_units);
  float accumulator_0 = 0.0F;
  float accumulator_1 = 0.0F;
  for (int unit = first_unit; unit < last_unit; ++unit) {
    const int k_begin = unit * active_chunk_size;
    int group;
    if constexpr (StaticGroupSize == 0) {
      group = k_begin / active_group_size;
    } else {
      group = unit / (StaticGroupSize / StaticChunkSize);
    }
#pragma unroll
    for (int index = lane_id;
         index < active_chunk_size;
         index += kWarpSize) {
      warp_x[index] = x[k_begin + index];
    }
    if (output_index < n) {
#pragma unroll
      for (int code = 0; code < LutEntries; ++code) {
        const int lut_offset = (group * LutEntries + code) * n + output_index;
        const __nv_bfloat162 pair =
            *reinterpret_cast<const __nv_bfloat162*>(
                group_lut + lut_offset);
        *reinterpret_cast<__nv_bfloat162*>(
            shared_lut + code * kVec2OutputsPerBlock + local_output) = pair;
      }
    }
    __syncwarp();
    if (output_index < n) {
#pragma unroll
      for (int block64 = 0;
           block64 < active_chunk_size / 64;
           ++block64) {
        const int packed_k = k_begin / 8 + block64 * 8;
        const uint4 bit0 = *reinterpret_cast<const uint4*>(
            packed_weight
            + int3_tiled_offset(packed_k, output_index, 0, n));
        const uint4 bit1 = *reinterpret_cast<const uint4*>(
            packed_weight
            + int3_tiled_offset(packed_k, output_index, 1, n));
        const uint4 bit2 = *reinterpret_cast<const uint4*>(
            packed_weight
            + int3_tiled_offset(packed_k, output_index, 2, n));
        const int x_index = block64 * 64;
        accumulate_lut3_vec2_word(
            bit0.x, bit1.x, bit2.x, warp_x, x_index,
            shared_lut, local_output, accumulator_0, accumulator_1);
        accumulate_lut3_vec2_word(
            bit0.y, bit1.y, bit2.y, warp_x, x_index + 16,
            shared_lut, local_output, accumulator_0, accumulator_1);
        accumulate_lut3_vec2_word(
            bit0.z, bit1.z, bit2.z, warp_x, x_index + 32,
            shared_lut, local_output, accumulator_0, accumulator_1);
        accumulate_lut3_vec2_word(
            bit0.w, bit1.w, bit2.w, warp_x, x_index + 48,
            shared_lut, local_output, accumulator_0, accumulator_1);
      }
      if (active_chunk_size % 64 != 0) {
        const int packed_k =
            k_begin / 8 + (active_chunk_size / 64) * 8;
        const uint2 bit0 = *reinterpret_cast<const uint2*>(
            packed_weight
            + int3_tiled_offset(packed_k, output_index, 0, n));
        const uint2 bit1 = *reinterpret_cast<const uint2*>(
            packed_weight
            + int3_tiled_offset(packed_k, output_index, 1, n));
        const uint2 bit2 = *reinterpret_cast<const uint2*>(
            packed_weight
            + int3_tiled_offset(packed_k, output_index, 2, n));
        const int x_index = (active_chunk_size / 64) * 64;
        accumulate_lut3_vec2_word(
            bit0.x, bit1.x, bit2.x, warp_x, x_index,
            shared_lut, local_output, accumulator_0, accumulator_1);
        accumulate_lut3_vec2_word(
            bit0.y, bit1.y, bit2.y, warp_x, x_index + 16,
            shared_lut, local_output, accumulator_0, accumulator_1);
      }
    }
    __syncwarp();
  }
  if (output_index >= n) {
    return;
  }
  if constexpr (DirectOutput) {
    if (bias != nullptr) {
      const float2 bias_pair = __bfloat1622float2(
          *reinterpret_cast<const __nv_bfloat162*>(bias + output_index));
      accumulator_0 += bias_pair.x;
      accumulator_1 += bias_pair.y;
    }
    output[output_index] = __float2bfloat16_rn(accumulator_0);
    output[output_index + 1] = __float2bfloat16_rn(accumulator_1);
  } else {
    partial[split_id * n + output_index] = accumulator_0;
    partial[split_id * n + output_index + 1] = accumulator_1;
  }
}

__device__ __forceinline__ void accumulate_mixnlq_code_pair(
    const uint8_t packed,
    const float x_even,
    const float x_odd,
    float& linear_sum,
    float& exponential_sum) {
  const unsigned int even_code =
      static_cast<unsigned int>(packed & 0x0FU);
  const unsigned int odd_code =
      static_cast<unsigned int>((packed >> 4) & 0x0FU);
  accumulate_mixnlq_code(
      even_code, x_even, linear_sum, exponential_sum);
  accumulate_mixnlq_code(
      odd_code, x_odd, linear_sum, exponential_sum);
}

__device__ __forceinline__ void accumulate_mixnlq_vec2_packed_pair(
    const uint16_t packed_pair,
    const float x_even,
    const float x_odd,
    float& linear_sum_0,
    float& exponential_sum_0,
    float& linear_sum_1,
    float& exponential_sum_1) {
  accumulate_mixnlq_code_pair(
      static_cast<uint8_t>(packed_pair & 0x00FFU),
      x_even,
      x_odd,
      linear_sum_0,
      exponential_sum_0);
  accumulate_mixnlq_code_pair(
      static_cast<uint8_t>(packed_pair >> 8),
      x_even,
      x_odd,
      linear_sum_1,
      exponential_sum_1);
}

__device__ __forceinline__ void accumulate_mixnlq_vec2_word(
    const uint32_t packed_word,
    const __nv_bfloat16* __restrict__ cached_x,
    const int x_index,
    float& linear_sum_0,
    float& exponential_sum_0,
    float& linear_sum_1,
    float& exponential_sum_1) {
  accumulate_mixnlq_vec2_packed_pair(
      static_cast<uint16_t>(packed_word & 0x0000FFFFU),
      __bfloat162float(cached_x[x_index]),
      __bfloat162float(cached_x[x_index + 1]),
      linear_sum_0,
      exponential_sum_0,
      linear_sum_1,
      exponential_sum_1);
  accumulate_mixnlq_vec2_packed_pair(
      static_cast<uint16_t>(packed_word >> 16),
      __bfloat162float(cached_x[x_index + 2]),
      __bfloat162float(cached_x[x_index + 3]),
      linear_sum_0,
      exponential_sum_0,
      linear_sum_1,
      exponential_sum_1);
}

// M=1 tiled specialization. Each thread computes two adjacent
// output channels. Offline tiling stores eight packed-K positions for the
// pair contiguously, so every lane issues one 16-byte uint4 load per tile.
// Quantization groups are divided into work units of at most 128 K values.
// This keeps the fast tiled path available for group sizes 32/64/128 and
// per-channel quantization while preserving enough Split-K parallelism.
template <int SplitK, bool DirectOutput, int StaticGroupSize>
__global__ void mixnlq_decode_vec2_tiled_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ linear_scale,
    const __nv_bfloat16* __restrict__ exp_scale,
    const __nv_bfloat16* __restrict__ bias,
    float* __restrict__ partial,
    __nv_bfloat16* __restrict__ output,
    const int n,
    const int group_size,
    const int work_units,
    const int chunk_size) {
  constexpr int MaxChunkSize = 128;
  constexpr int StaticChunkSize =
      StaticGroupSize == 0
      ? 0
      : (StaticGroupSize < MaxChunkSize
             ? StaticGroupSize
             : MaxChunkSize);
  const int active_group_size =
      StaticGroupSize == 0 ? group_size : StaticGroupSize;
  const int active_chunk_size =
      StaticGroupSize == 0 ? chunk_size : StaticChunkSize;
  extern __shared__ __nv_bfloat16 shared_storage[];

  const int warp_id = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;
  __nv_bfloat16* warp_x = shared_storage + warp_id * MaxChunkSize;

  const int output_index =
      static_cast<int>(blockIdx.x) * kVec2OutputsPerBlock
      + static_cast<int>(threadIdx.x) * kVec2OutputsPerThread;
  const int split_id = static_cast<int>(blockIdx.y);
  const int units_per_split = (work_units + SplitK - 1) / SplitK;
  const int first_unit = split_id * units_per_split;
  const int last_unit = min(first_unit + units_per_split, work_units);
  float accumulator_0 = 0.0F;
  float accumulator_1 = 0.0F;

  for (int unit = first_unit; unit < last_unit; ++unit) {
    const int k_begin = unit * active_chunk_size;
    int group;
    if constexpr (StaticGroupSize == 0) {
      group = k_begin / active_group_size;
    } else {
      group = unit / (StaticGroupSize / StaticChunkSize);
    }
#pragma unroll
    for (int index = lane_id;
         index < active_chunk_size;
         index += kWarpSize) {
      warp_x[index] = x[k_begin + index];
    }
    __syncwarp();

    if (output_index < n) {
      const int scale_offset = group * n + output_index;
      const __nv_bfloat162 linear_pair =
          *reinterpret_cast<const __nv_bfloat162*>(
              linear_scale + scale_offset);
      const __nv_bfloat162 exponential_pair =
          *reinterpret_cast<const __nv_bfloat162*>(
              exp_scale + scale_offset);
      const float2 linear_values =
          __bfloat1622float2(linear_pair);
      const float2 exponential_values =
          __bfloat1622float2(exponential_pair);
      const float linear_0 = linear_values.x;
      const float linear_1 = linear_values.y;
      const float exponential_0 = exponential_values.x;
      const float exponential_1 = exponential_values.y;
      float linear_sum_0 = 0.0F;
      float linear_sum_1 = 0.0F;
      float exponential_sum_0 = 0.0F;
      float exponential_sum_1 = 0.0F;

#pragma unroll
      for (int packed_tile = 0;
           packed_tile < active_chunk_size / (2 * kMixPackedKtile);
           ++packed_tile) {
        const int packed_k = k_begin / 2
            + packed_tile * kMixPackedKtile;
        const int tile_offset =
            mixnlq_tiled_offset(packed_k, output_index, n);
        const uint4 tile = *reinterpret_cast<const uint4*>(
            packed_weight + tile_offset);
        const int x_index = packed_tile * kMixPackedKtile * 2;
        accumulate_mixnlq_vec2_word(
            tile.x,
            warp_x,
            x_index,
            linear_sum_0,
            exponential_sum_0,
            linear_sum_1,
            exponential_sum_1);
        accumulate_mixnlq_vec2_word(
            tile.y,
            warp_x,
            x_index + 4,
            linear_sum_0,
            exponential_sum_0,
            linear_sum_1,
            exponential_sum_1);
        accumulate_mixnlq_vec2_word(
            tile.z,
            warp_x,
            x_index + 8,
            linear_sum_0,
            exponential_sum_0,
            linear_sum_1,
            exponential_sum_1);
        accumulate_mixnlq_vec2_word(
            tile.w,
            warp_x,
            x_index + 12,
            linear_sum_0,
            exponential_sum_0,
            linear_sum_1,
            exponential_sum_1);
      }
      accumulator_0 = fmaf(linear_0, linear_sum_0, accumulator_0);
      accumulator_0 =
          fmaf(exponential_0, exponential_sum_0, accumulator_0);
      accumulator_1 = fmaf(linear_1, linear_sum_1, accumulator_1);
      accumulator_1 =
          fmaf(exponential_1, exponential_sum_1, accumulator_1);
    }
    __syncwarp();
  }

  if (output_index >= n) {
    return;
  }
  if constexpr (DirectOutput) {
    if (bias != nullptr) {
      accumulator_0 += __bfloat162float(bias[output_index]);
      accumulator_1 += __bfloat162float(bias[output_index + 1]);
    }
    output[output_index] = __float2bfloat16_rn(accumulator_0);
    output[output_index + 1] = __float2bfloat16_rn(accumulator_1);
  } else {
    partial[split_id * n + output_index] = accumulator_0;
    partial[split_id * n + output_index + 1] = accumulator_1;
  }
}

__device__ __forceinline__ void accumulate_linear_vec2_word(
    const uint32_t packed_word,
    const __nv_bfloat16* __restrict__ cached_x,
    const int x_index,
    float& q_sum_0,
    float& q_sum_1,
    float& x_sum) {
#pragma unroll
  for (int pair = 0; pair < 2; ++pair) {
    const uint16_t packed_pair = static_cast<uint16_t>(
        packed_word >> (pair * 16));
    const uint8_t packed_0 =
        static_cast<uint8_t>(packed_pair & 0x00FFU);
    const uint8_t packed_1 =
        static_cast<uint8_t>(packed_pair >> 8);
    const int local_x = x_index + pair * 2;
    const float x_even = __bfloat162float(cached_x[local_x]);
    const float x_odd = __bfloat162float(cached_x[local_x + 1]);
    q_sum_0 = fmaf(
        x_even, static_cast<float>(packed_0 & 0x0FU), q_sum_0);
    q_sum_0 = fmaf(
        x_odd, static_cast<float>((packed_0 >> 4) & 0x0FU), q_sum_0);
    q_sum_1 = fmaf(
        x_even, static_cast<float>(packed_1 & 0x0FU), q_sum_1);
    q_sum_1 = fmaf(
        x_odd, static_cast<float>((packed_1 >> 4) & 0x0FU), q_sum_1);
    x_sum += x_even + x_odd;
  }
}

template <int SplitK, bool DirectOutput, int StaticGroupSize>
__global__ void linear_decode_vec2_tiled_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ scale,
    const __nv_bfloat16* __restrict__ zero_point,
    const __nv_bfloat16* __restrict__ bias,
    float* __restrict__ partial,
    __nv_bfloat16* __restrict__ output,
    const int n,
    const int group_size,
    const int work_units,
    const int chunk_size) {
  constexpr int MaxChunkSize = 128;
  constexpr int StaticChunkSize =
      StaticGroupSize == 0
      ? 0
      : (StaticGroupSize < MaxChunkSize
             ? StaticGroupSize
             : MaxChunkSize);
  const int active_group_size =
      StaticGroupSize == 0 ? group_size : StaticGroupSize;
  const int active_chunk_size =
      StaticGroupSize == 0 ? chunk_size : StaticChunkSize;
  extern __shared__ __nv_bfloat16 shared_storage[];

  const int warp_id = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;
  __nv_bfloat16* warp_x = shared_storage + warp_id * MaxChunkSize;
  const int output_index =
      static_cast<int>(blockIdx.x) * kVec2OutputsPerBlock
      + static_cast<int>(threadIdx.x) * kVec2OutputsPerThread;
  const int split_id = static_cast<int>(blockIdx.y);
  const int units_per_split = (work_units + SplitK - 1) / SplitK;
  const int first_unit = split_id * units_per_split;
  const int last_unit = min(first_unit + units_per_split, work_units);
  float accumulator_0 = 0.0F;
  float accumulator_1 = 0.0F;

  for (int unit = first_unit; unit < last_unit; ++unit) {
    const int k_begin = unit * active_chunk_size;
    int group;
    if constexpr (StaticGroupSize == 0) {
      group = k_begin / active_group_size;
    } else {
      group = unit / (StaticGroupSize / StaticChunkSize);
    }
#pragma unroll
    for (int index = lane_id;
         index < active_chunk_size;
         index += kWarpSize) {
      warp_x[index] = x[k_begin + index];
    }
    __syncwarp();

    if (output_index < n) {
      const int parameter_offset = group * n + output_index;
      const float scale_0 = __bfloat162float(scale[parameter_offset]);
      const float scale_1 = __bfloat162float(scale[parameter_offset + 1]);
      const float zero_0 =
          __bfloat162float(zero_point[parameter_offset]);
      const float zero_1 =
          __bfloat162float(zero_point[parameter_offset + 1]);
      float q_sum_0 = 0.0F;
      float q_sum_1 = 0.0F;
      float x_sum = 0.0F;

#pragma unroll
      for (int packed_tile = 0;
           packed_tile < active_chunk_size / (2 * kMixPackedKtile);
           ++packed_tile) {
        const int packed_k =
            k_begin / 2 + packed_tile * kMixPackedKtile;
        const int tile_offset =
            mixnlq_tiled_offset(packed_k, output_index, n);
        const uint4 tile = *reinterpret_cast<const uint4*>(
            packed_weight + tile_offset);
        const int x_index = packed_tile * kMixPackedKtile * 2;
        accumulate_linear_vec2_word(
            tile.x, warp_x, x_index, q_sum_0, q_sum_1, x_sum);
        accumulate_linear_vec2_word(
            tile.y, warp_x, x_index + 4, q_sum_0, q_sum_1, x_sum);
        accumulate_linear_vec2_word(
            tile.z, warp_x, x_index + 8, q_sum_0, q_sum_1, x_sum);
        accumulate_linear_vec2_word(
            tile.w, warp_x, x_index + 12, q_sum_0, q_sum_1, x_sum);
      }
      accumulator_0 = fmaf(
          scale_0, q_sum_0 - zero_0 * x_sum, accumulator_0);
      accumulator_1 = fmaf(
          scale_1, q_sum_1 - zero_1 * x_sum, accumulator_1);
    }
    __syncwarp();
  }

  if (output_index >= n) {
    return;
  }
  if constexpr (DirectOutput) {
    if (bias != nullptr) {
      accumulator_0 += __bfloat162float(bias[output_index]);
      accumulator_1 += __bfloat162float(bias[output_index + 1]);
    }
    output[output_index] = __float2bfloat16_rn(accumulator_0);
    output[output_index + 1] = __float2bfloat16_rn(accumulator_1);
  } else {
    partial[split_id * n + output_index] = accumulator_0;
    partial[split_id * n + output_index + 1] = accumulator_1;
  }
}

__device__ __forceinline__ void accumulate_lut_vec2_word(
    const uint32_t packed_word,
    const __nv_bfloat16* __restrict__ cached_x,
    const int x_index,
    const __nv_bfloat16* __restrict__ lut_0,
    const __nv_bfloat16* __restrict__ lut_1,
    float& accumulator_0,
    float& accumulator_1) {
#pragma unroll
  for (int pair = 0; pair < 2; ++pair) {
    const uint16_t packed_pair = static_cast<uint16_t>(
        packed_word >> (pair * 16));
    const uint8_t packed_0 =
        static_cast<uint8_t>(packed_pair & 0x00FFU);
    const uint8_t packed_1 =
        static_cast<uint8_t>(packed_pair >> 8);
    const int local_x = x_index + pair * 2;
    const float x_even = __bfloat162float(cached_x[local_x]);
    const float x_odd = __bfloat162float(cached_x[local_x + 1]);
    accumulator_0 = fmaf(
        x_even,
        __bfloat162float(lut_0[packed_0 & 0x0FU]),
        accumulator_0);
    accumulator_0 = fmaf(
        x_odd,
        __bfloat162float(lut_0[(packed_0 >> 4) & 0x0FU]),
        accumulator_0);
    accumulator_1 = fmaf(
        x_even,
        __bfloat162float(lut_1[packed_1 & 0x0FU]),
        accumulator_1);
    accumulator_1 = fmaf(
        x_odd,
        __bfloat162float(lut_1[(packed_1 >> 4) & 0x0FU]),
        accumulator_1);
  }
}

template <int SplitK, bool DirectOutput, int StaticGroupSize>
__global__ void lut_decode_vec2_tiled_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ group_lut,
    const __nv_bfloat16* __restrict__ bias,
    float* __restrict__ partial,
    __nv_bfloat16* __restrict__ output,
    const int n,
    const int group_size,
    const int work_units,
    const int chunk_size) {
  constexpr int MaxChunkSize = 128;
  constexpr int StaticChunkSize =
      StaticGroupSize == 0
      ? 0
      : (StaticGroupSize < MaxChunkSize
             ? StaticGroupSize
             : MaxChunkSize);
  const int active_group_size =
      StaticGroupSize == 0 ? group_size : StaticGroupSize;
  const int active_chunk_size =
      StaticGroupSize == 0 ? chunk_size : StaticChunkSize;
  constexpr int LutValuesPerThread = 2 * 16;
  extern __shared__ __nv_bfloat16 shared_storage[];

  const int warp_id = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;
  __nv_bfloat16* warp_x = shared_storage + warp_id * MaxChunkSize;
  __nv_bfloat16* shared_lut =
      shared_storage + kDecodeWarps * MaxChunkSize;
  const int output_index =
      static_cast<int>(blockIdx.x) * kVec2OutputsPerBlock
      + static_cast<int>(threadIdx.x) * kVec2OutputsPerThread;
  const int split_id = static_cast<int>(blockIdx.y);
  const int units_per_split = (work_units + SplitK - 1) / SplitK;
  const int first_unit = split_id * units_per_split;
  const int last_unit = min(first_unit + units_per_split, work_units);
  float accumulator_0 = 0.0F;
  float accumulator_1 = 0.0F;

  for (int unit = first_unit; unit < last_unit; ++unit) {
    const int k_begin = unit * active_chunk_size;
    int group;
    if constexpr (StaticGroupSize == 0) {
      group = k_begin / active_group_size;
    } else {
      group = unit / (StaticGroupSize / StaticChunkSize);
    }
#pragma unroll
    for (int index = lane_id;
         index < active_chunk_size;
         index += kWarpSize) {
      warp_x[index] = x[k_begin + index];
    }
    for (int index = threadIdx.x;
         index < blockDim.x * LutValuesPerThread;
         index += blockDim.x) {
      const int local_output = index >> 4;
      const int code = index & 0x0F;
      const int global_output =
          static_cast<int>(blockIdx.x) * kVec2OutputsPerBlock
          + local_output;
      shared_lut[index] = global_output < n
          ? group_lut[(group * n + global_output) * 16 + code]
          : __float2bfloat16_rn(0.0F);
    }
    __syncthreads();

    if (output_index < n) {
      const __nv_bfloat16* lut_0 =
          shared_lut + threadIdx.x * LutValuesPerThread;
      const __nv_bfloat16* lut_1 = lut_0 + 16;
#pragma unroll
      for (int packed_tile = 0;
           packed_tile < active_chunk_size / (2 * kMixPackedKtile);
           ++packed_tile) {
        const int packed_k =
            k_begin / 2 + packed_tile * kMixPackedKtile;
        const int tile_offset =
            mixnlq_tiled_offset(packed_k, output_index, n);
        const uint4 tile = *reinterpret_cast<const uint4*>(
            packed_weight + tile_offset);
        const int x_index = packed_tile * kMixPackedKtile * 2;
        accumulate_lut_vec2_word(
            tile.x, warp_x, x_index, lut_0, lut_1,
            accumulator_0, accumulator_1);
        accumulate_lut_vec2_word(
            tile.y, warp_x, x_index + 4, lut_0, lut_1,
            accumulator_0, accumulator_1);
        accumulate_lut_vec2_word(
            tile.z, warp_x, x_index + 8, lut_0, lut_1,
            accumulator_0, accumulator_1);
        accumulate_lut_vec2_word(
            tile.w, warp_x, x_index + 12, lut_0, lut_1,
            accumulator_0, accumulator_1);
      }
    }
    __syncthreads();
  }

  if (output_index >= n) {
    return;
  }
  if constexpr (DirectOutput) {
    if (bias != nullptr) {
      accumulator_0 += __bfloat162float(bias[output_index]);
      accumulator_1 += __bfloat162float(bias[output_index + 1]);
    }
    output[output_index] = __float2bfloat16_rn(accumulator_0);
    output[output_index + 1] = __float2bfloat16_rn(accumulator_1);
  } else {
    partial[split_id * n + output_index] = accumulator_0;
    partial[split_id * n + output_index + 1] = accumulator_1;
  }
}

template <int SplitK, bool DirectOutput>
__global__ void linear_decode_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ scale,
    const __nv_bfloat16* __restrict__ zero_point,
    const __nv_bfloat16* __restrict__ bias,
    float* __restrict__ partial,
    __nv_bfloat16* __restrict__ output,
    const int k,
    const int n,
    const int group_size,
    const int groups) {
  extern __shared__ __nv_bfloat16 shared_x[];

  const int output_index =
      static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int split_id = static_cast<int>(blockIdx.y);
  const int groups_per_split = (groups + SplitK - 1) / SplitK;
  const int first_group = split_id * groups_per_split;
  const int last_group = min(first_group + groups_per_split, groups);
  float accumulator = 0.0F;

  for (int group = first_group; group < last_group; ++group) {
    const int k_begin = group * group_size;
    for (int index = threadIdx.x; index < group_size; index += blockDim.x) {
      shared_x[index] = x[k_begin + index];
    }
    __syncthreads();

    if (output_index < n) {
      const int parameter_offset = group * n + output_index;
      const float group_scale = __bfloat162float(scale[parameter_offset]);
      const float group_zero =
          __bfloat162float(zero_point[parameter_offset]);
      const int packed_count = group_size / 2;
      float q_sum = 0.0F;
      float x_sum = 0.0F;

      for (int packed_index = 0; packed_index < packed_count; ++packed_index) {
        const uint8_t packed = packed_weight[mixnlq_tiled_offset(
            k_begin / 2 + packed_index, output_index, n)];
        const int x_index = packed_index * 2;
        const float x_even = __bfloat162float(shared_x[x_index]);
        const float x_odd = __bfloat162float(shared_x[x_index + 1]);
        const float q_even = static_cast<float>(packed & 0x0FU);
        const float q_odd = static_cast<float>((packed >> 4) & 0x0FU);
        q_sum = fmaf(x_even, q_even, q_sum);
        q_sum = fmaf(x_odd, q_odd, q_sum);
        x_sum += x_even + x_odd;
      }
      accumulator = fmaf(
          group_scale, q_sum - group_zero * x_sum, accumulator);
    }
    __syncthreads();
  }

  if (output_index >= n) {
    return;
  }
  if constexpr (DirectOutput) {
    if (bias != nullptr) {
      accumulator += __bfloat162float(bias[output_index]);
    }
    output[output_index] = __float2bfloat16_rn(accumulator);
  } else {
    partial[split_id * n + output_index] = accumulator;
  }
}

template <int SplitK, bool DirectOutput>
__global__ void lut_decode_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ group_lut,
    const __nv_bfloat16* __restrict__ bias,
    float* __restrict__ partial,
    __nv_bfloat16* __restrict__ output,
    const int k,
    const int n,
    const int group_size,
    const int groups) {
  extern __shared__ __nv_bfloat16 shared_storage[];
  __nv_bfloat16* shared_x = shared_storage;
  __nv_bfloat16* shared_lut = shared_storage + group_size;

  const int output_index =
      static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int split_id = static_cast<int>(blockIdx.y);
  const int groups_per_split = (groups + SplitK - 1) / SplitK;
  const int first_group = split_id * groups_per_split;
  const int last_group = min(first_group + groups_per_split, groups);
  float accumulator = 0.0F;

  for (int group = first_group; group < last_group; ++group) {
    const int k_begin = group * group_size;
    for (int index = threadIdx.x; index < group_size; index += blockDim.x) {
      shared_x[index] = x[k_begin + index];
    }

    // Stage the output-channel-local 16-entry tables with contiguous loads.
    // The shared layout is [thread/output, code].
    for (int index = threadIdx.x;
         index < blockDim.x * 16;
         index += blockDim.x) {
      const int local_output = index >> 4;
      const int code = index & 0x0F;
      const int global_output =
          static_cast<int>(blockIdx.x) * blockDim.x + local_output;
      shared_lut[index] = global_output < n
          ? group_lut[(group * n + global_output) * 16 + code]
          : __float2bfloat16_rn(0.0F);
    }
    __syncthreads();

    if (output_index < n) {
      const int packed_count = group_size / 2;
      const int local_lut_base = threadIdx.x * 16;

      for (int packed_index = 0; packed_index < packed_count; ++packed_index) {
        const uint8_t packed = packed_weight[mixnlq_tiled_offset(
            k_begin / 2 + packed_index, output_index, n)];
        const int x_index = packed_index * 2;
        const unsigned int even_code =
            static_cast<unsigned int>(packed & 0x0FU);
        const unsigned int odd_code =
            static_cast<unsigned int>((packed >> 4) & 0x0FU);
        const float x_even = __bfloat162float(shared_x[x_index]);
        const float x_odd = __bfloat162float(shared_x[x_index + 1]);
        const float weight_even =
            __bfloat162float(shared_lut[local_lut_base + even_code]);
        const float weight_odd =
            __bfloat162float(shared_lut[local_lut_base + odd_code]);
        accumulator = fmaf(x_even, weight_even, accumulator);
        accumulator = fmaf(x_odd, weight_odd, accumulator);
      }
    }
    __syncthreads();
  }

  if (output_index >= n) {
    return;
  }
  if constexpr (DirectOutput) {
    if (bias != nullptr) {
      accumulator += __bfloat162float(bias[output_index]);
    }
    output[output_index] = __float2bfloat16_rn(accumulator);
  } else {
    partial[split_id * n + output_index] = accumulator;
  }
}

template <int SplitK>
__global__ void reduce_splitk_kernel(
    const float* __restrict__ partial,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    const int n) {
  const int output_index =
      static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (output_index >= n) {
    return;
  }
  float value = 0.0F;
#pragma unroll
  for (int split = 0; split < SplitK; ++split) {
    value += partial[split * n + output_index];
  }
  if (bias != nullptr) {
    value += __bfloat162float(bias[output_index]);
  }
  output[output_index] = __float2bfloat16_rn(value);
}

int choose_split_k(const int64_t n, const int64_t groups, int64_t requested) {
  int split = static_cast<int>(requested);
  if (split == 0) {
    if (n >= 16384) {
      split = 1;
    } else if (n >= 8192) {
      split = 2;
    } else if (n >= 6144) {
      split = 4;
    } else {
      split = 8;
    }
  }
  TORCH_CHECK(
      split == 1 || split == 2 || split == 4 || split == 8
          || split == 16 || split == 32 || split == 64,
      "split_k must be 0 (auto), 1, 2, 4, 8, 16, 32, or 64");
  while (split > groups && split > 1) {
    split /= 2;
  }
  return split;
}

int choose_mixnlq_vec2_split_k(
    const int64_t k,
    const int64_t n,
    const int64_t groups,
    const int64_t requested) {
  int split = static_cast<int>(requested);
  if (split == 0) {
    // Keep roughly two or more CTAs per A100 SM for the four Llama projection
    // shapes while avoiding excessive partial-result traffic.
    if (n >= 16384) {
      split = 2;
    } else if (n >= 6144) {
      split = 8;
    } else if (k >= 8192) {
      split = 16;
    } else {
      split = 16;
    }
  }
  TORCH_CHECK(
      split == 1 || split == 2 || split == 4 || split == 8
          || split == 16 || split == 32 || split == 64,
      "vec2 split_k must be 0 (auto), 1, 2, 4, 8, 16, 32, or 64");
  while (split > groups && split > 1) {
    split /= 2;
  }
  return split;
}

bool supports_tiled_vec2_group(const int64_t group_size) {
  // A work unit must contain whole 16-value packed tiles and may not cross a
  // quantization-group boundary. Any group size divisible by 16 is supported;
  // this includes common 32/64/128/256 settings and per-channel K.
  return group_size >= 16 && group_size % 16 == 0;
}

int tiled_vec2_chunk_size(const int64_t group_size) {
  // Choose the largest packed-tile-aligned divisor not exceeding 128. This
  // lets unusual groups such as 96/160/192 use the same kernel without a work
  // unit crossing into the next group's quantization parameters.
  int chunk_size = 128;
  while (group_size % chunk_size != 0) {
    chunk_size /= 2;
  }
  return chunk_size;
}

template <int SplitK>
void launch_mixnlq_decode(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& linear_scale,
    const torch::Tensor& exp_scale,
    const torch::Tensor* bias,
    torch::Tensor& output,
    torch::Tensor* partial,
    const int k,
    const int n,
    const int group_size,
    const int groups,
    cudaStream_t stream) {
  const dim3 block(kDecodeThreads);
  const dim3 grid((n + kDecodeThreads - 1) / kDecodeThreads, SplitK);
  const size_t shared_bytes =
      static_cast<size_t>(group_size) * sizeof(__nv_bfloat16);
  const auto* bias_ptr = bias == nullptr
      ? nullptr
      : reinterpret_cast<const __nv_bfloat16*>(
            bias->data_ptr<at::BFloat16>());

  if constexpr (SplitK == 1) {
    mixnlq_decode_kernel<1, true>
        <<<grid, block, shared_bytes, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                x.data_ptr<at::BFloat16>()),
            packed_weight.data_ptr<uint8_t>(),
            reinterpret_cast<const __nv_bfloat16*>(
                linear_scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                exp_scale.data_ptr<at::BFloat16>()),
            bias_ptr,
            nullptr,
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()),
            k,
            n,
            group_size,
            groups);
  } else {
    mixnlq_decode_kernel<SplitK, false>
        <<<grid, block, shared_bytes, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                x.data_ptr<at::BFloat16>()),
            packed_weight.data_ptr<uint8_t>(),
            reinterpret_cast<const __nv_bfloat16*>(
                linear_scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                exp_scale.data_ptr<at::BFloat16>()),
            bias_ptr,
            partial->data_ptr<float>(),
            nullptr,
            k,
            n,
            group_size,
            groups);
    const dim3 reduce_grid((n + kReduceThreads - 1) / kReduceThreads);
    reduce_splitk_kernel<SplitK>
        <<<reduce_grid, kReduceThreads, 0, stream>>>(
            partial->data_ptr<float>(),
            bias_ptr,
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()),
            n);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int SplitK>
void launch_mixnlq3_decode(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& linear_scale,
    const torch::Tensor& exp_scale,
    const torch::Tensor& quant_bias,
    const torch::Tensor* output_bias,
    torch::Tensor& output,
    torch::Tensor* partial,
    const int k,
    const int n,
    const int group_size,
    const int groups,
    cudaStream_t stream) {
  const dim3 block(kDecodeThreads);
  const dim3 grid((n + kDecodeThreads - 1) / kDecodeThreads, SplitK);
  const size_t shared_bytes =
      static_cast<size_t>(group_size) * sizeof(__nv_bfloat16);
  const auto* output_bias_ptr = output_bias == nullptr
      ? nullptr
      : reinterpret_cast<const __nv_bfloat16*>(
            output_bias->data_ptr<at::BFloat16>());

  if constexpr (SplitK == 1) {
    mixnlq3_decode_kernel<1, true>
        <<<grid, block, shared_bytes, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                x.data_ptr<at::BFloat16>()),
            packed_weight.data_ptr<uint8_t>(),
            reinterpret_cast<const __nv_bfloat16*>(
                linear_scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                exp_scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                quant_bias.data_ptr<at::BFloat16>()),
            output_bias_ptr,
            nullptr,
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()),
            k,
            n,
            group_size,
            groups);
  } else {
    mixnlq3_decode_kernel<SplitK, false>
        <<<grid, block, shared_bytes, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                x.data_ptr<at::BFloat16>()),
            packed_weight.data_ptr<uint8_t>(),
            reinterpret_cast<const __nv_bfloat16*>(
                linear_scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                exp_scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                quant_bias.data_ptr<at::BFloat16>()),
            output_bias_ptr,
            partial->data_ptr<float>(),
            nullptr,
            k,
            n,
            group_size,
            groups);
    const dim3 reduce_grid((n + kReduceThreads - 1) / kReduceThreads);
    reduce_splitk_kernel<SplitK>
        <<<reduce_grid, kReduceThreads, 0, stream>>>(
            partial->data_ptr<float>(),
            output_bias_ptr,
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()),
            n);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int SplitK>
void launch_linear3_decode(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& scale,
    const torch::Tensor& zero_point,
    const torch::Tensor* bias,
    torch::Tensor& output,
    torch::Tensor* partial,
    const int n,
    const int group_size,
    const int groups,
    cudaStream_t stream) {
  const dim3 block(kDecodeThreads);
  const dim3 grid((n + kDecodeThreads - 1) / kDecodeThreads, SplitK);
  const size_t shared_bytes =
      static_cast<size_t>(group_size) * sizeof(__nv_bfloat16);
  const auto* bias_ptr = bias == nullptr
      ? nullptr
      : reinterpret_cast<const __nv_bfloat16*>(
            bias->data_ptr<at::BFloat16>());
  if constexpr (SplitK == 1) {
    linear3_decode_kernel<1, true><<<grid, block, shared_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(
            x.data_ptr<at::BFloat16>()),
        packed_weight.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(
            scale.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(
            zero_point.data_ptr<at::BFloat16>()),
        bias_ptr,
        nullptr,
        reinterpret_cast<__nv_bfloat16*>(
            output.data_ptr<at::BFloat16>()),
        n,
        group_size,
        groups);
  } else {
    linear3_decode_kernel<SplitK, false>
        <<<grid, block, shared_bytes, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                x.data_ptr<at::BFloat16>()),
            packed_weight.data_ptr<uint8_t>(),
            reinterpret_cast<const __nv_bfloat16*>(
                scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                zero_point.data_ptr<at::BFloat16>()),
            bias_ptr,
            partial->data_ptr<float>(),
            nullptr,
            n,
            group_size,
            groups);
    const dim3 reduce_grid((n + kReduceThreads - 1) / kReduceThreads);
    reduce_splitk_kernel<SplitK>
        <<<reduce_grid, kReduceThreads, 0, stream>>>(
            partial->data_ptr<float>(),
            bias_ptr,
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()),
            n);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int SplitK>
void launch_lut3_decode(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& group_lut,
    const torch::Tensor* bias,
    torch::Tensor& output,
    torch::Tensor* partial,
    const int n,
    const int group_size,
    const int groups,
    cudaStream_t stream) {
  const dim3 block(kDecodeThreads);
  const dim3 grid((n + kDecodeThreads - 1) / kDecodeThreads, SplitK);
  const size_t shared_bytes =
      static_cast<size_t>(group_size) * sizeof(__nv_bfloat16);
  const auto* bias_ptr = bias == nullptr
      ? nullptr
      : reinterpret_cast<const __nv_bfloat16*>(
            bias->data_ptr<at::BFloat16>());
  if constexpr (SplitK == 1) {
    lut3_decode_kernel<1, true><<<grid, block, shared_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(
            x.data_ptr<at::BFloat16>()),
        packed_weight.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(
            group_lut.data_ptr<at::BFloat16>()),
        bias_ptr,
        nullptr,
        reinterpret_cast<__nv_bfloat16*>(
            output.data_ptr<at::BFloat16>()),
        n,
        group_size,
        groups);
  } else {
    lut3_decode_kernel<SplitK, false>
        <<<grid, block, shared_bytes, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                x.data_ptr<at::BFloat16>()),
            packed_weight.data_ptr<uint8_t>(),
            reinterpret_cast<const __nv_bfloat16*>(
                group_lut.data_ptr<at::BFloat16>()),
            bias_ptr,
            partial->data_ptr<float>(),
            nullptr,
            n,
            group_size,
            groups);
    const dim3 reduce_grid((n + kReduceThreads - 1) / kReduceThreads);
    reduce_splitk_kernel<SplitK>
        <<<reduce_grid, kReduceThreads, 0, stream>>>(
            partial->data_ptr<float>(),
            bias_ptr,
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()),
            n);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int SplitK, int StaticGroupSize>
void launch_mixnlq3_decode_vec2_specialized(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& linear_scale,
    const torch::Tensor& exp_scale,
    const torch::Tensor& quant_bias,
    const torch::Tensor* bias,
    torch::Tensor& output,
    torch::Tensor* partial,
    const int n,
    const int group_size,
    const int work_units,
    const int chunk_size,
    cudaStream_t stream) {
  const dim3 block(kDecodeThreads);
  const dim3 grid(
      (n + kVec2OutputsPerBlock - 1) / kVec2OutputsPerBlock,
      SplitK);
  const size_t shared_bytes =
      static_cast<size_t>(kDecodeWarps * 128)
      * sizeof(__nv_bfloat16);
  const auto* bias_ptr = bias == nullptr
      ? nullptr
      : reinterpret_cast<const __nv_bfloat16*>(
            bias->data_ptr<at::BFloat16>());
#define MIXNLQ3_VEC2_ARGS                                            \
  reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),\
  packed_weight.data_ptr<uint8_t>(),                                 \
  reinterpret_cast<const __nv_bfloat16*>(                            \
      linear_scale.data_ptr<at::BFloat16>()),                        \
  reinterpret_cast<const __nv_bfloat16*>(                            \
      exp_scale.data_ptr<at::BFloat16>()),                           \
  reinterpret_cast<const __nv_bfloat16*>(                            \
      quant_bias.data_ptr<at::BFloat16>()),                          \
  bias_ptr
  if constexpr (SplitK == 1) {
    mixnlq3_decode_vec2_kernel<1, true, StaticGroupSize>
        <<<grid, block, shared_bytes, stream>>>(
            MIXNLQ3_VEC2_ARGS,
            nullptr,
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()),
            n, group_size, work_units, chunk_size);
  } else {
    mixnlq3_decode_vec2_kernel<SplitK, false, StaticGroupSize>
        <<<grid, block, shared_bytes, stream>>>(
            MIXNLQ3_VEC2_ARGS,
            partial->data_ptr<float>(),
            nullptr,
            n, group_size, work_units, chunk_size);
    const dim3 reduce_grid((n + kReduceThreads - 1) / kReduceThreads);
    reduce_splitk_kernel<SplitK>
        <<<reduce_grid, kReduceThreads, 0, stream>>>(
            partial->data_ptr<float>(), bias_ptr,
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()), n);
  }
#undef MIXNLQ3_VEC2_ARGS
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int SplitK, int StaticGroupSize>
void launch_linear3_decode_vec2_specialized(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& scale,
    const torch::Tensor& zero_point,
    const torch::Tensor* bias,
    torch::Tensor& output,
    torch::Tensor* partial,
    const int n,
    const int group_size,
    const int work_units,
    const int chunk_size,
    cudaStream_t stream) {
  const dim3 block(kDecodeThreads);
  const dim3 grid(
      (n + kVec2OutputsPerBlock - 1) / kVec2OutputsPerBlock,
      SplitK);
  const size_t shared_bytes =
      static_cast<size_t>(kDecodeWarps * 128)
      * sizeof(__nv_bfloat16);
  const auto* bias_ptr = bias == nullptr
      ? nullptr
      : reinterpret_cast<const __nv_bfloat16*>(
            bias->data_ptr<at::BFloat16>());
#define LINEAR3_VEC2_ARGS                                            \
  reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),\
  packed_weight.data_ptr<uint8_t>(),                                 \
  reinterpret_cast<const __nv_bfloat16*>(                            \
      scale.data_ptr<at::BFloat16>()),                               \
  reinterpret_cast<const __nv_bfloat16*>(                            \
      zero_point.data_ptr<at::BFloat16>()),                          \
  bias_ptr
  if constexpr (SplitK == 1) {
    linear3_decode_vec2_kernel<1, true, StaticGroupSize>
        <<<grid, block, shared_bytes, stream>>>(
            LINEAR3_VEC2_ARGS, nullptr,
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()),
            n, group_size, work_units, chunk_size);
  } else {
    linear3_decode_vec2_kernel<SplitK, false, StaticGroupSize>
        <<<grid, block, shared_bytes, stream>>>(
            LINEAR3_VEC2_ARGS, partial->data_ptr<float>(), nullptr,
            n, group_size, work_units, chunk_size);
    const dim3 reduce_grid((n + kReduceThreads - 1) / kReduceThreads);
    reduce_splitk_kernel<SplitK>
        <<<reduce_grid, kReduceThreads, 0, stream>>>(
            partial->data_ptr<float>(), bias_ptr,
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()), n);
  }
#undef LINEAR3_VEC2_ARGS
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int SplitK, int StaticGroupSize>
void launch_lut3_decode_vec2_specialized(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& group_lut,
    const torch::Tensor* bias,
    torch::Tensor& output,
    torch::Tensor* partial,
    const int n,
    const int group_size,
    const int work_units,
    const int chunk_size,
    cudaStream_t stream) {
  const dim3 block(kDecodeThreads);
  const dim3 grid(
      (n + kVec2OutputsPerBlock - 1) / kVec2OutputsPerBlock,
      SplitK);
  const size_t shared_bytes =
      static_cast<size_t>(
          kDecodeWarps * 128 + 8 * kVec2OutputsPerBlock)
      * sizeof(__nv_bfloat16);
  const auto* bias_ptr = bias == nullptr
      ? nullptr
      : reinterpret_cast<const __nv_bfloat16*>(
            bias->data_ptr<at::BFloat16>());
#define LUT3_VEC2_ARGS                                               \
  reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),\
  packed_weight.data_ptr<uint8_t>(),                                 \
  reinterpret_cast<const __nv_bfloat16*>(                            \
      group_lut.data_ptr<at::BFloat16>()),                           \
  bias_ptr
  if constexpr (SplitK == 1) {
    lut3_decode_vec2_kernel<1, true, StaticGroupSize>
        <<<grid, block, shared_bytes, stream>>>(
            LUT3_VEC2_ARGS, nullptr,
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()),
            n, group_size, work_units, chunk_size);
  } else {
    lut3_decode_vec2_kernel<SplitK, false, StaticGroupSize>
        <<<grid, block, shared_bytes, stream>>>(
            LUT3_VEC2_ARGS, partial->data_ptr<float>(), nullptr,
            n, group_size, work_units, chunk_size);
    const dim3 reduce_grid((n + kReduceThreads - 1) / kReduceThreads);
    reduce_splitk_kernel<SplitK>
        <<<reduce_grid, kReduceThreads, 0, stream>>>(
            partial->data_ptr<float>(), bias_ptr,
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()), n);
  }
#undef LUT3_VEC2_ARGS
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int SplitK, int StaticGroupSize>
void launch_mixnlq_decode_vec2_tiled_specialized(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& linear_scale,
    const torch::Tensor& exp_scale,
    const torch::Tensor* bias,
    torch::Tensor& output,
    torch::Tensor* partial,
    const int n,
    const int group_size,
    const int work_units,
    const int chunk_size,
    cudaStream_t stream) {
  const dim3 block(kDecodeThreads);
  const dim3 grid(
      (n + kVec2OutputsPerBlock - 1) / kVec2OutputsPerBlock,
      SplitK);
  const size_t shared_bytes =
      static_cast<size_t>(kDecodeWarps * 128)
      * sizeof(__nv_bfloat16);
  const auto* bias_ptr = bias == nullptr
      ? nullptr
      : reinterpret_cast<const __nv_bfloat16*>(
            bias->data_ptr<at::BFloat16>());

  if constexpr (SplitK == 1) {
    mixnlq_decode_vec2_tiled_kernel<1, true, StaticGroupSize>
        <<<grid, block, shared_bytes, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                x.data_ptr<at::BFloat16>()),
            packed_weight.data_ptr<uint8_t>(),
            reinterpret_cast<const __nv_bfloat16*>(
                linear_scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                exp_scale.data_ptr<at::BFloat16>()),
            bias_ptr,
            nullptr,
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()),
            n,
            group_size,
            work_units,
            chunk_size);
  } else {
    mixnlq_decode_vec2_tiled_kernel<SplitK, false, StaticGroupSize>
        <<<grid, block, shared_bytes, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                x.data_ptr<at::BFloat16>()),
            packed_weight.data_ptr<uint8_t>(),
            reinterpret_cast<const __nv_bfloat16*>(
                linear_scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                exp_scale.data_ptr<at::BFloat16>()),
            bias_ptr,
            partial->data_ptr<float>(),
            nullptr,
            n,
            group_size,
            work_units,
            chunk_size);
    const dim3 reduce_grid((n + kReduceThreads - 1) / kReduceThreads);
    reduce_splitk_kernel<SplitK>
        <<<reduce_grid, kReduceThreads, 0, stream>>>(
            partial->data_ptr<float>(),
            bias_ptr,
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()),
            n);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int SplitK, int StaticGroupSize>
void launch_linear_decode_specialized(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& scale,
    const torch::Tensor& zero_point,
    const torch::Tensor* bias,
    torch::Tensor& output,
    torch::Tensor* partial,
    const int k,
    const int n,
    const int group_size,
    const int work_units,
    const int chunk_size,
    cudaStream_t stream) {
  const dim3 block(kDecodeThreads);
  const dim3 grid(
      (n + kVec2OutputsPerBlock - 1) / kVec2OutputsPerBlock,
      SplitK);
  const size_t shared_bytes =
      static_cast<size_t>(kDecodeWarps * 128)
      * sizeof(__nv_bfloat16);
  const auto* bias_ptr = bias == nullptr
      ? nullptr
      : reinterpret_cast<const __nv_bfloat16*>(
            bias->data_ptr<at::BFloat16>());

  if constexpr (SplitK == 1) {
    linear_decode_vec2_tiled_kernel<1, true, StaticGroupSize>
        <<<grid, block, shared_bytes, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                x.data_ptr<at::BFloat16>()),
            packed_weight.data_ptr<uint8_t>(),
            reinterpret_cast<const __nv_bfloat16*>(
                scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                zero_point.data_ptr<at::BFloat16>()),
            bias_ptr,
            nullptr,
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()),
            n,
            group_size,
            work_units,
            chunk_size);
  } else {
    linear_decode_vec2_tiled_kernel<SplitK, false, StaticGroupSize>
        <<<grid, block, shared_bytes, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                x.data_ptr<at::BFloat16>()),
            packed_weight.data_ptr<uint8_t>(),
            reinterpret_cast<const __nv_bfloat16*>(
                scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                zero_point.data_ptr<at::BFloat16>()),
            bias_ptr,
            partial->data_ptr<float>(),
            nullptr,
            n,
            group_size,
            work_units,
            chunk_size);
    const dim3 reduce_grid((n + kReduceThreads - 1) / kReduceThreads);
    reduce_splitk_kernel<SplitK>
        <<<reduce_grid, kReduceThreads, 0, stream>>>(
            partial->data_ptr<float>(),
            bias_ptr,
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()),
            n);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int SplitK, int StaticGroupSize>
void launch_lut_decode_specialized(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& group_lut,
    const torch::Tensor* bias,
    torch::Tensor& output,
    torch::Tensor* partial,
    const int k,
    const int n,
    const int group_size,
    const int work_units,
    const int chunk_size,
    cudaStream_t stream) {
  const dim3 block(kDecodeThreads);
  const dim3 grid(
      (n + kVec2OutputsPerBlock - 1) / kVec2OutputsPerBlock,
      SplitK);
  const size_t shared_bytes =
      static_cast<size_t>(
          kDecodeWarps * 128 + kDecodeThreads * 2 * 16)
      * sizeof(__nv_bfloat16);
  const auto* bias_ptr = bias == nullptr
      ? nullptr
      : reinterpret_cast<const __nv_bfloat16*>(
            bias->data_ptr<at::BFloat16>());

  if constexpr (SplitK == 1) {
    lut_decode_vec2_tiled_kernel<1, true, StaticGroupSize>
        <<<grid, block, shared_bytes, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                x.data_ptr<at::BFloat16>()),
            packed_weight.data_ptr<uint8_t>(),
            reinterpret_cast<const __nv_bfloat16*>(
                group_lut.data_ptr<at::BFloat16>()),
            bias_ptr,
            nullptr,
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()),
            n,
            group_size,
            work_units,
            chunk_size);
  } else {
    lut_decode_vec2_tiled_kernel<SplitK, false, StaticGroupSize>
        <<<grid, block, shared_bytes, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                x.data_ptr<at::BFloat16>()),
            packed_weight.data_ptr<uint8_t>(),
            reinterpret_cast<const __nv_bfloat16*>(
                group_lut.data_ptr<at::BFloat16>()),
            bias_ptr,
            partial->data_ptr<float>(),
            nullptr,
            n,
            group_size,
            work_units,
            chunk_size);
    const dim3 reduce_grid((n + kReduceThreads - 1) / kReduceThreads);
    reduce_splitk_kernel<SplitK>
        <<<reduce_grid, kReduceThreads, 0, stream>>>(
            partial->data_ptr<float>(),
            bias_ptr,
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()),
            n);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int SplitK>
void launch_mixnlq_decode_vec2_tiled(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& linear_scale,
    const torch::Tensor& exp_scale,
    const torch::Tensor* bias,
    torch::Tensor& output,
    torch::Tensor* partial,
    const int n,
    const int group_size,
    const int work_units,
    const int chunk_size,
    cudaStream_t stream) {
#define LAUNCH_MIXNLQ_STATIC_GROUP(GroupSize)                         \
  launch_mixnlq_decode_vec2_tiled_specialized<SplitK, GroupSize>(    \
      x, packed_weight, linear_scale, exp_scale, bias, output,        \
      partial, n, group_size, work_units, chunk_size, stream)
  switch (group_size) {
    case 32:
      LAUNCH_MIXNLQ_STATIC_GROUP(32);
      break;
    case 64:
      LAUNCH_MIXNLQ_STATIC_GROUP(64);
      break;
    case 128:
      LAUNCH_MIXNLQ_STATIC_GROUP(128);
      break;
    case 256:
      LAUNCH_MIXNLQ_STATIC_GROUP(256);
      break;
    default:
      LAUNCH_MIXNLQ_STATIC_GROUP(0);
      break;
  }
#undef LAUNCH_MIXNLQ_STATIC_GROUP
}

template <int SplitK>
void launch_linear_decode(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& scale,
    const torch::Tensor& zero_point,
    const torch::Tensor* bias,
    torch::Tensor& output,
    torch::Tensor* partial,
    const int k,
    const int n,
    const int group_size,
    const int work_units,
    const int chunk_size,
    cudaStream_t stream) {
#define LAUNCH_LINEAR_STATIC_GROUP(GroupSize)                         \
  launch_linear_decode_specialized<SplitK, GroupSize>(               \
      x, packed_weight, scale, zero_point, bias, output, partial,     \
      k, n, group_size, work_units, chunk_size, stream)
  switch (group_size) {
    case 32:
      LAUNCH_LINEAR_STATIC_GROUP(32);
      break;
    case 64:
      LAUNCH_LINEAR_STATIC_GROUP(64);
      break;
    case 128:
      LAUNCH_LINEAR_STATIC_GROUP(128);
      break;
    case 256:
      LAUNCH_LINEAR_STATIC_GROUP(256);
      break;
    default:
      LAUNCH_LINEAR_STATIC_GROUP(0);
      break;
  }
#undef LAUNCH_LINEAR_STATIC_GROUP
}

template <int SplitK>
void launch_lut_decode(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& group_lut,
    const torch::Tensor* bias,
    torch::Tensor& output,
    torch::Tensor* partial,
    const int k,
    const int n,
    const int group_size,
    const int work_units,
    const int chunk_size,
    cudaStream_t stream) {
#define LAUNCH_LUT_STATIC_GROUP(GroupSize)                            \
  launch_lut_decode_specialized<SplitK, GroupSize>(                  \
      x, packed_weight, group_lut, bias, output, partial, k, n,      \
      group_size, work_units, chunk_size, stream)
  switch (group_size) {
    case 32:
      LAUNCH_LUT_STATIC_GROUP(32);
      break;
    case 64:
      LAUNCH_LUT_STATIC_GROUP(64);
      break;
    case 128:
      LAUNCH_LUT_STATIC_GROUP(128);
      break;
    case 256:
      LAUNCH_LUT_STATIC_GROUP(256);
      break;
    default:
      LAUNCH_LUT_STATIC_GROUP(0);
      break;
  }
#undef LAUNCH_LUT_STATIC_GROUP
}

template <int SplitK>
void launch_mixnlq3_decode_vec2(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& linear_scale,
    const torch::Tensor& exp_scale,
    const torch::Tensor& quant_bias,
    const torch::Tensor* bias,
    torch::Tensor& output,
    torch::Tensor* partial,
    const int n,
    const int group_size,
    const int work_units,
    const int chunk_size,
    cudaStream_t stream) {
#define LAUNCH_MIXNLQ3_GROUP(GroupSize)                              \
  launch_mixnlq3_decode_vec2_specialized<SplitK, GroupSize>(         \
      x, packed_weight, linear_scale, exp_scale, quant_bias, bias,   \
      output, partial, n, group_size, work_units, chunk_size, stream)
  switch (group_size) {
    case 32: LAUNCH_MIXNLQ3_GROUP(32); break;
    case 64: LAUNCH_MIXNLQ3_GROUP(64); break;
    case 128: LAUNCH_MIXNLQ3_GROUP(128); break;
    case 256: LAUNCH_MIXNLQ3_GROUP(256); break;
    default: LAUNCH_MIXNLQ3_GROUP(0); break;
  }
#undef LAUNCH_MIXNLQ3_GROUP
}

template <int SplitK>
void launch_linear3_decode_vec2(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& scale,
    const torch::Tensor& zero_point,
    const torch::Tensor* bias,
    torch::Tensor& output,
    torch::Tensor* partial,
    const int n,
    const int group_size,
    const int work_units,
    const int chunk_size,
    cudaStream_t stream) {
#define LAUNCH_LINEAR3_GROUP(GroupSize)                              \
  launch_linear3_decode_vec2_specialized<SplitK, GroupSize>(         \
      x, packed_weight, scale, zero_point, bias, output, partial,    \
      n, group_size, work_units, chunk_size, stream)
  switch (group_size) {
    case 32: LAUNCH_LINEAR3_GROUP(32); break;
    case 64: LAUNCH_LINEAR3_GROUP(64); break;
    case 128: LAUNCH_LINEAR3_GROUP(128); break;
    case 256: LAUNCH_LINEAR3_GROUP(256); break;
    default: LAUNCH_LINEAR3_GROUP(0); break;
  }
#undef LAUNCH_LINEAR3_GROUP
}

template <int SplitK>
void launch_lut3_decode_vec2(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& group_lut,
    const torch::Tensor* bias,
    torch::Tensor& output,
    torch::Tensor* partial,
    const int n,
    const int group_size,
    const int work_units,
    const int chunk_size,
    cudaStream_t stream) {
#define LAUNCH_LUT3_GROUP(GroupSize)                                 \
  launch_lut3_decode_vec2_specialized<SplitK, GroupSize>(            \
      x, packed_weight, group_lut, bias, output, partial, n,         \
      group_size, work_units, chunk_size, stream)
  switch (group_size) {
    case 32: LAUNCH_LUT3_GROUP(32); break;
    case 64: LAUNCH_LUT3_GROUP(64); break;
    case 128: LAUNCH_LUT3_GROUP(128); break;
    case 256: LAUNCH_LUT3_GROUP(256); break;
    default: LAUNCH_LUT3_GROUP(0); break;
  }
#undef LAUNCH_LUT3_GROUP
}

}  // namespace

torch::Tensor mixnlq_decode_cuda(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& linear_scale,
    const torch::Tensor& exp_scale,
    const c10::optional<torch::Tensor>& bias,
    int64_t group_size,
    int64_t requested_split_k,
    int64_t kernel_variant) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(packed_weight.is_cuda(), "packed_weight must be a CUDA tensor");
  TORCH_CHECK(linear_scale.is_cuda(), "linear_scale must be a CUDA tensor");
  TORCH_CHECK(exp_scale.is_cuda(), "exp_scale must be a CUDA tensor");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(packed_weight.is_contiguous(), "packed_weight must be contiguous");
  TORCH_CHECK(linear_scale.is_contiguous(), "linear_scale must be contiguous");
  TORCH_CHECK(exp_scale.is_contiguous(), "exp_scale must be contiguous");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be BF16");
  TORCH_CHECK(
      linear_scale.scalar_type() == at::kBFloat16,
      "linear_scale must be BF16");
  TORCH_CHECK(
      exp_scale.scalar_type() == at::kBFloat16,
      "exp_scale must be BF16");
  TORCH_CHECK(
      packed_weight.scalar_type() == at::kByte,
      "packed_weight must be uint8");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 1, "x must have shape (1, K)");
  TORCH_CHECK(
      packed_weight.dim() == 5,
      "MixNLQ packed_weight must use the 5D N64/Kp8 tiled layout");
  TORCH_CHECK(linear_scale.dim() == 2, "linear_scale must be 2D");
  TORCH_CHECK(exp_scale.dim() == 2, "exp_scale must be 2D");

  const int64_t k64 = x.size(1);
  const int64_t n64 = linear_scale.size(1);
  TORCH_CHECK(k64 % 2 == 0, "K must be even for packed INT4");
  TORCH_CHECK(
      k64 % (2 * kMixPackedKtile) == 0,
      "K must be divisible by 16 for MixNLQ tiled weights");
  TORCH_CHECK(
      n64 % kMixNtile == 0,
      "N must be divisible by 64 for MixNLQ tiled weights");
  TORCH_CHECK(group_size > 0 && group_size % 16 == 0, "invalid group_size");
  TORCH_CHECK(k64 % group_size == 0, "K must be divisible by group_size");
  const int64_t groups64 = k64 / group_size;
  TORCH_CHECK(
      packed_weight.size(0) == k64 / (2 * kMixPackedKtile)
          && packed_weight.size(1) == n64 / kMixNtile
          && packed_weight.size(2) == kMixNtile / 2
          && packed_weight.size(3) == kMixPackedKtile
          && packed_weight.size(4) == 2,
      "MixNLQ packed_weight shape must be (K/16, N/64, 32, 8, 2)");
  TORCH_CHECK(
      linear_scale.size(0) == groups64 && linear_scale.size(1) == n64,
      "linear_scale shape must be (K/group_size, N)");
  TORCH_CHECK(
      exp_scale.sizes() == linear_scale.sizes(),
      "exp_scale shape must equal linear_scale shape");
  TORCH_CHECK(
      kernel_variant >= 0 && kernel_variant <= 2,
      "kernel_variant must be 0 (auto), 1 (scalar), or 2 (vec2)");

  const torch::Tensor* bias_ptr = nullptr;
  if (bias.has_value() && bias->defined() && bias->numel() != 0) {
    TORCH_CHECK(bias->is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(bias->scalar_type() == at::kBFloat16, "bias must be BF16");
    TORCH_CHECK(bias->numel() == n64, "bias must have N elements");
    bias_ptr = &bias.value();
  }

  at::cuda::CUDAGuard device_guard(x.device());
  const bool vec2_supported = supports_tiled_vec2_group(group_size);
  TORCH_CHECK(
      kernel_variant != 2 || vec2_supported,
      "vec2 requires group_size divisible by 16");
  const bool use_vec2 =
      kernel_variant == 2 || (kernel_variant == 0 && vec2_supported);
  const int chunk_size = use_vec2
      ? tiled_vec2_chunk_size(group_size)
      : static_cast<int>(group_size);
  const int64_t work_units64 = k64 / chunk_size;
  const int split_k = use_vec2
      ? choose_mixnlq_vec2_split_k(
            k64, n64, work_units64, requested_split_k)
      : choose_split_k(n64, groups64, requested_split_k);
  auto output = torch::empty({1, n64}, x.options());
  torch::Tensor partial;
  if (split_k != 1) {
    partial = torch::empty(
        {split_k, n64},
        x.options().dtype(torch::kFloat32));
  }

  const int k = static_cast<int>(k64);
  const int n = static_cast<int>(n64);
  const int groups = static_cast<int>(groups64);
  const int group = static_cast<int>(group_size);
  const int work_units = static_cast<int>(work_units64);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  if (use_vec2) {
    switch (split_k) {
      case 1:
        launch_mixnlq_decode_vec2_tiled<1>(
            x, packed_weight, linear_scale, exp_scale, bias_ptr, output,
            nullptr, n, group, work_units, chunk_size, stream);
        break;
      case 2:
        launch_mixnlq_decode_vec2_tiled<2>(
            x, packed_weight, linear_scale, exp_scale, bias_ptr, output,
            &partial, n, group, work_units, chunk_size, stream);
        break;
      case 4:
        launch_mixnlq_decode_vec2_tiled<4>(
            x, packed_weight, linear_scale, exp_scale, bias_ptr, output,
            &partial, n, group, work_units, chunk_size, stream);
        break;
      case 8:
        launch_mixnlq_decode_vec2_tiled<8>(
            x, packed_weight, linear_scale, exp_scale, bias_ptr, output,
            &partial, n, group, work_units, chunk_size, stream);
        break;
      case 16:
        launch_mixnlq_decode_vec2_tiled<16>(
            x, packed_weight, linear_scale, exp_scale, bias_ptr, output,
            &partial, n, group, work_units, chunk_size, stream);
        break;
      case 32:
        launch_mixnlq_decode_vec2_tiled<32>(
            x, packed_weight, linear_scale, exp_scale, bias_ptr, output,
            &partial, n, group, work_units, chunk_size, stream);
        break;
      case 64:
        launch_mixnlq_decode_vec2_tiled<64>(
            x, packed_weight, linear_scale, exp_scale, bias_ptr, output,
            &partial, n, group, work_units, chunk_size, stream);
        break;
      default:
        TORCH_CHECK(false, "unreachable vec2 split_k");
    }
    return output;
  }
  switch (split_k) {
    case 1:
      launch_mixnlq_decode<1>(
          x, packed_weight, linear_scale, exp_scale, bias_ptr, output, nullptr,
          k, n, group, groups, stream);
      break;
    case 2:
      launch_mixnlq_decode<2>(
          x, packed_weight, linear_scale, exp_scale, bias_ptr, output, &partial,
          k, n, group, groups, stream);
      break;
    case 4:
      launch_mixnlq_decode<4>(
          x, packed_weight, linear_scale, exp_scale, bias_ptr, output, &partial,
          k, n, group, groups, stream);
      break;
    case 8:
      launch_mixnlq_decode<8>(
          x, packed_weight, linear_scale, exp_scale, bias_ptr, output, &partial,
          k, n, group, groups, stream);
      break;
    case 16:
      launch_mixnlq_decode<16>(
          x, packed_weight, linear_scale, exp_scale, bias_ptr, output, &partial,
          k, n, group, groups, stream);
      break;
    case 32:
      launch_mixnlq_decode<32>(
          x, packed_weight, linear_scale, exp_scale, bias_ptr, output, &partial,
          k, n, group, groups, stream);
      break;
    case 64:
      launch_mixnlq_decode<64>(
          x, packed_weight, linear_scale, exp_scale, bias_ptr, output, &partial,
          k, n, group, groups, stream);
      break;
    default:
      TORCH_CHECK(false, "unreachable split_k");
  }
  return output;
}

torch::Tensor mixnlq3_decode_cuda(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& linear_scale,
    const torch::Tensor& exp_scale,
    const torch::Tensor& quant_bias,
    const c10::optional<torch::Tensor>& bias,
    int64_t group_size,
    int64_t requested_split_k) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(packed_weight.is_cuda(), "packed_weight must be CUDA");
  TORCH_CHECK(linear_scale.is_cuda(), "linear_scale must be CUDA");
  TORCH_CHECK(exp_scale.is_cuda(), "exp_scale must be CUDA");
  TORCH_CHECK(quant_bias.is_cuda(), "quant_bias must be CUDA");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(packed_weight.is_contiguous(), "packed_weight must be contiguous");
  TORCH_CHECK(linear_scale.is_contiguous(), "linear_scale must be contiguous");
  TORCH_CHECK(exp_scale.is_contiguous(), "exp_scale must be contiguous");
  TORCH_CHECK(quant_bias.is_contiguous(), "quant_bias must be contiguous");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be BF16");
  TORCH_CHECK(
      linear_scale.scalar_type() == at::kBFloat16
          && exp_scale.scalar_type() == at::kBFloat16
          && quant_bias.scalar_type() == at::kBFloat16,
      "MixNLQ3 scales must be BF16");
  TORCH_CHECK(
      packed_weight.scalar_type() == at::kByte,
      "packed_weight must be uint8");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 1, "x must have shape (1, K)");
  TORCH_CHECK(
      packed_weight.dim() == 6,
      "MixNLQ3 packed_weight must use K128/N256 tiled layout");
  TORCH_CHECK(
      linear_scale.dim() == 2
          && exp_scale.dim() == 2
          && quant_bias.dim() == 2,
      "MixNLQ3 scales must be 2D");

  const int64_t k64 = x.size(1);
  const int64_t n64 = linear_scale.size(1);
  TORCH_CHECK(k64 % 128 == 0, "MixNLQ3 requires K divisible by 128");
  TORCH_CHECK(n64 % 256 == 0, "MixNLQ3 requires N divisible by 256");
  TORCH_CHECK(
      supports_tiled_vec2_group(group_size),
      "MixNLQ3 vec2 requires group_size divisible by 16");
  TORCH_CHECK(k64 % group_size == 0, "K must be divisible by group_size");
  const int64_t groups64 = k64 / group_size;
  TORCH_CHECK(
      packed_weight.size(0) == k64 / 128
          && packed_weight.size(1) == n64 / 256
          && packed_weight.size(2) == 3
          && packed_weight.size(3) == 2
          && packed_weight.size(4) == 128
          && packed_weight.size(5) == 16,
      "MixNLQ3 packed_weight tiled shape mismatch");
  TORCH_CHECK(
      linear_scale.size(0) == groups64,
      "linear_scale shape must be (K/group_size, N)");
  TORCH_CHECK(
      exp_scale.sizes() == linear_scale.sizes()
          && quant_bias.sizes() == linear_scale.sizes(),
      "MixNLQ3 scale tensors must have identical shapes");

  const torch::Tensor* bias_ptr = nullptr;
  if (bias.has_value() && bias->defined() && bias->numel() != 0) {
    TORCH_CHECK(bias->is_cuda(), "bias must be CUDA");
    TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(bias->scalar_type() == at::kBFloat16, "bias must be BF16");
    TORCH_CHECK(bias->numel() == n64, "bias must have N elements");
    bias_ptr = &bias.value();
  }

  at::cuda::CUDAGuard device_guard(x.device());
  const int chunk_size = tiled_vec2_chunk_size(group_size);
  const int64_t work_units64 = k64 / chunk_size;
  const int split_k = choose_mixnlq_vec2_split_k(
      k64, n64, work_units64, requested_split_k);
  auto output = torch::empty({1, n64}, x.options());
  torch::Tensor partial;
  if (split_k != 1) {
    partial = torch::empty(
        {split_k, n64},
        x.options().dtype(torch::kFloat32));
  }
  const int n = static_cast<int>(n64);
  const int group = static_cast<int>(group_size);
  const int work_units = static_cast<int>(work_units64);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
#define LAUNCH_MIXNLQ3_VEC2(SplitK)                                 \
  launch_mixnlq3_decode_vec2<SplitK>(                               \
      x, packed_weight, linear_scale, exp_scale, quant_bias,         \
      bias_ptr, output, split_k == 1 ? nullptr : &partial, n, group, \
      work_units, chunk_size, stream)
  switch (split_k) {
    case 1: LAUNCH_MIXNLQ3_VEC2(1); break;
    case 2: LAUNCH_MIXNLQ3_VEC2(2); break;
    case 4: LAUNCH_MIXNLQ3_VEC2(4); break;
    case 8: LAUNCH_MIXNLQ3_VEC2(8); break;
    case 16: LAUNCH_MIXNLQ3_VEC2(16); break;
    case 32: LAUNCH_MIXNLQ3_VEC2(32); break;
    case 64: LAUNCH_MIXNLQ3_VEC2(64); break;
    default: TORCH_CHECK(false, "unreachable MixNLQ3 vec2 split_k");
  }
#undef LAUNCH_MIXNLQ3_VEC2
  return output;
}

torch::Tensor linear3_decode_cuda(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& scale,
    const torch::Tensor& zero_point,
    const c10::optional<torch::Tensor>& bias,
    int64_t group_size,
    int64_t requested_split_k) {
  TORCH_CHECK(x.is_cuda() && packed_weight.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(scale.is_cuda() && zero_point.is_cuda(), "parameters must be CUDA");
  TORCH_CHECK(x.is_contiguous() && packed_weight.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(scale.is_contiguous() && zero_point.is_contiguous(), "parameters must be contiguous");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be BF16");
  TORCH_CHECK(
      scale.scalar_type() == at::kBFloat16
          && zero_point.scalar_type() == at::kBFloat16,
      "Linear3 parameters must be BF16");
  TORCH_CHECK(packed_weight.scalar_type() == at::kByte, "packed_weight must be uint8");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 1, "x must have shape (1, K)");
  TORCH_CHECK(
      packed_weight.dim() == 6,
      "Linear3 packed_weight must use K128/N256 tiled layout");
  const int64_t k64 = x.size(1);
  const int64_t n64 = scale.size(1);
  TORCH_CHECK(k64 % 128 == 0, "Linear3 requires K divisible by 128");
  TORCH_CHECK(n64 % 256 == 0, "Linear3 requires N divisible by 256");
  TORCH_CHECK(
      supports_tiled_vec2_group(group_size),
      "Linear3 vec2 requires group_size divisible by 16");
  TORCH_CHECK(k64 % group_size == 0, "K must be divisible by group_size");
  const int64_t groups64 = k64 / group_size;
  TORCH_CHECK(
      packed_weight.size(0) == k64 / 128
          && packed_weight.size(1) == n64 / 256
          && packed_weight.size(2) == 3
          && packed_weight.size(3) == 2
          && packed_weight.size(4) == 128
          && packed_weight.size(5) == 16,
      "Linear3 packed_weight shape mismatch");
  TORCH_CHECK(
      scale.dim() == 2 && scale.size(0) == groups64,
      "Linear3 scale must have shape (K/group_size, N)");
  TORCH_CHECK(
      zero_point.sizes() == scale.sizes(),
      "Linear3 scale and zero_point shapes must match");
  const torch::Tensor* bias_ptr = nullptr;
  if (bias.has_value() && bias->defined() && bias->numel() != 0) {
    TORCH_CHECK(bias->is_cuda() && bias->is_contiguous(), "bias must be contiguous CUDA");
    TORCH_CHECK(bias->scalar_type() == at::kBFloat16, "bias must be BF16");
    TORCH_CHECK(bias->numel() == n64, "bias must have N elements");
    bias_ptr = &bias.value();
  }
  at::cuda::CUDAGuard device_guard(x.device());
  const int chunk_size = tiled_vec2_chunk_size(group_size);
  const int64_t work_units64 = k64 / chunk_size;
  const int split_k = choose_mixnlq_vec2_split_k(
      k64, n64, work_units64, requested_split_k);
  auto output = torch::empty({1, n64}, x.options());
  torch::Tensor partial;
  if (split_k != 1) {
    partial = torch::empty({split_k, n64}, x.options().dtype(torch::kFloat32));
  }
  const int n = static_cast<int>(n64);
  const int group = static_cast<int>(group_size);
  const int work_units = static_cast<int>(work_units64);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
#define LAUNCH_LINEAR3(SplitK)                                      \
  launch_linear3_decode_vec2<SplitK>(                               \
      x, packed_weight, scale, zero_point, bias_ptr, output,         \
      split_k == 1 ? nullptr : &partial, n, group, work_units,       \
      chunk_size, stream)
  switch (split_k) {
    case 1: LAUNCH_LINEAR3(1); break;
    case 2: LAUNCH_LINEAR3(2); break;
    case 4: LAUNCH_LINEAR3(4); break;
    case 8: LAUNCH_LINEAR3(8); break;
    case 16: LAUNCH_LINEAR3(16); break;
    case 32: LAUNCH_LINEAR3(32); break;
    case 64: LAUNCH_LINEAR3(64); break;
    default: TORCH_CHECK(false, "unreachable Linear3 vec2 split_k");
  }
#undef LAUNCH_LINEAR3
  return output;
}

torch::Tensor lut3_decode_cuda(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& group_lut,
    const c10::optional<torch::Tensor>& bias,
    int64_t group_size,
    int64_t requested_split_k) {
  TORCH_CHECK(x.is_cuda() && packed_weight.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(group_lut.is_cuda(), "group_lut must be CUDA");
  TORCH_CHECK(x.is_contiguous() && packed_weight.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(group_lut.is_contiguous(), "group_lut must be contiguous");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be BF16");
  TORCH_CHECK(group_lut.scalar_type() == at::kBFloat16, "group_lut must be BF16");
  TORCH_CHECK(packed_weight.scalar_type() == at::kByte, "packed_weight must be uint8");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 1, "x must have shape (1, K)");
  TORCH_CHECK(
      packed_weight.dim() == 6,
      "LUT3 packed_weight must use K128/N256 tiled layout");
  TORCH_CHECK(group_lut.dim() == 3 && group_lut.size(1) == 8,
      "LUT3 group_lut must have shape (K/group_size, 8, N)");
  const int64_t k64 = x.size(1);
  const int64_t n64 = group_lut.size(2);
  TORCH_CHECK(k64 % 128 == 0, "LUT3 requires K divisible by 128");
  TORCH_CHECK(n64 % 256 == 0, "LUT3 requires N divisible by 256");
  TORCH_CHECK(
      supports_tiled_vec2_group(group_size),
      "LUT3 vec2 requires group_size divisible by 16");
  TORCH_CHECK(k64 % group_size == 0, "K must be divisible by group_size");
  const int64_t groups64 = k64 / group_size;
  TORCH_CHECK(
      packed_weight.size(0) == k64 / 128
          && packed_weight.size(1) == n64 / 256
          && packed_weight.size(2) == 3
          && packed_weight.size(3) == 2
          && packed_weight.size(4) == 128
          && packed_weight.size(5) == 16,
      "LUT3 packed_weight shape mismatch");
  TORCH_CHECK(group_lut.size(0) == groups64, "LUT3 group count mismatch");
  const torch::Tensor* bias_ptr = nullptr;
  if (bias.has_value() && bias->defined() && bias->numel() != 0) {
    TORCH_CHECK(bias->is_cuda() && bias->is_contiguous(), "bias must be contiguous CUDA");
    TORCH_CHECK(bias->scalar_type() == at::kBFloat16, "bias must be BF16");
    TORCH_CHECK(bias->numel() == n64, "bias must have N elements");
    bias_ptr = &bias.value();
  }
  at::cuda::CUDAGuard device_guard(x.device());
  const int chunk_size = tiled_vec2_chunk_size(group_size);
  const int64_t work_units64 = k64 / chunk_size;
  const int split_k = choose_mixnlq_vec2_split_k(
      k64, n64, work_units64, requested_split_k);
  auto output = torch::empty({1, n64}, x.options());
  torch::Tensor partial;
  if (split_k != 1) {
    partial = torch::empty({split_k, n64}, x.options().dtype(torch::kFloat32));
  }
  const int n = static_cast<int>(n64);
  const int group = static_cast<int>(group_size);
  const int work_units = static_cast<int>(work_units64);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
#define LAUNCH_LUT3(SplitK)                                         \
  launch_lut3_decode_vec2<SplitK>(                                  \
      x, packed_weight, group_lut, bias_ptr, output,                 \
      split_k == 1 ? nullptr : &partial, n, group, work_units,       \
      chunk_size, stream)
  switch (split_k) {
    case 1: LAUNCH_LUT3(1); break;
    case 2: LAUNCH_LUT3(2); break;
    case 4: LAUNCH_LUT3(4); break;
    case 8: LAUNCH_LUT3(8); break;
    case 16: LAUNCH_LUT3(16); break;
    case 32: LAUNCH_LUT3(32); break;
    case 64: LAUNCH_LUT3(64); break;
    default: TORCH_CHECK(false, "unreachable LUT3 vec2 split_k");
  }
#undef LAUNCH_LUT3
  return output;
}

torch::Tensor linear_decode_cuda(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& scale,
    const torch::Tensor& zero_point,
    const c10::optional<torch::Tensor>& bias,
    int64_t group_size,
    int64_t requested_split_k) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(packed_weight.is_cuda(), "packed_weight must be a CUDA tensor");
  TORCH_CHECK(scale.is_cuda(), "scale must be a CUDA tensor");
  TORCH_CHECK(zero_point.is_cuda(), "zero_point must be a CUDA tensor");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(packed_weight.is_contiguous(), "packed_weight must be contiguous");
  TORCH_CHECK(scale.is_contiguous(), "scale must be contiguous");
  TORCH_CHECK(zero_point.is_contiguous(), "zero_point must be contiguous");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be BF16");
  TORCH_CHECK(scale.scalar_type() == at::kBFloat16, "scale must be BF16");
  TORCH_CHECK(
      zero_point.scalar_type() == at::kBFloat16,
      "zero_point must be BF16");
  TORCH_CHECK(
      packed_weight.scalar_type() == at::kByte,
      "packed_weight must be uint8");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 1, "x must have shape (1, K)");
  TORCH_CHECK(
      packed_weight.dim() == 5,
      "linear packed_weight must use the 5D N64/Kp8 tiled layout");
  TORCH_CHECK(scale.dim() == 2, "scale must be 2D");
  TORCH_CHECK(zero_point.dim() == 2, "zero_point must be 2D");

  const int64_t k64 = x.size(1);
  const int64_t n64 = scale.size(1);
  TORCH_CHECK(
      k64 % (2 * kMixPackedKtile) == 0,
      "K must be divisible by 16 for tiled INT4");
  TORCH_CHECK(
      n64 % kMixNtile == 0,
      "N must be divisible by 64 for tiled INT4");
  TORCH_CHECK(
      supports_tiled_vec2_group(group_size),
      "tiled CUDA Linear Decode requires group_size divisible by 16");
  TORCH_CHECK(k64 % group_size == 0, "K must be divisible by group_size");
  const int64_t groups64 = k64 / group_size;
  TORCH_CHECK(
      packed_weight.size(0) == k64 / (2 * kMixPackedKtile)
          && packed_weight.size(1) == n64 / kMixNtile
          && packed_weight.size(2) == kMixNtile / 2
          && packed_weight.size(3) == kMixPackedKtile
          && packed_weight.size(4) == 2,
      "linear packed_weight shape must be (K/16, N/64, 32, 8, 2)");
  TORCH_CHECK(
      scale.size(0) == groups64 && scale.size(1) == n64,
      "scale shape must be (K/group_size, N)");
  TORCH_CHECK(
      zero_point.sizes() == scale.sizes(),
      "zero_point shape must equal scale shape");

  const torch::Tensor* bias_ptr = nullptr;
  if (bias.has_value() && bias->defined() && bias->numel() != 0) {
    TORCH_CHECK(bias->is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(bias->scalar_type() == at::kBFloat16, "bias must be BF16");
    TORCH_CHECK(bias->numel() == n64, "bias must have N elements");
    bias_ptr = &bias.value();
  }

  at::cuda::CUDAGuard device_guard(x.device());
  const int chunk_size = tiled_vec2_chunk_size(group_size);
  const int64_t work_units64 = k64 / chunk_size;
  const int split_k = choose_mixnlq_vec2_split_k(
      k64, n64, work_units64, requested_split_k);
  auto output = torch::empty({1, n64}, x.options());
  torch::Tensor partial;
  if (split_k != 1) {
    partial = torch::empty(
        {split_k, n64},
        x.options().dtype(torch::kFloat32));
  }

  const int k = static_cast<int>(k64);
  const int n = static_cast<int>(n64);
  const int groups = static_cast<int>(groups64);
  const int group = static_cast<int>(group_size);
  const int work_units = static_cast<int>(work_units64);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  switch (split_k) {
    case 1:
      launch_linear_decode<1>(
          x, packed_weight, scale, zero_point, bias_ptr, output, nullptr,
          k, n, group, work_units, chunk_size, stream);
      break;
    case 2:
      launch_linear_decode<2>(
          x, packed_weight, scale, zero_point, bias_ptr, output, &partial,
          k, n, group, work_units, chunk_size, stream);
      break;
    case 4:
      launch_linear_decode<4>(
          x, packed_weight, scale, zero_point, bias_ptr, output, &partial,
          k, n, group, work_units, chunk_size, stream);
      break;
    case 8:
      launch_linear_decode<8>(
          x, packed_weight, scale, zero_point, bias_ptr, output, &partial,
          k, n, group, work_units, chunk_size, stream);
      break;
    case 16:
      launch_linear_decode<16>(
          x, packed_weight, scale, zero_point, bias_ptr, output, &partial,
          k, n, group, work_units, chunk_size, stream);
      break;
    case 32:
      launch_linear_decode<32>(
          x, packed_weight, scale, zero_point, bias_ptr, output, &partial,
          k, n, group, work_units, chunk_size, stream);
      break;
    case 64:
      launch_linear_decode<64>(
          x, packed_weight, scale, zero_point, bias_ptr, output, &partial,
          k, n, group, work_units, chunk_size, stream);
      break;
    default:
      TORCH_CHECK(false, "unreachable split_k");
  }
  return output;
}

torch::Tensor lut_decode_cuda(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& group_lut,
    const c10::optional<torch::Tensor>& bias,
    int64_t group_size,
    int64_t requested_split_k) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(packed_weight.is_cuda(), "packed_weight must be a CUDA tensor");
  TORCH_CHECK(group_lut.is_cuda(), "group_lut must be a CUDA tensor");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(packed_weight.is_contiguous(), "packed_weight must be contiguous");
  TORCH_CHECK(group_lut.is_contiguous(), "group_lut must be contiguous");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be BF16");
  TORCH_CHECK(
      group_lut.scalar_type() == at::kBFloat16,
      "group_lut must be BF16");
  TORCH_CHECK(
      packed_weight.scalar_type() == at::kByte,
      "packed_weight must be uint8");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 1, "x must have shape (1, K)");
  TORCH_CHECK(
      packed_weight.dim() == 5,
      "LUT packed_weight must use the 5D N64/Kp8 tiled layout");
  TORCH_CHECK(group_lut.dim() == 3, "group_lut must be 3D");

  const int64_t k64 = x.size(1);
  const int64_t n64 = group_lut.size(1);
  TORCH_CHECK(
      k64 % (2 * kMixPackedKtile) == 0,
      "K must be divisible by 16 for tiled INT4");
  TORCH_CHECK(
      n64 % kMixNtile == 0,
      "N must be divisible by 64 for tiled INT4");
  TORCH_CHECK(
      supports_tiled_vec2_group(group_size),
      "tiled CUDA LUT Decode requires group_size divisible by 16");
  TORCH_CHECK(k64 % group_size == 0, "K must be divisible by group_size");
  const int64_t groups64 = k64 / group_size;
  TORCH_CHECK(
      packed_weight.size(0) == k64 / (2 * kMixPackedKtile)
          && packed_weight.size(1) == n64 / kMixNtile
          && packed_weight.size(2) == kMixNtile / 2
          && packed_weight.size(3) == kMixPackedKtile
          && packed_weight.size(4) == 2,
      "LUT packed_weight shape must be (K/16, N/64, 32, 8, 2)");
  TORCH_CHECK(
      group_lut.size(0) == groups64
          && group_lut.size(1) == n64
          && group_lut.size(2) == 16,
      "group_lut shape must be (K/group_size, N, 16)");

  const torch::Tensor* bias_ptr = nullptr;
  if (bias.has_value() && bias->defined() && bias->numel() != 0) {
    TORCH_CHECK(bias->is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(bias->scalar_type() == at::kBFloat16, "bias must be BF16");
    TORCH_CHECK(bias->numel() == n64, "bias must have N elements");
    bias_ptr = &bias.value();
  }

  at::cuda::CUDAGuard device_guard(x.device());
  const int chunk_size = tiled_vec2_chunk_size(group_size);
  const int64_t work_units64 = k64 / chunk_size;
  const int split_k = choose_mixnlq_vec2_split_k(
      k64, n64, work_units64, requested_split_k);
  auto output = torch::empty({1, n64}, x.options());
  torch::Tensor partial;
  if (split_k != 1) {
    partial = torch::empty(
        {split_k, n64},
        x.options().dtype(torch::kFloat32));
  }

  const int k = static_cast<int>(k64);
  const int n = static_cast<int>(n64);
  const int groups = static_cast<int>(groups64);
  const int group = static_cast<int>(group_size);
  const int work_units = static_cast<int>(work_units64);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  switch (split_k) {
    case 1:
      launch_lut_decode<1>(
          x, packed_weight, group_lut, bias_ptr, output, nullptr,
          k, n, group, work_units, chunk_size, stream);
      break;
    case 2:
      launch_lut_decode<2>(
          x, packed_weight, group_lut, bias_ptr, output, &partial,
          k, n, group, work_units, chunk_size, stream);
      break;
    case 4:
      launch_lut_decode<4>(
          x, packed_weight, group_lut, bias_ptr, output, &partial,
          k, n, group, work_units, chunk_size, stream);
      break;
    case 8:
      launch_lut_decode<8>(
          x, packed_weight, group_lut, bias_ptr, output, &partial,
          k, n, group, work_units, chunk_size, stream);
      break;
    case 16:
      launch_lut_decode<16>(
          x, packed_weight, group_lut, bias_ptr, output, &partial,
          k, n, group, work_units, chunk_size, stream);
      break;
    case 32:
      launch_lut_decode<32>(
          x, packed_weight, group_lut, bias_ptr, output, &partial,
          k, n, group, work_units, chunk_size, stream);
      break;
    case 64:
      launch_lut_decode<64>(
          x, packed_weight, group_lut, bias_ptr, output, &partial,
          k, n, group, work_units, chunk_size, stream);
      break;
    default:
      TORCH_CHECK(false, "unreachable split_k");
  }
  return output;
}
