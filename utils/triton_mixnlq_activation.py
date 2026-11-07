"""Fused dynamic per-token MixNLQ activation QDQ implemented in Triton."""

import torch
import triton
import triton.language as tl


@triton.jit
def _round_to_nearest_even(values):
    """Match torch.round, including its round-half-to-even tie rule."""
    lower = tl.floor(values)
    fraction = values - lower
    rounded_up = lower + 1.0
    lower_is_odd = (lower - 2.0 * tl.floor(0.5 * lower)) != 0.0
    tie_result = tl.where(lower_is_odd, rounded_up, lower)
    return tl.where(
        fraction < 0.5,
        lower,
        tl.where(fraction > 0.5, rounded_up, tie_result),
    )


@triton.jit
def _dynamic_rtn_token_qdq_kernel(
    input_ptr,
    output_ptr,
    WIDTH: tl.constexpr,
    MAXQ: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Affine RTN QDQ with one independently quantized token per program."""
    row = tl.program_id(0)
    row_offset = row * WIDTH
    offsets = tl.arange(0, BLOCK_SIZE)

    xmin = 0.0
    xmax = 0.0
    for block_start in tl.range(0, WIDTH, BLOCK_SIZE):
        columns = block_start + offsets
        mask = columns < WIDTH
        values = tl.load(
            input_ptr + row_offset + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        xmin = tl.minimum(
            xmin,
            tl.min(tl.where(mask, values, 3.402823466e38), axis=0),
        )
        xmax = tl.maximum(
            xmax,
            tl.max(tl.where(mask, values, -3.402823466e38), axis=0),
        )

    scale = (xmax - xmin) / MAXQ
    valid_scale = scale > 0.0
    safe_scale = tl.where(valid_scale, scale, 1.0)
    zero = _round_to_nearest_even(-xmin / safe_scale)

    for block_start in tl.range(0, WIDTH, BLOCK_SIZE):
        columns = block_start + offsets
        mask = columns < WIDTH
        values = tl.load(
            input_ptr + row_offset + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        quantized = _round_to_nearest_even(values / safe_scale) + zero
        quantized = tl.maximum(0.0, tl.minimum(quantized, MAXQ))
        dequantized = scale * (quantized - zero)
        # A constant row has no quantization error and avoids 0/0.
        dequantized = tl.where(valid_scale, dequantized, values)
        tl.store(
            output_ptr + row_offset + columns,
            dequantized,
            mask=mask,
        )


@triton.jit
def _dynamic_rtn_channel_qdq_kernel(
    input_ptr,
    output_ptr,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    MAXQ: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Affine RTN QDQ with one independently quantized input channel."""
    column = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)

    xmin = 0.0
    xmax = 0.0
    for block_start in tl.range(0, ROWS, BLOCK_SIZE):
        rows = block_start + offsets
        mask = rows < ROWS
        values = tl.load(
            input_ptr + rows * WIDTH + column,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        xmin = tl.minimum(
            xmin,
            tl.min(tl.where(mask, values, 3.402823466e38), axis=0),
        )
        xmax = tl.maximum(
            xmax,
            tl.max(tl.where(mask, values, -3.402823466e38), axis=0),
        )

    scale = (xmax - xmin) / MAXQ
    valid_scale = scale > 0.0
    safe_scale = tl.where(valid_scale, scale, 1.0)
    zero = _round_to_nearest_even(-xmin / safe_scale)

    for block_start in tl.range(0, ROWS, BLOCK_SIZE):
        rows = block_start + offsets
        mask = rows < ROWS
        values = tl.load(
            input_ptr + rows * WIDTH + column,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        quantized = _round_to_nearest_even(values / safe_scale) + zero
        quantized = tl.maximum(0.0, tl.minimum(quantized, MAXQ))
        dequantized = scale * (quantized - zero)
        dequantized = tl.where(valid_scale, dequantized, values)
        tl.store(
            output_ptr + rows * WIDTH + column,
            dequantized,
            mask=mask,
        )


@triton.jit
def _mixnlq_nearest_index(
    abs_values,
    exp_scale,
    linear_scale,
    NUM_LEVELS: tl.constexpr,
    INDEX_BITS: tl.constexpr,
):
    """Find the nearest monotonic codebook level with exact binary search."""
    low = (abs_values * 0.0).to(tl.int32)
    high = low + (NUM_LEVELS - 1)
    for _ in tl.static_range(0, INDEX_BITS):
        middle = (low + high + 1) // 2
        boundary = middle - 1
        boundary_f32 = boundary.to(tl.float32)
        threshold = (
            1.5 * exp_scale * tl.exp2(boundary_f32)
            + linear_scale * (boundary_f32 + 0.5)
        )
        move_right = abs_values > threshold
        low = tl.where(move_right, middle, low)
        high = tl.where(move_right, high, middle - 1)
    return low


@triton.jit
def _mixnlq_bias3_nearest_index(
    abs_values,
    exp_scale,
    linear_scale,
    bias,
):
    """Exact nearest-level assignment for the four-level 3-bit codebook."""
    best_index = (abs_values * 0.0).to(tl.int32)
    best_distance = tl.abs(abs_values - (exp_scale + bias))
    for level in tl.static_range(1, 4):
        level_f32 = level + 0.0
        magnitude = (
            exp_scale * ((1 << level) + 0.0)
            + linear_scale * level_f32
            + bias
        )
        distance = tl.abs(abs_values - magnitude)
        better = distance < best_distance
        best_index = tl.where(better, level, best_index)
        best_distance = tl.where(better, distance, best_distance)
    return best_index


@triton.jit
def _dynamic_mixnlq_bias3_token_qdq_kernel(
    input_ptr,
    output_ptr,
    WIDTH: tl.constexpr,
    ITERATIONS: tl.constexpr,
    GRID_NUM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Dynamic per-token QDQ matching pseudo_mix_quant_bias(wbits=3)."""
    row = tl.program_id(0)
    row_offset = row * WIDTH
    offsets = tl.arange(0, BLOCK_SIZE)

    token_absmin = 3.402823466e38
    token_absmax = 0.0
    for block_start in tl.range(0, WIDTH, BLOCK_SIZE):
        columns = block_start + offsets
        mask = columns < WIDTH
        values = tl.load(
            input_ptr + row_offset + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        abs_values = tl.abs(values)
        token_absmin = tl.minimum(
            token_absmin,
            tl.min(
                tl.where(mask, abs_values, 3.402823466e38),
                axis=0,
            ),
        )
        token_absmax = tl.maximum(
            token_absmax,
            tl.max(tl.where(mask, abs_values, 0.0), axis=0),
        )

    value_range = token_absmax - token_absmin
    # For wbits=3: maxq=3, 2**maxq=8 and linear_factor=7/8.
    original_linear_scale = value_range / 3.0
    original_exp_scale = value_range / 8.0
    linear_factor = 0.875

    best_mse = token_absmax * 0.0 + 3.402823466e38
    best_exp_lambda = token_absmax * 0.0
    for grid_index in tl.static_range(0, GRID_NUM + 1):
        exp_lambda = grid_index * (1.0 / GRID_NUM)
        exp_scale_candidate = original_exp_scale * exp_lambda
        linear_scale_candidate = (
            original_linear_scale
            * (1.0 - exp_lambda)
            * linear_factor
        )
        bias_candidate = -exp_scale_candidate
        squared_error = 0.0
        for block_start in tl.range(0, WIDTH, BLOCK_SIZE):
            columns = block_start + offsets
            mask = columns < WIDTH
            values = tl.load(
                input_ptr + row_offset + columns,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            abs_values = tl.abs(values)
            indices = _mixnlq_bias3_nearest_index(
                abs_values,
                exp_scale_candidate,
                linear_scale_candidate,
                bias_candidate,
            )
            index_f32 = indices.to(tl.float32)
            magnitude = (
                exp_scale_candidate * tl.exp2(index_f32)
                + linear_scale_candidate * index_f32
                + bias_candidate
            )
            error = abs_values - magnitude
            squared_error += tl.sum(
                tl.where(mask, error * error, 0.0),
                axis=0,
            )
        better = squared_error < best_mse
        best_mse = tl.where(better, squared_error, best_mse)
        best_exp_lambda = tl.where(
            better,
            exp_lambda,
            best_exp_lambda,
        )

    exp_scale = original_exp_scale * best_exp_lambda
    linear_scale = (
        original_linear_scale
        * (1.0 - best_exp_lambda)
        * linear_factor
    )
    # The reference keeps this bias fixed during scale iterations.
    fixed_bias = -exp_scale
    assignment_exp_scale = exp_scale
    assignment_linear_scale = linear_scale

    for _ in tl.range(0, ITERATIONS):
        assignment_exp_scale = exp_scale
        assignment_linear_scale = linear_scale
        numerator_linear = 0.0
        denominator_linear = 0.0
        numerator_exp = 0.0
        denominator_exp = 0.0

        for block_start in tl.range(0, WIDTH, BLOCK_SIZE):
            columns = block_start + offsets
            mask = columns < WIDTH
            values = tl.load(
                input_ptr + row_offset + columns,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            abs_values = tl.abs(values)
            indices = _mixnlq_bias3_nearest_index(
                abs_values,
                assignment_exp_scale,
                assignment_linear_scale,
                fixed_bias,
            )
            index_f32 = indices.to(tl.float32)
            pow2_indices = tl.exp2(index_f32)
            common_residual = abs_values - fixed_bias

            numerator_linear += tl.sum(
                tl.where(
                    mask,
                    (
                        common_residual
                        - assignment_exp_scale * pow2_indices
                    )
                    * index_f32,
                    0.0,
                ),
                axis=0,
            )
            denominator_linear += tl.sum(
                tl.where(mask, index_f32 * index_f32, 0.0),
                axis=0,
            )
            numerator_exp += tl.sum(
                tl.where(
                    mask,
                    (
                        common_residual
                        - assignment_linear_scale * index_f32
                    )
                    * pow2_indices,
                    0.0,
                ),
                axis=0,
            )
            denominator_exp += tl.sum(
                tl.where(mask, pow2_indices * pow2_indices, 0.0),
                axis=0,
            )

        new_linear_scale = numerator_linear / tl.maximum(
            denominator_linear,
            1.0e-8,
        )
        new_exp_scale = numerator_exp / tl.maximum(
            denominator_exp,
            1.0e-8,
        )
        linear_scale = (
            0.5 * assignment_linear_scale + 0.5 * new_linear_scale
        )
        exp_scale = 0.5 * assignment_exp_scale + 0.5 * new_exp_scale

    # pseudo_mix_quant_bias uses the assignments from the final iteration,
    # then recomputes bias after the last scale update.
    residual_sum = 0.0
    for block_start in tl.range(0, WIDTH, BLOCK_SIZE):
        columns = block_start + offsets
        mask = columns < WIDTH
        values = tl.load(
            input_ptr + row_offset + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        abs_values = tl.abs(values)
        indices = _mixnlq_bias3_nearest_index(
            abs_values,
            assignment_exp_scale,
            assignment_linear_scale,
            fixed_bias,
        )
        index_f32 = indices.to(tl.float32)
        residual_sum += tl.sum(
            tl.where(
                mask,
                (
                    abs_values
                    - linear_scale * index_f32
                    - exp_scale * tl.exp2(index_f32)
                ),
                0.0,
            ),
            axis=0,
        )
    final_bias = residual_sum / WIDTH

    for block_start in tl.range(0, WIDTH, BLOCK_SIZE):
        columns = block_start + offsets
        mask = columns < WIDTH
        values = tl.load(
            input_ptr + row_offset + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        indices = _mixnlq_bias3_nearest_index(
            tl.abs(values),
            assignment_exp_scale,
            assignment_linear_scale,
            fixed_bias,
        )
        index_f32 = indices.to(tl.float32)
        magnitude = (
            exp_scale * tl.exp2(index_f32)
            + linear_scale * index_f32
            + final_bias
        )
        dequantized = tl.where(values < 0.0, -magnitude, magnitude)
        dequantized = tl.where(values == 0.0, 0.0, dequantized)
        tl.store(
            output_ptr + row_offset + columns,
            dequantized,
            mask=mask,
        )


@triton.jit
def _dynamic_mixnlq_token_qdq_kernel(
    input_ptr,
    output_ptr,
    WIDTH: tl.constexpr,
    NUM_LEVELS: tl.constexpr,
    INDEX_BITS: tl.constexpr,
    MAX_POW2: tl.constexpr,
    ITERATIONS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_offset = row * WIDTH
    offsets = tl.arange(0, BLOCK_SIZE)

    token_absmax = 0.0
    for block_start in tl.range(0, WIDTH, BLOCK_SIZE):
        columns = block_start + offsets
        mask = columns < WIDTH
        values = tl.load(
            input_ptr + row_offset + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        token_absmax = tl.maximum(
            token_absmax,
            tl.max(tl.where(mask, tl.abs(values), 0.0), axis=0),
        )

    # Fixed 50/50 initialization: at the highest level, exponential and
    # linear terms each contribute half of token_absmax.
    exp_scale = 0.5 * token_absmax / MAX_POW2
    linear_scale = 0.5 * token_absmax / (NUM_LEVELS - 1)

    for _ in tl.range(0, ITERATIONS):
        sum_pow2_sq = 0.0
        sum_index_sq = 0.0
        sum_cross = 0.0
        sum_value_pow2 = 0.0
        sum_value_index = 0.0

        for block_start in tl.range(0, WIDTH, BLOCK_SIZE):
            columns = block_start + offsets
            mask = columns < WIDTH
            values = tl.load(
                input_ptr + row_offset + columns,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            abs_values = tl.abs(values)
            indices = _mixnlq_nearest_index(
                abs_values,
                exp_scale,
                linear_scale,
                NUM_LEVELS,
                INDEX_BITS,
            )
            index_f32 = indices.to(tl.float32)
            pow2_indices = tl.exp2(index_f32)

            sum_pow2_sq += tl.sum(
                tl.where(mask, pow2_indices * pow2_indices, 0.0),
                axis=0,
            )
            sum_index_sq += tl.sum(
                tl.where(mask, index_f32 * index_f32, 0.0),
                axis=0,
            )
            sum_cross += tl.sum(
                tl.where(mask, pow2_indices * index_f32, 0.0),
                axis=0,
            )
            sum_value_pow2 += tl.sum(
                tl.where(mask, abs_values * pow2_indices, 0.0),
                axis=0,
            )
            sum_value_index += tl.sum(
                tl.where(mask, abs_values * index_f32, 0.0),
                axis=0,
            )

        determinant = (
            sum_pow2_sq * sum_index_sq - sum_cross * sum_cross
        )
        valid_solve = tl.abs(determinant) > 1.0e-12
        safe_determinant = tl.where(valid_solve, determinant, 1.0)
        solved_exp = (
            sum_value_pow2 * sum_index_sq
            - sum_value_index * sum_cross
        ) / safe_determinant
        solved_linear = (
            sum_value_index * sum_pow2_sq
            - sum_value_pow2 * sum_cross
        ) / safe_determinant

        # Project the two-parameter solve onto the non-negative quadrant. If
        # one coefficient is negative, re-solve the remaining one-dimensional
        # least-squares problem instead of merely clipping both coefficients.
        exp_only = sum_value_pow2 / tl.maximum(sum_pow2_sq, 1.0e-12)
        linear_only = sum_value_index / tl.maximum(sum_index_sq, 1.0e-12)
        exp_negative = solved_exp < 0.0
        linear_negative = solved_linear < 0.0
        solved_exp = tl.where(linear_negative, exp_only, solved_exp)
        solved_linear = tl.where(exp_negative, linear_only, solved_linear)
        solved_exp = tl.where(exp_negative, 0.0, solved_exp)
        solved_linear = tl.where(linear_negative, 0.0, solved_linear)

        # Non-negative coefficients keep the codebook monotonic, which makes
        # midpoint assignment possible without materializing all distances.
        exp_scale = tl.where(
            valid_solve,
            tl.maximum(solved_exp, 0.0),
            exp_scale,
        )
        linear_scale = tl.where(
            valid_solve,
            tl.maximum(solved_linear, 0.0),
            linear_scale,
        )

    for block_start in tl.range(0, WIDTH, BLOCK_SIZE):
        columns = block_start + offsets
        mask = columns < WIDTH
        values = tl.load(
            input_ptr + row_offset + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        abs_values = tl.abs(values)
        indices = _mixnlq_nearest_index(
            abs_values,
            exp_scale,
            linear_scale,
            NUM_LEVELS,
            INDEX_BITS,
        )
        index_f32 = indices.to(tl.float32)
        magnitude = (
            exp_scale * tl.exp2(index_f32)
            + linear_scale * index_f32
        )
        dequantized = tl.where(values < 0.0, -magnitude, magnitude)
        dequantized = tl.where(values == 0.0, 0.0, dequantized)
        tl.store(
            output_ptr + row_offset + columns,
            dequantized,
            mask=mask,
        )


def dynamic_mixnlq_token_qdq(
    inputs: torch.Tensor,
    bits: int = 4,
    iterations: int = 2,
) -> torch.Tensor:
    """Quantize-dequantize contiguous CUDA token rows in one kernel launch.

    Three-bit activation QDQ uses the bias-aware four-level codebook from
    ``pseudo_mix_quant_bias``. Other bit widths retain the two-scale codebook.
    """
    if not inputs.is_cuda:
        raise ValueError("dynamic MixNLQ Triton QDQ requires a CUDA tensor")
    if inputs.ndim != 2:
        raise ValueError("inputs must have shape [tokens, hidden_size]")
    if not inputs.is_contiguous():
        raise ValueError("inputs must be contiguous")
    if bits < 2 or bits > 6:
        raise ValueError("the fused dynamic kernel supports 2 to 6 bits")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if inputs.shape[1] <= 0:
        raise ValueError("hidden_size must be positive")

    output = torch.empty_like(inputs)
    if bits == 3:
        with torch.cuda.device(inputs.device):
            _dynamic_mixnlq_bias3_token_qdq_kernel[(inputs.shape[0],)](
                inputs,
                output,
                WIDTH=inputs.shape[1],
                ITERATIONS=iterations,
                GRID_NUM=10,
                BLOCK_SIZE=256,
                num_warps=4,
            )
        return output

    num_levels = 2 ** (bits - 1)
    max_pow2 = 2 ** (num_levels - 1)
    with torch.cuda.device(inputs.device):
        _dynamic_mixnlq_token_qdq_kernel[(inputs.shape[0],)](
            inputs,
            output,
            WIDTH=inputs.shape[1],
            NUM_LEVELS=num_levels,
            INDEX_BITS=bits - 1,
            MAX_POW2=max_pow2,
            ITERATIONS=iterations,
            BLOCK_SIZE=256,
            num_warps=4,
        )
    return output


def dynamic_rtn_qdq(
    inputs: torch.Tensor,
    bits: int = 4,
    granularity: str = "token",
) -> torch.Tensor:
    """Dynamically affine-quantize and dequantize CUDA activations.

    ``token`` computes one scale/zero-point over the hidden dimension of each
    token. ``channel`` computes one scale/zero-point over all token rows for
    each input channel. The returned tensor retains the input dtype and shape.
    """
    if not inputs.is_cuda:
        raise ValueError("dynamic RTN Triton QDQ requires a CUDA tensor")
    if inputs.ndim != 2:
        raise ValueError("inputs must have shape [tokens, hidden_size]")
    if not inputs.is_contiguous():
        raise ValueError("inputs must be contiguous")
    if bits < 2 or bits > 8:
        raise ValueError("the fused dynamic RTN kernel supports 2 to 8 bits")
    if granularity not in {"token", "channel"}:
        raise ValueError("granularity must be token or channel")
    if inputs.shape[0] <= 0 or inputs.shape[1] <= 0:
        raise ValueError("activation dimensions must be positive")

    output = torch.empty_like(inputs)
    maxq = 2 ** bits - 1
    with torch.cuda.device(inputs.device):
        if granularity == "token":
            _dynamic_rtn_token_qdq_kernel[(inputs.shape[0],)](
                inputs,
                output,
                WIDTH=inputs.shape[1],
                MAXQ=maxq,
                BLOCK_SIZE=256,
                num_warps=4,
            )
        else:
            _dynamic_rtn_channel_qdq_kernel[(inputs.shape[1],)](
                inputs,
                output,
                ROWS=inputs.shape[0],
                WIDTH=inputs.shape[1],
                MAXQ=maxq,
                BLOCK_SIZE=256,
                num_warps=4,
            )
    return output
