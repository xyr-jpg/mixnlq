import torch
import torch.nn as nn
import torch
import numpy as np
from torch import float32, float16, Tensor
from functools import partial
from typing import Union

# Shrinking operator
def shrink_lp_op(x: Tensor, beta: float, lp_norm: float) -> Tensor:
    if lp_norm == 1: 
        #torch.sign(x) * torch.nn.functional.relu(torch.abs(x) - 1.0 / beta)
        out = torch.abs(x)
        out.sub_(1.0 / beta).clamp_min_(0.0)
        out.mul_(torch.sign(x))
        return out
    else:
        #torch.sign(x) * torch.nn.functional.relu(torch.abs(x) - (1.0 / beta) * torch.pow(torch.abs(x), lp_norm - 1))
        out = torch.abs(x)
        out.sub_((1.0 / beta) * out.pow(lp_norm - 1)).clamp_min_(0.0)
        out.mul_(torch.sign(x))
        return out

def optimize_weights_proximal_legacy_step(W_f, scale, zero, min_max, beta, lp_norm, axis):
    W_q = torch.round(W_f * scale + zero).clamp_(min_max[0], min_max[1])
    W_r = (W_q - zero) / scale
    W_e = shrink_lp_op(W_f - W_r, beta, lp_norm)
    zero = torch.mean(W_q - (W_f - W_e) * scale, axis=axis, keepdim=True)
    return W_r, W_q, zero, scale

@torch.inference_mode()
def optimize_weights_proximal_legacy(
    tensor: Tensor,
    scale: Tensor,
    zero: Tensor,
    min_max: list,
    axis: int = 0,
    device: Union[str, None] = None,
    opt_params: dict = {"lp_norm": 0.7, "beta": 1e1, "kappa": 1.01, "iters": 20},
    verbose: bool = False,
) -> tuple:
    lp_norm, beta, kappa, iters = (
        opt_params["lp_norm"],
        opt_params["beta"],
        opt_params["kappa"],
        opt_params["iters"],
    )

    if device is None:
        device = tensor.device
    else:
        device = torch.device(device)

    dtype = float16 if (device.type == "cuda") else float32
    W_f   = tensor.to(dtype=dtype, device=device)
    scale = scale.to(dtype=dtype, device=device)
    zero  = zero.to(dtype=dtype, device=device)

    best_error = torch.tensor(torch.inf, dtype=float32, device=device)
    for i in range(iters):
        W_r, W_q, zero, scale = optimize_weights_proximal_legacy_step(W_f, scale, zero, min_max, beta, lp_norm, axis)
        current_error = torch.abs(W_f - W_r).mean().float()
        # print(f"{i}, {current_error}")
        beta = beta * kappa
        if verbose: 
            print(i, np.round(current_error, 6))
        
        if current_error < best_error:
            best_error = current_error
        else:
            break

    scale = scale.to(tensor.device)
    zero = zero.to(tensor.device)
    del W_f, W_q, W_r
    torch.cuda.empty_cache()

    W_q = torch.round(tensor * scale + zero).clamp_(min_max[0], min_max[1])
    return W_q, scale, zero

def optimize_weights_proximal_v2(
    tensor: Tensor,
    scale: Tensor,
    zero: Tensor,
    min_max: list,
    axis: int = 0,
    device: Union[str, None] = None,
    dtype: Union[torch.dtype, None] = None,
    opt_params: dict = {
        "lp_norm": 0.7,
        "beta": 1e1,
        "kappa": 1.01,
        "iters": 20,
        "tol": 0.0,
        "early_stop": True,
        "scale_gridsearch": False,
    },
    verbose: bool = False,
) -> tuple:
    # Params
    lp_norm = max(opt_params["lp_norm"], 0.1)
    beta = opt_params["beta"]
    kappa = opt_params["kappa"]
    iters = opt_params["iters"]
    early_stop = opt_params["early_stop"]
    tol = opt_params["tol"]

    # Check
    assert lp_norm <= 1.0, "lp_norm should be <=1"
    assert beta > 0.0, "beta should be > 0"
    assert kappa > 1.0, "kappa should be > 1"
    assert iters > 1, "iters should be > 1"

    # Cast/device
    if device is None:
        device = tensor.device
    else:
        device = torch.device(device)

    if dtype is None:
        dtype = float16 if (device.type == "cuda") else float32

    W_f = tensor.to(device=device, dtype=dtype)
    scale = scale.to(device=device, dtype=dtype)
    zero = zero.to(device=device, dtype=dtype)

    # Update scale: works slightly better. Tested on Llama2 only
    # if opt_params["scale_gridsearch"]:
    #     scale = update_scale_grid_search(W_f, scale, zero, axis, min_max)

    # Optimize for zero-point
    best_error = torch.tensor(1e4, dtype=float32, device=device)
    scale_prev, zero_prev = scale.clone(), zero.clone()
    for i in range(iters):
        W_q = torch.round(W_f * scale + zero).clamp(min_max[0], min_max[1])
        W_r = (W_q - zero) / scale

        # current_error = float(torch.pow(torch.abs(W_f - W_r), max(0.80, lp_norm)).mean())
        current_error = torch.abs(W_f - W_r).mean().float()

        if verbose:
            print(i, np.round(current_error, 6))

        if early_stop:
            if best_error - current_error > tol:
                best_error = current_error
                scale_prev, zero_prev = scale.clone(), zero.clone()
            else:
                scale, zero = scale_prev.clone(), zero_prev.clone()
                break

        W_e = shrink_lp_op(W_f - W_r, beta, lp_norm)
        zero = torch.mean(W_q - (W_f - W_e) * scale, axis=axis, keepdim=True)
        beta *= kappa

    # Clean-up
    scale = scale.to(tensor.device)
    zero = zero.to(tensor.device)
    del W_f, W_q, W_r, W_e, scale_prev, zero_prev
    torch.cuda.empty_cache()

    W_q = torch.round(tensor * scale + zero).clamp(min_max[0], min_max[1])

    return W_q, scale, zero

def quantize(
    # cls,
    tensor: Tensor,
    nbits: float = 4,
    channel_wise: bool = True,
    group_size: int = 64,
    optimize: bool = True,
    round_zero: bool = False,
    axis: int = 0,
    # bitpack: bool = True,
    # compute_dtype: Union[torch.dtype, None] = None,
    # view_as_float: bool = False,
    device: str = "cuda",
) -> tuple:
    W = tensor.float()
    shape = W.shape
    # Reshape for grouping
    
    if (group_size is not None) and channel_wise:
        W = (
            W.reshape([-1, group_size])
            if (axis == 1)
            else W.reshape([group_size, -1])
        )
    # Get min/max values
    if not channel_wise:
        _min, _max = W.min(), W.max()
        optimize = False
    else:
        _min = W.min(axis=axis, keepdim=True)[0]
        _max = W.max(axis=axis, keepdim=True)[0]

    max_v = round(2**nbits - 1)
    min_v = 0
    min_max = [min_v, max_v]
    # Note: here we work with the inverse of the scale to avoid division and quantize instead via W*scale + zero, the scale is inverted later on.
    denom = (_max - _min)
    scale = (max_v / denom)  
    scale = torch.where(denom.abs() <= 1e-4, torch.full_like(scale, 1.0), scale) #Avoid small denom values
    scale = scale.clamp(max=2e4) # clamp to avoid half-precision problems
    zero = -_min * scale
    # Round zero as in: https://github.com/casper-hansen/AutoAWQ/blob/main/awq/quantize/quantizer.py#L42C9-L42C14
    if round_zero:
        zero = torch.round(zero)
    # Fine-tune weights
    if optimize:
        W_q, scale, zero = optimize_weights_proximal_legacy(
            tensor=W,
            scale=scale,
            zero=zero,
            min_max=min_max,
            axis=axis,
            device=device,
        )
    else:
        W_q = (W * scale + zero).round_().clamp_(min_max[0], min_max[1])
    # cleanup
    del W, _min, _max
    torch.cuda.empty_cache()
    # Store meta-data (we invert the scale for dequantization)
    scale = 1.0 / scale
    meta = {
        "nbits": nbits,
        "group_size": group_size,
        "shape": shape,
        "scale": scale,
        "zero": zero,
        "axis": axis,
        # "packing": Quantizer.bit_to_packing[nbits],
    }
    # meta["unpack_view_dtype"] = Quantizer.unpack_view_dtype[meta["packing"]]
    # Pack bits
    # meta["view_as_float"] = view_as_float
    # if bitpack:
    #     W_q = Quantizer.pack[meta["packing"]](W_q)
    #     if view_as_float:
    #         W_q = W_q.view(
    #             torch.float32 if compute_dtype is None else compute_dtype
    #         )  # store quantized weights as compute_dtype
    # else:
    #     W_q = W_q.to(tensor.dtype)
    #     meta["packing"] = None
    torch.cuda.empty_cache()
    return W_q, meta

# Main dequantization: bit_unpacking > (W_q - z)*s > reshape
# @classmethod
def dequantize(W_q: Tensor, meta: dict) -> Tensor:
    # compute_dtype = meta["compute_dtype"] if ("compute_dtype" in meta) else float16
    # if meta["packing"]:
    #     if meta["view_as_float"]:
    #         W_q = W_q.view(meta["unpack_view_dtype"])
    #     # W_r = Quantizer.unpack[meta["packing"]](W_q, dtype=compute_dtype)
    #     if meta["nbits"] == 3:
    #         W_r = W_r[
    #             : meta["group_size"]
    #             if meta["axis"] == 0
    #             else meta["shape"][0] * meta["shape"][1] // meta["group_size"]
    #         ]
    # else:
    #     W_r = W_q.to(compute_dtype)
    W_r = ((W_q - meta["zero"]) * meta["scale"]).reshape(meta["shape"])
    return W_r