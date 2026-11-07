import torch
import math

_pos_vals = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]) #, device='cuda')
E2M1_DICT = torch.cat([-_pos_vals.flip(0)[:-1], _pos_vals])

def pad_and_block(W, block_size):
    """将 m x n 矩阵展平并补齐为 block_size 的整数倍，然后重组为二维块矩阵"""
    w_flat = W.flatten()
    pad_len = (block_size - (w_flat.numel() % block_size)) % block_size
    if pad_len > 0:
        w_flat = torch.nn.functional.pad(w_flat, (0, pad_len))
    return w_flat.view(-1, block_size), pad_len

def unblock_and_unpad(blocks, original_shape, pad_len):
    """将处理后的块矩阵还原回原始的 m x n 形状"""
    w_flat = blocks.flatten()
    if pad_len > 0:
        w_flat = w_flat[:-pad_len]
    return w_flat.view(original_shape)


def quantize_matrix_nvfp4(W, block_size=16):
    blocks, pad_len = pad_and_block(W, block_size)
    
    # E4M3 的最大可表示物理值为 448.0
    E4M3_MAX = 448.0
    
    # 1. Tensor-wise 全局缩放 (FP32)
    global_amax = W.abs().max().clamp(min=1e-8)
    global_scale = global_amax / (6.0 * E4M3_MAX) 
    
    # 剔除全局缩放后的基础块
    blocks_base = blocks / global_scale
    
    # 2. Block-wise 局部缩放 (模拟 FP8 E4M3)
    block_amax = blocks_base.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)
    
    block_scale_fp8 = (block_amax / 6.0).to(torch.float8_e4m3fn)
    
    # 3. 映射与吸附
    scale_f32 = block_scale_fp8.to(torch.float32).clamp(min=1e-12)
    w_scaled = blocks_base / scale_f32
    
    # 广播计算最近邻 (推理 PTQ 适用)
    idx = torch.argmin(torch.abs(w_scaled.unsqueeze(-1) - E2M1_DICT.to(w_scaled.device)), dim=-1)
    w_fp4_mantissas = (E2M1_DICT.to(w_scaled.device))[idx]
    
    # 4. 反量化还原 (硬件行为：FP4_Value * FP8_Scale * FP32_Scale)
    w_sim_blocks = w_fp4_mantissas * scale_f32 * global_scale
    w_sim = unblock_and_unpad(w_sim_blocks, W.shape, pad_len)
    
    return w_sim, global_scale, block_scale_fp8

def quantize_matrix_mxfp4(W, block_size=32):
    # 假设 pad_and_block 逻辑不变
    blocks, pad_len = pad_and_block(W, block_size)
    device = blocks.device
    
    # 1. 计算 Block-wise scale (E8M0)
    block_amax = blocks.abs().max(dim=1, keepdim=True)[0]
    
    # 避免零除与 log2 报错
    epsilon = 1e-12
    e_val = torch.ceil(torch.log2((block_amax + epsilon) / 6.0))
    
    e_val = torch.clamp(e_val, min=-127.0, max=127.0)
    scale_e8m0 = 2.0 ** e_val
    scale_e8m0 = torch.where(block_amax == 0, torch.ones_like(scale_e8m0), scale_e8m0)
    
    # 2. 映射与吸附
    w_scaled = blocks / scale_e8m0
    e2m1_dict = E2M1_DICT.to(w_scaled.device)
    
    # 使用绝对误差寻找最近邻 (注意: 这里的 unsqueeze 依然有大矩阵显存压力)
    idx = torch.argmin(torch.abs(w_scaled.unsqueeze(-1) - e2m1_dict), dim=-1)
    w_fp4_mantissas = e2m1_dict[idx]
    
    # 3. 反量化还原
    w_sim_blocks = w_fp4_mantissas * scale_e8m0
    w_sim = unblock_and_unpad(w_sim_blocks, W.shape, pad_len)
    
    return w_sim, scale_e8m0