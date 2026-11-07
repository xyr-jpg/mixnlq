import torch
from utils.hadamard_utils import hadamard_matrix
from utils.hqq import quantize, dequantize
from utils.quant_utils import pseudo_RTN_quant
import math
from scipy.stats import norm

def RTN_had_quant(wbits, W,  group_size=0, quantile=1):
    """Round-to-Nearest quantization using hadamard matrix"""
    had = hadamard_matrix(W.shape[1], device=W.device).to(W.dtype)
    W = W @ had

    shape = W.shape
    if group_size>0:
        W = W.reshape(-1, group_size)
    tmp = torch.zeros(W.shape[0], device=W.device, dtype=W.dtype)
    xmin = torch.minimum(W.min(1)[0], tmp)
    xmax = torch.maximum(W.max(1)[0], tmp)
        
    if quantile<1:
       xmax *= quantile
       xmin *= quantile
    maxq = 2 ** wbits - 1
    scale = (xmax - xmin) / maxq
    zeros = torch.round(-xmin / scale)
    q = torch.clamp(torch.round(W / scale.unsqueeze(1)) + zeros.unsqueeze(1), 0, maxq)
    deq_W = scale.unsqueeze(1) * (q - zeros.unsqueeze(1))
    deq_W = deq_W.reshape(shape)
    return deq_W @ had.T

def NF4_quant(wbits, W, group_size=0):
    """NF4 quantization"""
    # assert wbits==4, "only support wbits==4 yet"
    shape = W.shape
    if group_size>0:
        W = W.reshape(-1, group_size)
    
    offset = 0.9677083
    step_pos = torch.linspace(offset, 0.5, 2**wbits//2+1)[:-1] #概率区间
    step_neg = torch.linspace(offset, 0.5, 2**wbits//2)[:-1]
    # v1 = gennorm.ppf(step_pos, beta=shape).tolist() 
    v1 = norm.ppf(step_pos).tolist() 
    v2 = [0]
    # v3 = (-gennorm.ppf(step_neg, beta=shape)).tolist() 
    v3 = (-norm.ppf(step_neg)).tolist() 
    v = v1 + v2 + v3
    level_param = torch.tensor(v, dtype=W.dtype, device=W.device)
    level_param = level_param.sort().values
    level_param /= (level_param.max()) 
    level_param = level_param.repeat(W.shape[0],1)
    scale = torch.abs(W).max(dim=1, keepdim=True).values
    scale = torch.clamp(scale, min=1e-8)
    
    normalized = W / scale

    # 量化：找到最近的等级
    expanded_levels = level_param.unsqueeze(1)
    expanded_input = normalized.unsqueeze(-1)

    # 计算与所有量化等级的距离
    distances = torch.abs(expanded_input - expanded_levels)
    indices = torch.argmin(distances, dim=-1)
    indices = indices.to(torch.uint8)
    # 确保索引在 0-15 范围内
    indices = torch.clamp(indices, 0, 2**wbits-1)
    # 反量化用于前向传播
    # dequantized = torch.zeros_like(W)
    level_param_expanded = level_param.unsqueeze(1).expand(-1, indices.shape[1], -1)
    
    # 扩展 indices 以匹配 gather 的维度要求
    indices_expanded = indices.unsqueeze(-1).to(torch.int64)

    # 使用 gather 进行批量查找
    dequantized = torch.gather(level_param_expanded, 2, indices_expanded).squeeze(-1)
    dequantized = dequantized * scale
    
    return dequantized.reshape(shape)

def HQQ_quant(wbits, W, group_size=0):
    """HQQ quantization"""
    # shape = W.shape
    if group_size <= 0:
        group_size = W.shape[1]
    W_q, meta = quantize(W.float(), wbits, True, group_size, device=W.device)
    deq_W = dequantize(W_q, meta)
    return deq_W.to(W.dtype)

def sinq_quant(wbits, W, group_size=0, niter=50):
    shape = W.shape
    if group_size > 0:
        W = W.reshape(-1, group_size)
    else:
        group_size = W.shape[1]
    std_dim0 = W.std(dim=0)  # std along columns, shape (n,)
    std_dim1 = W.std(dim=1)
    sigma_min = min(std_dim0.min(), std_dim1.min())
    
    # Avoid division by zero
    sigma_min = 1.0 #max(sigma_min, 1e-8)

    W_hat = W.clone()

    sigma_1_final = torch.ones(W.size(0), device=W.device)  # row factors
    sigma_0_final = torch.ones(W.size(1), device=W.device)  # column factors
    
    # print(f"std_dim0: {std_dim0.max(), std_dim0.min()}")
    # print(f"std_dim1: {std_dim1.max(), std_dim1.min()}")

    for i in range(niter):
        # Current standard deviations
        current_std_0 = W_hat.std(dim=0)  # columns
        current_std_1 = W_hat.std(dim=1)  # rows
        
        # Compute scaling factors
        sigma_tilde_0 = current_std_0 / sigma_min
        sigma_tilde_1 = current_std_1 / sigma_min
        
        # Avoid division by zero
        sigma_tilde_0 = torch.clamp(sigma_tilde_0, min=1e-8)
        sigma_tilde_1 = torch.clamp(sigma_tilde_1, min=1e-8)
        
        # Scale columns
        W_hat = W_hat / sigma_tilde_0
        # Scale rows  
        W_hat = W_hat / sigma_tilde_1.unsqueeze(1)  # add dimension for broadcasting
        
        # Accumulate scaling factors
        sigma_0_final = sigma_0_final * sigma_tilde_0
        sigma_1_final = sigma_1_final * sigma_tilde_1
    # print(W_hat)
    # print(W_hat.std(dim=0).min(),W_hat.std(dim=0).max())
    # print(W_hat.std(dim=1).min(),W_hat.std(dim=1).max())
    q, scale, zeros, dequant_What = pseudo_RTN_quant(wbits=wbits, W=W_hat)

    s_final = sigma_1_final * scale
    dequantized = (q.float() - zeros.unsqueeze(1)) * s_final.unsqueeze(1) * sigma_0_final.unsqueeze(0)
    return dequantized.reshape(shape)

def AWQ_quant(wbits, W, X, group_size=0):
    org_out = W @ X

    best_error = float("inf")
    best_ratio = -1
    best_scales = None
    best_deq_W = None
    n_grid = 20
    history = []
    x_max = X.abs().view(X.shape[0],-1).mean(-1) #torch.max(torch.mean(torch.abs(X), dim=-1)) * torch.ones(W.shape[0], device=W.device, dtype=W.dtype)
    # print("x_max", x_max)
    for ratio in range(n_grid):
        ratio = ratio * 1 / n_grid
        scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
        scales = scales / (scales.max() * scales.min()).sqrt()
        q, scale, zeros, deq_W = pseudo_RTN_quant(wbits=wbits, W=W * (scales.unsqueeze(0)), group_size=group_size)
        deq_W = deq_W / (scales.unsqueeze(0))
        out = deq_W @ X
        
        loss = (org_out - out).float().pow(2).mean().item()
        
        history.append(loss)
        is_best = loss < best_error
        if is_best:
            best_error = loss
            best_ratio = ratio
            best_scales = scales
            best_deq_W = deq_W
    # print("best_ratio",best_ratio)
    
    return best_deq_W


    
#Power of two quantization
def PoT_quant(wbits, W, group_size=0, quantile=1):
    maxq = 2 ** wbits - 1
    shape = W.shape
    if group_size > 0:
        W = W.reshape(-1, group_size)
    if quantile<1:
        xmax = torch.quantile(W.float(), quantile, dim=1)
        xmin = torch.quantile(W.float(), (1-quantile), dim=1)
    else:
        tmp = torch.zeros(W.shape[0], device=W.device, dtype=W.dtype)
        xmin = torch.minimum(W.min(1)[0], tmp)
        xmax = torch.maximum(W.max(1)[0], tmp)
    xabs = torch.max(torch.abs(xmin), torch.abs(xmax))
    
    sequence = []
    for i in range(2 ** (wbits-1)-1):
        sequence.append(2 ** (-2 * (wbits-1) + 2 + i))
        sequence.append(-2 ** (-2 * (wbits-1) + 2 + i))
    sequence.insert((2 ** (wbits-1) - 1) * 2, 0)
    sequence = torch.tensor(sequence, dtype=W.dtype, device=W.device)
    sequence, _ = torch.sort(sequence)
    
    W_normalized = W / xabs.unsqueeze(-1)
    sequence_expanded = sequence.unsqueeze(0).unsqueeze(0)  # (1, 1, k)
    
    W_quant = W.clone()
    slice_num = 1 #16
    slice_delta = W.shape[0]//slice_num
    for i in range(slice_num):
        W_expanded = W_normalized[slice_delta*i : slice_delta*(i+1)].unsqueeze(-1)
        
        # 计算每个元素与codebook中所有值的距离
        distances = torch.abs(W_expanded - sequence_expanded)  # (m, n, k)
    
        # 找到最小距离的索引
        indices = torch.argmin(distances, dim=-1)  # (m, n)
    
        # 用codebook中的值替换
        W_quant[slice_delta*i : slice_delta*(i+1)] = sequence[indices]
    
    W_quant = W_quant * xabs.unsqueeze(-1)

    return W_quant.reshape(shape)
