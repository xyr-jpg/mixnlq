import torch
import numpy as np
import math

def pseudo_RTN_quant(wbits, W, group_size=0, quantile=1):
    """Round-to-Nearest group quantization"""
    # group_size = W.shape[1] // num_groups
    maxq = 2 ** wbits - 1

    # Reshape W for group-wise processing
    W_reshape = W.clone()
    if group_size>0:
        W_reshape = W_reshape.reshape(-1, group_size)

    tmp = torch.zeros(W_reshape.shape[0], device=W.device, dtype=W.dtype)
    xmin = torch.min(W_reshape.min(dim=1)[0], tmp)
    xmax = torch.max(W_reshape.max(dim=1)[0], tmp)
    if quantile<1:
       xmax *= quantile
       xmin *= quantile
    
    # Calculate scale and zero offset for each group
    scale = (xmax - xmin) / maxq
    zeros = torch.round(-xmin / scale)
    q = torch.clamp(torch.round(W_reshape / scale.unsqueeze(1)) + zeros.unsqueeze(1), 0, maxq)
    
    # Reconstruct the quantized weights
    quant_W_reshape = scale.unsqueeze(1) * (q - zeros.unsqueeze(1))
    quant_W = quant_W_reshape.reshape(W.shape)
    return q, scale, zeros, quant_W

def pseudo_mix_quant_bias(wbits, W, w_group_size=0, iterations=100, grid_num=10, X=None, alpha=1.5):    
    """Optimized MixNLQ with bias"""
    wshape = W.shape

    if w_group_size > 0:
        W = W.reshape(-1, w_group_size)
    abs_W = torch.abs(W)
    sign_W = torch.sign(W)

    if X is None or alpha==0:
        lam_scale = 1.0
    else:
        lam = X.unsqueeze(0).expand(wshape)
        lam = lam** alpha
        lam = lam / lam.max(dim=1, keepdim=True)[0].clamp_min(1e-8)
        lam_scale = lam.reshape(W.shape)  

    maxq = 2 ** (wbits - 1) - 1
    
    xmin = abs_W.min(dim=1, keepdim=True)[0]
    xmax = abs_W.max(dim=1, keepdim=True)[0]
    
    org_linear_scale = (xmax - xmin) / maxq
    org_exp_scale = (xmax - xmin) * (2 ** (-maxq))

    best_mse = None
    best_exp_lamb = torch.zeros_like(org_exp_scale)

    assert grid_num > 0, "grid_num must > 0 !"

    levels = torch.arange(0, maxq + 1, dtype=W.dtype, device=W.device)  # shape: (maxq+1,)
    pow2_levels = 2.0 ** levels                                         # shape: (maxq+1,)
    
    abs_W_exp = abs_W.unsqueeze(-1) 
    
    # --- Grid Search 阶段 ---
    linear_factor = (2**maxq - 1) / (2**maxq)
    
    for i in range(grid_num + 1):
        exp_lamb = (i / grid_num)
        linear_lamb = (1 - exp_lamb) * linear_factor
        
        exp_scale = org_exp_scale * exp_lamb        # shape: (B, 1)
        linear_scale = org_linear_scale * linear_lamb # shape: (B, 1)
        bias = -exp_scale                           # shape: (B, 1)

        quant_level_mix = exp_scale * pow2_levels + linear_scale * levels + bias
        
        distances = torch.abs(abs_W_exp - quant_level_mix.unsqueeze(1)) # shape: (B, N, maxq+1)
        indices = torch.argmin(distances, dim=-1)                       # shape: (B, N)

        pow2_indices = 2.0 ** indices
        quant_W1 = exp_scale * pow2_indices
        quant_W2 = linear_scale * indices
        quant_W_abs = quant_W1 + quant_W2 + bias
        
        mse = torch.mean((abs_W - quant_W_abs)**2, dim=-1, keepdim=True)
        
        if best_mse is None:
            best_mse = mse
            best_exp_lamb = torch.full_like(best_exp_lamb, exp_lamb)
        else:
            better_mask = mse < best_mse
            best_exp_lamb = torch.where(better_mask, exp_lamb, best_exp_lamb)
            best_mse = torch.where(better_mask, mse, best_mse)

    exp_scale = org_exp_scale * best_exp_lamb
    linear_scale = org_linear_scale * (1 - best_exp_lamb) * linear_factor
    bias = -exp_scale

    for it in range(iterations):
        quant_level_mix = exp_scale * pow2_levels + linear_scale * levels + bias
        
        distances = torch.abs(abs_W_exp - quant_level_mix.unsqueeze(1))
        indices = torch.argmin(distances, dim=-1)
        
        pow2_indices = 2.0 ** indices

        common_residual = abs_W - bias

        denom_linear = torch.sum(indices**2 * lam_scale, dim=-1, keepdim=True).clamp_min(1e-8)
        new_linear_scale = torch.sum((common_residual - exp_scale * pow2_indices) * indices * lam_scale, dim=-1, keepdim=True) / denom_linear
        
        denom_exp = torch.sum(pow2_indices**2 * lam_scale, dim=-1, keepdim=True).clamp_min(1e-8)
        new_exp_scale = torch.sum((common_residual - linear_scale * indices) * pow2_indices * lam_scale, dim=-1, keepdim=True) / denom_exp
        
        linear_scale = 0.5 * linear_scale + 0.5 * new_linear_scale
        exp_scale = 0.5 * exp_scale + 0.5 * new_exp_scale

    exp_scale = exp_scale.to(W.dtype)
    linear_scale = linear_scale.to(W.dtype)
    indices = indices.to(W.dtype)
    sign_W = sign_W.to(W.dtype)

    pow2_indices = 2.0 ** indices
    bias = torch.mean((abs_W - linear_scale * indices - exp_scale * pow2_indices), dim=-1, keepdim=True)
    
    quant_W_abs = exp_scale * pow2_indices + linear_scale * indices + bias
    quant_W = sign_W * quant_W_abs
    
    quant_W = quant_W.reshape(wshape)    
    sign_W = torch.sign(quant_W)
    
    return quant_W, [sign_W, indices, exp_scale.squeeze(-1), linear_scale.squeeze(-1), bias.squeeze(-1)]

def pseudo_mix_quant(wbits, W, w_group_size=0, iterations=100, grid_num=10, X=None, alpha=1.0):  
    """Optimized MixNLQ (Bug Fixed)"""
    wshape = W.shape
    
    if w_group_size > 0:
        W = W.reshape(-1, w_group_size)
    abs_W = torch.abs(W)
    sign_W = torch.sign(W)

    if X is None or alpha==0:
        lam_scale = 1.0
    else:
        lam = X.unsqueeze(0).expand(wshape)
        lam = lam ** alpha
        lam = lam / lam.max(dim=1, keepdim=True)[0].clamp_min(1e-8)
        lam_scale = lam.reshape(W.shape)

    maxq = 2 ** (wbits - 1) - 1
    
    xmin = abs_W.min(dim=1, keepdim=True)[0]
    xmax = abs_W.max(dim=1, keepdim=True)[0]
    
    org_linear_scale = (xmax - xmin) / maxq
    org_exp_scale = (xmax - xmin) * (2 ** (-maxq))
   
    best_mse = None
    best_exp_lamb = torch.zeros_like(org_exp_scale)
    assert grid_num > 0, "grid_num must > 0 !"
    
    levels = torch.arange(0, maxq + 1, dtype=W.dtype, device=W.device)  
    pow2_levels = 2.0 ** levels                                         
    
    abs_W_exp = abs_W.unsqueeze(-1) 
    linear_factor = (2**maxq - 1) / (2**maxq)
    
    # --- Grid Search 阶段 ---
    for i in range(grid_num + 1):
        exp_lamb = i / grid_num
        linear_lamb = (1 - exp_lamb) * linear_factor
        
        exp_scale = org_exp_scale * exp_lamb        
        linear_scale = org_linear_scale * linear_lamb 
        
        quant_level_mix = exp_scale * pow2_levels + linear_scale * levels
        
        distances = torch.abs(abs_W_exp - quant_level_mix.unsqueeze(1)) 
        indices = torch.argmin(distances, dim=-1)
        
        pow2_indices = 2.0 ** indices
        quant_W_abs = exp_scale * pow2_indices + linear_scale * indices
        
        mse = torch.mean((abs_W - quant_W_abs)**2, dim=-1, keepdim=True)
        
        if best_mse is None:
            best_mse = mse
            best_exp_lamb = torch.full_like(best_exp_lamb, exp_lamb)
        else:
            better_mask = mse < best_mse
            best_exp_lamb = torch.where(better_mask, exp_lamb, best_exp_lamb)
            best_mse = torch.where(better_mask, mse, best_mse)

    exp_scale = org_exp_scale * best_exp_lamb
    linear_scale = org_linear_scale * (1 - best_exp_lamb) * linear_factor
    
    for it in range(iterations):
        quant_level_mix = exp_scale * pow2_levels + linear_scale * levels

        distances = torch.abs(abs_W_exp - quant_level_mix.unsqueeze(1)) 
        indices = torch.argmin(distances, dim=-1)

        pow_indices = 2.0 ** indices
        pow_indices_sq = pow_indices ** 2
        indices_sq = indices ** 2

        numerator_linear = torch.sum((abs_W - exp_scale * pow_indices) * indices * lam_scale, dim=-1, keepdim=True)
        denominator_linear = torch.sum(indices_sq * lam_scale, dim=-1, keepdim=True)
        new_linear_scale = numerator_linear / (denominator_linear + 1e-8)  

        numerator_exp = torch.sum((abs_W - linear_scale * indices) * pow_indices * lam_scale, dim=-1, keepdim=True)
        denominator_exp = torch.sum(pow_indices_sq * lam_scale, dim=-1, keepdim=True)
        new_exp_scale = numerator_exp / (denominator_exp + 1e-8)
        
        linear_scale = 0.5 * linear_scale + 0.5 * new_linear_scale
        exp_scale = 0.5 * exp_scale + 0.5 * new_exp_scale

    linear_scale = linear_scale.to(W.dtype)
    exp_scale = exp_scale.to(W.dtype)
    indices = indices.to(W.dtype)
    sign_W = sign_W.to(W.dtype)

    pow_indices = 2.0 ** indices
    quant_W_abs = exp_scale * pow_indices + linear_scale * indices
    quant_W = sign_W * quant_W_abs

    quant_W = quant_W.reshape(wshape)    
    sign_W = torch.sign(quant_W)
    
    return quant_W, [sign_W, indices, exp_scale.squeeze(-1), linear_scale.squeeze(-1)]

def pseudo_mix_quant_otherbase(wbits, W, w_group_size=0, iterations=100, grid_num=10, X=None, base=2):  
    """Optimized MixNLQ """
    assert base > 0, "base must > 0 !"
    wshape = W.shape
    
    if w_group_size > 0:
        W = W.reshape(-1, w_group_size)
    abs_W = torch.abs(W)
    sign_W = torch.sign(W)

    if X is None:
        lam_scale = 1.0
    else:
        lam = X.unsqueeze(0).expand(wshape)
        lam = lam / lam.max(dim=1, keepdim=True)[0].clamp_min(1e-8)
        lam_scale = lam.reshape(W.shape)  

    maxq = 2 ** (wbits - 1) - 1
    
    xmin = abs_W.min(dim=1, keepdim=True)[0]
    xmax = abs_W.max(dim=1, keepdim=True)[0]
    
    org_linear_scale = (xmax - xmin) / maxq
    org_exp_scale = (xmax - xmin) * (base ** (-maxq))
   
    best_mse = None
    best_exp_lamb = torch.zeros_like(org_exp_scale)
    assert grid_num > 0, "grid_num must > 0 !"
  
    levels = torch.arange(0, maxq + 1, dtype=W.dtype, device=W.device)  
    pow2_levels = base ** levels                                         
    
    abs_W_exp = abs_W.unsqueeze(-1) 
    linear_factor = (2**maxq - 1) / (2**maxq)
    
    # --- Grid Search 阶段 ---
    for i in range(grid_num + 1):
        exp_lamb = i / grid_num
        linear_lamb = (1 - exp_lamb) * linear_factor
        
        exp_scale = org_exp_scale * exp_lamb        
        linear_scale = org_linear_scale * linear_lamb 
        
        quant_level_mix = exp_scale * pow2_levels + linear_scale * levels
        
        distances = torch.abs(abs_W_exp - quant_level_mix.unsqueeze(1)) 
        indices = torch.argmin(distances, dim=-1)
        
        pow2_indices = base ** indices
        quant_W_abs = exp_scale * pow2_indices + linear_scale * indices
        
        mse = torch.mean((abs_W - quant_W_abs)**2, dim=-1, keepdim=True)
        
        if best_mse is None:
            best_mse = mse
            best_exp_lamb = torch.full_like(best_exp_lamb, exp_lamb)
        else:
            better_mask = mse < best_mse
            best_exp_lamb = torch.where(better_mask, exp_lamb, best_exp_lamb)
            best_mse = torch.where(better_mask, mse, best_mse)

    exp_scale = org_exp_scale * best_exp_lamb
    linear_scale = org_linear_scale * (1 - best_exp_lamb) * linear_factor

    for it in range(iterations):
        quant_level_mix = exp_scale * pow2_levels + linear_scale * levels

        distances = torch.abs(abs_W_exp - quant_level_mix.unsqueeze(1)) 
        indices = torch.argmin(distances, dim=-1)

        pow_indices = base ** indices
        pow_indices_sq = pow_indices ** 2
        indices_sq = indices ** 2

        numerator_linear = torch.sum((abs_W - exp_scale * pow_indices) * indices * lam_scale, dim=-1, keepdim=True)
        denominator_linear = torch.sum(indices_sq * lam_scale, dim=-1, keepdim=True)
        new_linear_scale = numerator_linear / (denominator_linear + 1e-8)  

        numerator_exp = torch.sum((abs_W - linear_scale * indices) * pow_indices * lam_scale, dim=-1, keepdim=True)
        denominator_exp = torch.sum(pow_indices_sq * lam_scale, dim=-1, keepdim=True)
        new_exp_scale = numerator_exp / (denominator_exp + 1e-8)
        
        linear_scale = 0.5 * linear_scale + 0.5 * new_linear_scale
        exp_scale = 0.5 * exp_scale + 0.5 * new_exp_scale

    linear_scale = linear_scale.to(W.dtype)
    exp_scale = exp_scale.to(W.dtype)
    indices = indices.to(W.dtype)
    sign_W = sign_W.to(W.dtype)

    pow_indices = base ** indices
    quant_W_abs = exp_scale * pow_indices + linear_scale * indices
    quant_W = sign_W * quant_W_abs

    quant_W = quant_W.reshape(wshape)    
    sign_W = torch.sign(quant_W)
    
    return quant_W, [sign_W, indices, exp_scale.squeeze(-1), linear_scale.squeeze(-1)]


def pseudo_mix_quant_act(wbits, W, w_group_size = 0, iterations=100, exp_lamb=-1, mix8bit=False):  
    wshape = W.shape
    # wdtype = W.dtype
    # W = W.float()
    if w_group_size > 0:
        W = W.reshape(-1, w_group_size)

    #1bit sign
    sign_W = torch.sign(W)

    maxw = torch.abs(W).std(dim=-1)
    mask_8bit = torch.zeros_like(maxw)
    if mix8bit == True:
        # maxw = torch.max(torch.abs(W), dim=-1)[0]
        sort_row = torch.argsort(maxw, descending=True)
        # mask_8bit = torch.zeros_like(maxw)
        mask_8bit[sort_row[:int(0.1 * len(maxw))]] = 1

  
    #quantization exp  3bit
    maxq = 2 ** (wbits-1) - 1

    if exp_lamb >= 0 and exp_lamb <= 1:
        linear_lamb = 1.0 - exp_lamb
    else: 
        #use gaussian distribution to adjust
        gd_level = gd_quant(W, wbits=wbits)
        max_inter = gd_level[-1]-gd_level[-2]
        linear_inter = 1 / (maxq) 
        exp_inter = 0.5 
        linear_lamb = (max_inter - exp_inter) / (linear_inter-exp_inter)
        exp_lamb = (max_inter - linear_inter) / (exp_inter-linear_inter)

        # print(linear_lamb,exp_lamb)
    #RTN quantization
    xmin = torch.abs(W).min(dim=1)[0]
    xmax = torch.abs(W).max(dim=1)[0] 
    # Calculate scale and zero offset for each group
    linear_scale = (xmax - xmin) / (maxq)
    zeros = -xmin / linear_scale

    linear_scale *= linear_lamb   #init
    quant_level_2 = linear_scale.unsqueeze(1) * (torch.arange(0, maxq+1, dtype=W.dtype, device=W.device).repeat(W.shape[0], 1) - zeros.unsqueeze(1))
    
    # exp quantization
    quant_level_1 = torch.arange(0, maxq+1, dtype=W.dtype, device=W.device).repeat(W.shape[0], 1)
    exp_scale = (xmax - xmin) * (2**(-maxq))
    exp_scale *= exp_lamb
    quant_level_1 =  exp_scale.unsqueeze(1)  * (2 ** quant_level_1)

    for it in range(iterations):
        quant_level_1 = torch.arange(0, maxq+1, dtype=W.dtype, device=W.device).repeat(W.shape[0], 1)
        quant_level_1 =  exp_scale.unsqueeze(-1) * (2 ** quant_level_1)
        quant_level_2 = linear_scale.unsqueeze(1) * (torch.arange(0, maxq+1, dtype=W.dtype, device=W.device).repeat(W.shape[0], 1))# - zeros.unsqueeze(1))
        quant_level_mix = quant_level_1 + quant_level_2

        # 计算与所有量化等级的距离
        distances = torch.abs(torch.abs(W).unsqueeze(-1) - quant_level_mix.unsqueeze(1)) 
        indices = torch.argmin(distances, dim=-1)
        indices = torch.clamp(indices, 0, maxq)

        #update U and scale
        linear_scale = torch.sum((torch.abs(W) - exp_scale.unsqueeze(1) * (2.0 ** indices)) * indices, dim=-1) / torch.sum(indices**2, dim=-1)
        exp_scale = torch.sum((torch.abs(W) - indices * linear_scale.unsqueeze(1)) * (2.0 ** indices), dim=-1) / torch.sum((2.0 ** indices)**2, dim=-1)

    exp_scale = exp_scale.to(W.dtype)
    linear_scale = linear_scale.to(W.dtype)
    indices = indices.to(W.dtype)
    sign_W = sign_W.to(W.dtype)

    quant_W1 =  exp_scale.unsqueeze(-1) * (2.0 ** (indices))
    quant_W2 = linear_scale.unsqueeze(-1) * (indices)
    quant_W = sign_W*((quant_W1+quant_W2).to(W.dtype))
    quant_W = quant_W.reshape(wshape)    
    
    # if mix8bit == False:
    #     return quant_W, [sign_W, indices, exp_scale, linear_scale, quant_level_mix]
    # else:
    return quant_W, [sign_W, indices, exp_scale, linear_scale, mask_8bit]


def pseudo_mix_quant_apply(W, sign_W, exp_scale, linear_scale, wbits=4):
    maxq = 2**(wbits-1)-1
    quant_level_1 = torch.arange(0, maxq+1, dtype=W.dtype, device=W.device).repeat(W.shape[0], 1)
    quant_level_1 =  exp_scale.unsqueeze(-1) * (2 ** quant_level_1)
    quant_level_2 = linear_scale.unsqueeze(1) * (torch.arange(0, maxq+1, dtype=W.dtype, device=W.device).repeat(W.shape[0], 1))# - zeros.unsqueeze(1))
    quant_level_mix = quant_level_1 + quant_level_2

    distances = torch.abs(torch.abs(W).unsqueeze(-1) - quant_level_mix.unsqueeze(1)) 
    indices = torch.argmin(distances, dim=-1)
    indices = torch.clamp(indices, 0, maxq)

    quant_W1 =  exp_scale.unsqueeze(-1) * (2.0 ** (indices)) # + exp_bias.unsqueeze(-1)))
    quant_W2 = linear_scale.unsqueeze(-1) * (indices) # - zeros.unsqueeze(1))
    # zero = torch.mean((torch.abs(W) - (quant_W1+quant_W2)), dim=1)
    quant_W = sign_W *((quant_W1+quant_W2).to(W.dtype))
    return quant_W


import time
def fasterquant(W, H, dXXT=None, wbits=4, blocksize=128, percdamp=0.01, groupsize=-1, actorder=False, static_groups=False, mixquant_params = None):
    torch.cuda.empty_cache()
    wdtype=W.dtype
    W = W.float()
    tick = time.time()

    maxq = 2**wbits -1
    
    if groupsize > 0:
        group_num = W.shape[1] // groupsize
    else:
        group_num = 1

    if mixquant_params == None:
        q, scale, zeros, quant_W = pseudo_RTN_quant(wbits, W, group_size=groupsize, quantile=1)
    else:
        if len(mixquant_params) == 5:
            [_, q, exp_scale, linear_scale, bias] = mixquant_params
        else:
            [_, q, exp_scale, linear_scale] = mixquant_params
            bias=None
        quant_level_1 = torch.arange(0, maxq+1, dtype=exp_scale.dtype, device=exp_scale.device).repeat(W.shape[0] * group_num, 1)
        quant_level_1 =  exp_scale.unsqueeze(-1) * (2 ** quant_level_1)
        quant_level_2 = linear_scale.unsqueeze(1) * (torch.arange(0, maxq+1, dtype=linear_scale.dtype, device=linear_scale.device).repeat(W.shape[0] * group_num, 1))# - zeros.unsqueeze(1))
        if bias is not None:
            quant_level_mix = quant_level_1 + quant_level_2 + bias.unsqueeze(-1)
        else:
            quant_level_mix = quant_level_1 + quant_level_2

    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    if dXXT != None:
        dXXT[:, dead] = 0

    perm = torch.argsort(torch.diag(H), descending=True)
    W = W[:, perm]
    H = H[perm][:, perm]
    if dXXT != None:
        dXXT = dXXT[perm][:, perm]
    invperm = torch.argsort(perm)

    Losses = torch.zeros_like(W)
    Q = torch.zeros_like(W, dtype=torch.int32)
    quant_W = torch.zeros_like(W)
    sign_W = torch.zeros_like(W, dtype=torch.int32)

    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[0], device=H.device)
    H[diag, diag] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    Hinv = H

    if dXXT != None:
        alpha = 0.25
        P = alpha * ((dXXT @ Hinv.T).triu_(diagonal=1)) @ Hinv

    cnt = 0    
    for i1 in range(0, W.shape[1], blocksize):
        i2 = min(i1+blocksize, W.shape[1])
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Losses1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        if dXXT != None:
            P1 = P[i1:i2, i1:i2]

        for i in range(count):
            w = W1[:, i]
            d = Hinv1[i, i]

            if groupsize > 0 :
                cnt = perm[i1+i] // groupsize
                
            if mixquant_params == None:
                if groupsize > 0:
                    q1 = torch.clamp(torch.round(w / scale[cnt::group_num]) + zeros[cnt::group_num], 0, maxq)
                    deq_q1 = scale[cnt::group_num] * (q1  - zeros[cnt::group_num])
                else:
                    q1 = torch.clamp(torch.round(w / scale) + zeros, 0, maxq)
                    deq_q1 = scale * (q1  - zeros)
            else:
                if groupsize > 0:
                    distances = torch.abs(torch.abs(w.unsqueeze(-1)) - quant_level_mix[cnt::group_num]) 
                    q1 = torch.argmin(distances, dim=-1)
                    q1 = torch.clamp(q1, 0, maxq)
                    quant_W1 =  exp_scale[cnt::group_num] * (2.0 ** (q1)) 
                    quant_W2 = linear_scale[cnt::group_num] * (q1)
                    if bias is not None:
                        deq_q1 = torch.sign(w) * ((quant_W1+quant_W2+bias[cnt::group_num]).to(W.dtype)) 
                    else:
                        deq_q1 = torch.sign(w) * ((quant_W1+quant_W2).to(W.dtype))

                else:
                    distances = torch.abs(torch.abs(w.unsqueeze(-1)) - quant_level_mix) 
                    q1 = torch.argmin(distances, dim=-1)
                    q1 = torch.clamp(q1, 0, maxq)
                    quant_W1 =  exp_scale * (2.0 ** (q1)) 
                    quant_W2 = linear_scale * (q1) 
                    if bias is not None:
                        deq_q1 = torch.sign(w) * ((quant_W1+quant_W2+bias).to(W.dtype)) 
                    else:
                        deq_q1 = torch.sign(w) * ((quant_W1+quant_W2).to(W.dtype))
                   
            Q[:, i1+i] = q1
            quant_W[:, i1+i] = deq_q1
            sign_W[:, i1+i] = torch.sign(deq_q1)
            Losses1[:, i] = (w-deq_q1) ** 2 / d ** 2
            err1 = (w-deq_q1) / d
            if dXXT != None:
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0)) - w.unsqueeze(1).matmul(P1[i, i:].unsqueeze(0))
            else:
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1
        
        Losses[:, i1:i2] = Losses1 / 2
        if dXXT != None:
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])- W1.matmul(P[i1:i2, i2:])
        else:
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

    quant_W = quant_W[:, invperm]    
    Q = Q[:, invperm]
    sign_W = sign_W[:, invperm]

    print('time %.2f' % (time.time() - tick))
    print('error', torch.sum(Losses).item())

    # Reconstruct the quantized weights
    return quant_W.to(wdtype) , sign_W, Q


@torch.inference_mode()
def activation_aware_mixnlq_quant(
    W,
    activation_samples=None,
    hessian=None,
    wbits=4,
    group_size=0,
    iterations=100,
):
    """Run the activation-aware MixNLQ path used by ``quant.py``.

    Prefer passing ``hessian`` from the online sequence-wise collector. The
    ``activation_samples`` path remains available for older callers.
    """
    if hessian is None:
        if activation_samples is None:
            raise ValueError(
                "activation-aware MixNLQ requires a calibration Hessian"
            )
        sample_rows = activation_samples.reshape(-1, W.shape[1]).float()
        if sample_rows.shape[0] == 0:
            raise ValueError(
                "activation-aware MixNLQ requires calibration tokens"
            )
        hessian = sample_rows.transpose(0, 1).matmul(sample_rows)
        hessian.mul_(2.0 / sample_rows.shape[0])
    else:
        hessian = hessian.clone()
    importance = hessian.diagonal().clamp_min(0).sqrt()

    if wbits <= 3:
        _, mix_parameters = pseudo_mix_quant_bias(
            wbits=wbits,
            W=W.detach().clone(),
            w_group_size=group_size,
            iterations=iterations,
            X=importance,
            alpha=1.5,
        )
    else:
        _, mix_parameters = pseudo_mix_quant(
            wbits=wbits,
            W=W.detach().clone(),
            w_group_size=group_size,
            iterations=iterations,
            X=importance,
            alpha=1.0,
        )

    dequantized, signs, indices = fasterquant(
        W=W.detach().clone(),
        H=hessian,
        wbits=wbits,
        mixquant_params=mix_parameters,
        groupsize=group_size,
        dXXT=None,
    )
    return dequantized, signs, indices, mix_parameters
