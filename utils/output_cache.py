import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import math
import torch
import torch.nn as nn
import transformers
import scipy
from utils.quant_utils import pseudo_mix_quant, pseudo_mix_quant_apply, pseudo_RTN_quant

class OutputCache:
    """record input and output in each layer block"""
    def __init__(self, layer, W, xbits=4, do_quant_act=False, qunat_x_iters=10, seqlen=1024):
        self.layer = layer
        self.hook = None
        self.dev = W.device
        self.rows = W.shape[0]
        self.columns = W.shape[1] 
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)  #hessian matrix
        self.do_quant_act = do_quant_act
        self.xbits = xbits
        self.x_exp_scale = torch.zeros(W.shape[1], dtype=W.dtype, device=W.device)
        self.x_linear_scale = torch.zeros(W.shape[1], dtype=W.dtype, device=W.device)
        self.n = 0
        self.qunat_x_iters = qunat_x_iters
        self.avg_X = torch.zeros((self.columns), device=self.dev,dtype=W.dtype) 
        self.X = torch.zeros((self.columns, seqlen), device=self.dev)    #bf16
        self.dXXT = torch.zeros((self.columns, self.columns), device=self.dev)  
        self.mask_8bit = torch.zeros(self.columns, device=self.dev)
        self.nsamples = 0
        self.cmp_flag = 0
    
    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
     
        self.X *= self.nsamples / (self.nsamples + tmp)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.avg_X *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        sort_X, _ = torch.sort(inp, dim=1) 
        self.X += (sort_X) / (self.nsamples) 
        self.avg_X += torch.mean(torch.abs(inp), dim=-1) / (self.nsamples) 

        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def add_batch_(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()

        mask = self.mask_8bit.clone().int()
        deq_x = inp.clone()
        if len(inp[mask==1]) > 0:
            q, scale, zeros, deq_x[mask==1] = pseudo_RTN_quant(8, inp[mask==1], group_size=inp.shape[-1])

            sign = torch.sign(deq_x[mask==0])
            max_w = torch.max(torch.abs(inp[mask==0]).reshape(-1, inp[mask==0].shape[-1]), dim=-1)[0]
            exp_scale = self.x_exp_scale[mask==0] * max_w
            linear_scale = self.x_linear_scale[mask==0] * max_w
            deq_x[mask==0] = pseudo_mix_quant_apply(inp[mask==0], sign, exp_scale, linear_scale, wbits=self.xbits)

        else:   
            sign = torch.sign(inp)
            max_w = torch.max(torch.abs(inp).reshape(-1, inp.shape[-1]), dim=-1)[0]
            exp_scale = self.x_exp_scale * max_w 
            linear_scale = self.x_linear_scale * max_w 
            deq_x = pseudo_mix_quant_apply(inp, sign, exp_scale, linear_scale, wbits=4)
        
        self.dXXT *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # self.deq_X += deq_x / (self.nsamples) 

        deq_x = math.sqrt(2 / self.nsamples) * deq_x
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # dX = math.sqrt(2 / self.nsamples) * dX.float()
        dX = inp - deq_x
        # print(deq_x.shape, dX.shape, inp.shape)
        self.dXXT += dX.matmul(inp.t())

    #define hook            
    def hook_fn(self, module, input, output):
        self.layer_output.append(output)
    
    def add_hook(self):
        if self.hook is None:
            self.hook = self.layer.register_forward_hook(self.hook_fn)
    
    def clear_hook(self):
        if self.hook is not None:
            self.hook.remove()
        self.hook = None
        torch.cuda.empty_cache()
        
    def clear_output(self):
        del self.layer_output
        self.layer_output = []
    