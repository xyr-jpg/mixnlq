"""Inference-only QKV and Gate/Up projection fusion for Hugging Face Llama."""

from types import MethodType

import torch
from torch import nn

from utils.triton_quant_linear import CustomBF16Linear, TritonInt4Linear


def _bias_or_zeros(module: nn.Module, size: int, reference: torch.Tensor) -> torch.Tensor:
    if getattr(module, "has_bias", False):
        return module.bias
    if isinstance(module, nn.Linear) and module.bias is not None:
        return module.bias
    return torch.zeros(size, dtype=reference.dtype, device=reference.device)


def fuse_linear_modules(modules: list[nn.Module]) -> nn.Module:
    first = modules[0]
    out_sizes = [module.out_features for module in modules]
    if any(module.in_features != first.in_features for module in modules):
        raise ValueError("Fused projections must have the same in_features")

    if all(isinstance(module, nn.Linear) for module in modules):
        has_bias = any(module.bias is not None for module in modules)
        fused = nn.Linear(
            first.in_features,
            sum(out_sizes),
            bias=has_bias,
            dtype=first.weight.dtype,
            device=first.weight.device,
        )
        fused.weight.data.copy_(torch.cat([module.weight for module in modules], dim=0))
        if has_bias:
            fused.bias.data.copy_(
                torch.cat(
                    [
                        _bias_or_zeros(module, size, first.weight)
                        for module, size in zip(modules, out_sizes)
                    ]
                )
            )
        return fused

    if all(isinstance(module, CustomBF16Linear) for module in modules):
        reference = first.weight_kn
        temporary = nn.Linear(
            first.in_features,
            sum(out_sizes),
            bias=any(module.has_bias for module in modules),
            dtype=reference.dtype,
            device=reference.device,
        )
        temporary.weight.data.copy_(
            torch.cat([module.weight_kn.transpose(0, 1) for module in modules], dim=0)
        )
        if temporary.bias is not None:
            temporary.bias.data.copy_(
                torch.cat(
                    [
                        _bias_or_zeros(module, size, reference)
                        for module, size in zip(modules, out_sizes)
                    ]
                )
            )
        return CustomBF16Linear(temporary)

    if all(isinstance(module, TritonInt4Linear) for module in modules):
        quant_mode = first.quant_mode
        if any(
            module.quant_mode != quant_mode
            or module.group_size != first.group_size
            or module.in_features != first.in_features
            or module.packed_layout != first.packed_layout
            for module in modules
        ):
            raise ValueError("INT4 projections must use identical quantization settings")
        has_bias = any(module.has_bias for module in modules)
        reference = first.scale_0
        bias = (
            torch.cat(
                [
                    _bias_or_zeros(module, size, reference)
                    for module, size in zip(modules, out_sizes)
                ]
            )
            if has_bias
            else None
        )
        scale_2 = None
        if quant_mode in {"nf4_lut", "lut3"}:
            if any(module.group_lut.numel() == 0 for module in modules):
                raise ValueError(
                    f"Cannot fuse {quant_mode} projections: a source layer "
                    "has no group_lut buffer"
                )
            scale_0 = torch.empty(
                0, dtype=reference.dtype, device=reference.device
            )
            scale_1 = torch.empty(
                0, dtype=reference.dtype, device=reference.device
            )
            codebook = None
            group_lut = torch.cat(
                [module.group_lut for module in modules],
                dim=1 if quant_mode == "nf4_lut" else 2,
            ).contiguous()
        elif quant_mode == "nf4":
            scale_0 = torch.cat([module.scale_0 for module in modules], dim=1)
            scale_1 = torch.empty(
                0, dtype=reference.dtype, device=reference.device
            )
            codebook = first.codebook
            group_lut = None
        else:
            scale_0 = torch.cat([module.scale_0 for module in modules], dim=1)
            scale_1 = torch.cat([module.scale_1 for module in modules], dim=1)
            if quant_mode == "mixnlq3":
                scale_2 = torch.cat(
                    [module.scale_2 for module in modules], dim=1
                )
            codebook = None
            group_lut = None
        return TritonInt4Linear(
            packed_weight=torch.cat(
                [module.packed_weight for module in modules],
                # K-major uses dim 1 as N. The shared tiled INT4 layout is
                # [K/16, N/64, lane, packed-K-inner, output-pair], so its
                # output-tile dimension is also dim 1.
                dim=1,
            ),
            scale_0=scale_0,
            scale_1=scale_1,
            bias=bias,
            in_features=first.in_features,
            out_features=sum(out_sizes),
            group_size=first.group_size,
            quant_mode=quant_mode,
            codebook=codebook,
            group_lut=group_lut,
            scale_2=scale_2,
        )

    raise TypeError(f"Unsupported projection types: {[type(module) for module in modules]}")


class FusedGateUpMLP(nn.Module):
    def __init__(self, mlp: nn.Module) -> None:
        super().__init__()
        self.intermediate_size = mlp.gate_proj.out_features
        self.gate_up_proj = fuse_linear_modules([mlp.gate_proj, mlp.up_proj])
        self.down_proj = mlp.down_proj
        self.act_fn = mlp.act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.split(self.intermediate_size, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)


def _fused_llama_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings,
    attention_mask,
    past_key_value=None,
    cache_position=None,
    **kwargs,
):
    from transformers.models.llama.modeling_llama import (
        ALL_ATTENTION_FUNCTIONS,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    input_shape = hidden_states.shape[:-1]
    q_size, k_size, v_size = self._fused_qkv_sizes
    qkv = self.qkv_proj(hidden_states)
    query_states, key_states, value_states = qkv.split(
        (q_size, k_size, v_size), dim=-1
    )
    query_states = query_states.view(
        *input_shape, self.config.num_attention_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        *input_shape, self.config.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        *input_shape, self.config.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )
    if past_key_value is not None:
        key_states, value_states = past_key_value.update(
            key_states,
            value_states,
            self.layer_idx,
            {"sin": sin, "cos": cos, "cache_position": cache_position},
        )

    attention_interface = eager_attention_forward
    if self.config._attn_implementation != "eager":
        if not (
            self.config._attn_implementation == "sdpa"
            and kwargs.get("output_attentions", False)
        ):
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]
    causal_mask = attention_mask
    if causal_mask is not None:
        causal_mask = causal_mask[..., : key_states.shape[-2]]
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        causal_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(attn_output), attn_weights


def fuse_llama_projections(model: nn.Module) -> tuple[int, int]:
    if getattr(model.config, "model_type", None) != "llama":
        return 0, 0
    qkv_count = 0
    mlp_count = 0
    for layer in model.model.layers:
        attention = layer.self_attn
        q_size = attention.q_proj.out_features
        k_size = attention.k_proj.out_features
        v_size = attention.v_proj.out_features
        attention.qkv_proj = fuse_linear_modules(
            [attention.q_proj, attention.k_proj, attention.v_proj]
        )
        attention._fused_qkv_sizes = (q_size, k_size, v_size)
        attention.q_proj = None
        attention.k_proj = None
        attention.v_proj = None
        attention.forward = MethodType(_fused_llama_attention_forward, attention)
        qkv_count += 1

        layer.mlp = FusedGateUpMLP(layer.mlp)
        mlp_count += 1
    return qkv_count, mlp_count
