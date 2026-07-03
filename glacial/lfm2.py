"""LFM2 MoE math helpers for the Glacial runtime.

This module contains all LFM2-specific execution math, proven by parity probes
against HF golden traces.  It is private to the ``Lfm2MoeBackend`` adapter.

Key differences from Granite (see docs/lfm2-reference.md):
  - Hybrid conv/attention: 18 conv layers + 6 GQA attention layers
  - First 2 layers dense MLP, layers 2-23 MoE (32 experts, 4 active)
  - Short conv: gated depthwise conv1d (kernel=3) with conv state cache
  - Attention: Q/K layernorm (RMSNorm on Q and K after projection)
  - Router: sigmoid + expert_bias + top-k (not softmax)
  - Experts: separate per-expert w1/w2/w3 tensors (not stacked 3D)
  - No multipliers (standard residuals, no embedding/attention/logits scaling)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from glacial.weights import BF16_BYTES, SafetensorsWeights, WeightBudget

FINAL_NORM_TENSOR = "model.embedding_norm.weight"
EMBED_TENSOR = "model.embed_tokens.weight"


def scalar_config(config: dict[str, Any]) -> dict[str, float]:
    """Extract LFM2 scalar config. No multipliers — standard transformer."""
    rope_params = config.get("rope_parameters", {})
    if not isinstance(rope_params, dict):
        rope_params = {}
    return {
        "norm_eps": float(config.get("norm_eps", 1e-5)),
        "rope_theta": float(rope_params.get("rope_theta", 5_000_000.0)),
        # No multipliers for LFM2 — standard residuals and scaling
        "embedding_multiplier": 1.0,
        "logits_scaling": 1.0,
    }


def layer_tensor(layer_idx: int, suffix: str) -> str:
    return f"model.layers.{layer_idx}.{suffix}"


# ---------------------------------------------------------------------------
# RMSNorm (identical math to Granite, different eps default)
# ---------------------------------------------------------------------------

def lfm2_rmsnorm(hidden, weight, *, eps: float):
    """RMSNorm matching Lfm2MoeRMSNorm.forward.

    Mathematically identical to Granite's granite_rmsnorm:
      x_f32 = x.float()
      normed = x_f32 * rsqrt(mean(x_f32^2) + eps)
      return weight * normed.to(input_dtype)
    """
    import torch

    input_dtype = hidden.dtype
    hidden_f32 = hidden.to(torch.float32)
    variance = hidden_f32.pow(2).mean(-1, keepdim=True)
    normalized = hidden_f32 * torch.rsqrt(variance + eps)
    return weight * normalized.to(input_dtype)


# ---------------------------------------------------------------------------
# RoPE (identical math to Granite, different theta)
# ---------------------------------------------------------------------------

def rotate_half(x):
    import torch
    return torch.cat((-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1)


def lfm2_rope_cos_sin(*, position_ids, head_dim: int, rope_theta: float, dtype):
    import torch

    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype=dtype)
    sin = emb.sin().to(dtype=dtype)
    return cos, sin


def apply_rotary_pos_emb(q, k, cos, sin, *, unsqueeze_dim: int = 1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states, n_rep: int):
    import torch

    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# ---------------------------------------------------------------------------
# Causal mask (same structure as Granite's)
# ---------------------------------------------------------------------------

def build_causal_mask(*, attention_mask, input_tensor, cache_position):
    import torch

    dtype = input_tensor.dtype
    device = input_tensor.device
    min_dtype = torch.finfo(dtype).min
    sequence_length = input_tensor.shape[1]
    target_length = attention_mask.shape[-1] if attention_mask is not None else sequence_length + 1

    causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
    causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
    if attention_mask is not None:
        causal_mask = causal_mask.clone()
        mask_length = attention_mask.shape[-1]
        padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
        padding_mask = padding_mask == 0
        causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)
    return causal_mask


# ---------------------------------------------------------------------------
# Short conv (gated depthwise conv1d)
# ---------------------------------------------------------------------------

def run_short_conv(
    *,
    layer_idx: int,
    hidden,  # [batch, seq, hidden_size]
    model_file: Path,
    header: dict[str, Any],
    payload_start: int,
    budget: WeightBudget | None = None,
    conv_state=None,  # [hidden_size, L_cache] or None for prefill
):
    """Run gated short convolution.

    Prefill (conv_state=None): processes full sequence, returns output and
    the final conv state for subsequent decode.

    Decode (conv_state provided): processes single token using the conv
    state, returns output and updated conv state.
    """
    import torch
    import torch.nn.functional as F

    provider = SafetensorsWeights(model_file, header=header, payload_start=payload_start, budget=budget)
    L_cache = 3  # kernel size

    in_proj_name = layer_tensor(layer_idx, "conv.in_proj.weight")
    out_proj_name = layer_tensor(layer_idx, "conv.out_proj.weight")
    conv_name = layer_tensor(layer_idx, "conv.conv.weight")

    with provider.tensor(in_proj_name) as in_proj_weight, \
         provider.tensor(out_proj_name) as out_proj_weight, \
         provider.tensor(conv_name) as conv_weight:

        batch_size, seq_len, hidden_size = hidden.shape

        # in_proj: [3*hidden, hidden] → B, C, x gates
        BCx = F.linear(hidden, in_proj_weight)  # [batch, seq, 3*hidden]
        BCx = BCx.transpose(-1, -2)  # [batch, 3*hidden, seq]
        B, C, x = BCx.chunk(3, dim=-2)  # each [batch, hidden, seq]

        Bx = B * x  # input gate [batch, hidden, seq]

        if conv_state is not None:
            # ---- Decode (single token) ----
            # conv_state: [hidden, L_cache] containing last L_cache Bx values
            # Update: shift left, append new Bx
            new_Bx = Bx[:, :, 0]  # [batch, hidden]
            # Roll conv_state and add new Bx at the end
            updated_state = torch.cat([conv_state[:, 1:], new_Bx[0:1].transpose(0, 1)], dim=-1)
            # conv_weight: [hidden, 1, L_cache] → [hidden, L_cache]
            w = conv_weight[:, 0, :]  # [hidden, L_cache]
            conv_out = (updated_state * w).sum(dim=-1)  # [hidden]
            conv_out = conv_out.unsqueeze(0).unsqueeze(-1)  # [1, hidden, 1]
            final_state = updated_state
        else:
            # ---- Prefill (full sequence) ----
            # Depthwise conv1d: [batch, hidden, seq] * [hidden, 1, L_cache] → [batch, hidden, seq+2]
            conv_out = F.conv1d(Bx, conv_weight, groups=hidden_size, padding=L_cache - 1)
            conv_out = conv_out[..., :seq_len]  # truncate to original length (causal)

            # Save conv state: last L_cache-1 Bx values for decode
            # conv_state shape: [hidden, L_cache] (need L_cache values for kernel size 3)
            final_state = Bx[0, :, -L_cache:].transpose(0, 1)  # [hidden, L_cache]

        y = C * conv_out  # output gate [batch, hidden, seq]
        y = y.transpose(-1, -2).contiguous()  # [batch, seq, hidden]
        output = F.linear(y, out_proj_weight)  # [batch, seq, hidden]

    return output, final_state


# ---------------------------------------------------------------------------
# Attention (with Q/K layernorm + RoPE + GQA)
# ---------------------------------------------------------------------------

def run_attention(
    *,
    layer_idx: int,
    hidden,  # [batch, seq, hidden_size]
    model_file: Path,
    header: dict[str, Any],
    payload_start: int,
    inputs: dict[str, Any],
    config: dict[str, Any],
    scalars: dict[str, float],
    kv_pair: tuple[Any, Any] | None = None,
    budget: WeightBudget | None = None,
):
    """Run GQA attention with Q/K layernorm and RoPE.

    Returns (attn_output, (key_cache, value_cache)).
    """
    import torch
    import torch.nn.functional as F

    provider = SafetensorsWeights(model_file, header=header, payload_start=payload_start, budget=budget)

    hidden_size = int(config.get("hidden_size", 2048))
    num_attention_heads = int(config.get("num_attention_heads", 32))
    num_key_value_heads = int(config.get("num_key_value_heads", 8))
    num_key_value_groups = num_attention_heads // num_key_value_heads
    head_dim = hidden_size // num_attention_heads  # 64

    attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long, device=hidden.device)
    position_ids = torch.tensor(inputs["position_ids"], dtype=torch.long, device=hidden.device)
    cache_position = torch.tensor(inputs["cache_position"], dtype=torch.long, device=hidden.device)

    q_name = layer_tensor(layer_idx, "self_attn.q_proj.weight")
    k_name = layer_tensor(layer_idx, "self_attn.k_proj.weight")
    v_name = layer_tensor(layer_idx, "self_attn.v_proj.weight")
    o_name = layer_tensor(layer_idx, "self_attn.out_proj.weight")
    q_norm_name = layer_tensor(layer_idx, "self_attn.q_layernorm.weight")
    k_norm_name = layer_tensor(layer_idx, "self_attn.k_layernorm.weight")

    with provider.tensor(o_name) as o_weight:
        with provider.tensor(q_name) as q_weight, \
             provider.tensor(k_name) as k_weight, \
             provider.tensor(v_name) as v_weight, \
             provider.tensor(q_norm_name) as q_norm_weight, \
             provider.tensor(k_norm_name) as k_norm_weight:

            q_proj = F.linear(hidden, q_weight)  # [batch, seq, num_heads * head_dim]
            k_proj = F.linear(hidden, k_weight)  # [batch, seq, num_kv_heads * head_dim]
            v_proj = F.linear(hidden, v_weight)  # [batch, seq, num_kv_heads * head_dim]

            batch_size, q_len, _ = q_proj.shape

            # Q/K layernorm (new vs Granite)
            query_states = lfm2_rmsnorm(
                q_proj.view(batch_size, q_len, num_attention_heads, head_dim),
                q_norm_weight, eps=scalars["norm_eps"],
            ).transpose(1, 2)  # [batch, num_heads, seq, head_dim]

            key_states = lfm2_rmsnorm(
                k_proj.view(batch_size, q_len, num_key_value_heads, head_dim),
                k_norm_weight, eps=scalars["norm_eps"],
            ).transpose(1, 2)  # [batch, num_kv_heads, seq, head_dim]

            value_states = v_proj.view(batch_size, q_len, num_key_value_heads, head_dim).transpose(1, 2)

        # RoPE
        cos, sin = lfm2_rope_cos_sin(
            position_ids=position_ids, head_dim=head_dim,
            rope_theta=scalars["rope_theta"], dtype=query_states.dtype,
        )
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)

        # KV cache
        if kv_pair is None:
            key_cache = key_states.contiguous()
            value_cache = value_states.contiguous()
        else:
            past_key, past_value = kv_pair
            key_cache = torch.cat([past_key, key_states], dim=2).contiguous()
            value_cache = torch.cat([past_value, value_states], dim=2).contiguous()

        # GQA: repeat KV heads
        key_for_attn = repeat_kv(key_cache, num_key_value_groups)
        value_for_attn = repeat_kv(value_cache, num_key_value_groups)

        # Attention: standard 1/sqrt(head_dim) scaling (no multiplier)
        scaling = head_dim ** -0.5
        attn_weights = torch.matmul(query_states, key_for_attn.transpose(2, 3)) * scaling
        causal_mask = build_causal_mask(
            attention_mask=attention_mask, input_tensor=hidden, cache_position=cache_position,
        )
        attn_weights = attn_weights + causal_mask[:, :, :, : key_for_attn.shape[-2]]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_for_attn)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, q_len, -1)
        attn_output = F.linear(attn_output, o_weight)

    return attn_output, (key_cache, value_cache)


# ---------------------------------------------------------------------------
# MoE router (sigmoid + expert_bias + top-k)
# ---------------------------------------------------------------------------

def route_tokens(
    router_input,  # [num_tokens, hidden_size]
    router_weight,  # [num_experts, hidden_size]
    *,
    top_k: int,
    num_experts: int,
    expert_bias=None,  # [num_experts] F32 or None
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = 1.0,
    use_expert_bias: bool = True,
):
    """Route tokens to experts using sigmoid + expert_bias + top-k.

    Different from Granite's softmax router:
      1. Sigmoid activation (not softmax on full logits)
      2. Expert bias added before top-k selection
      3. Top-k weights normalized (norm_topk_prob)
      4. Scaled by routed_scaling_factor
    """
    import torch
    import torch.nn.functional as F

    router_logits = F.linear(router_input, router_weight)  # [num_tokens, num_experts]
    routing_weights = router_logits.sigmoid()  # sigmoid, not softmax!

    if use_expert_bias and expert_bias is not None:
        scores = routing_weights + expert_bias  # [num_tokens, num_experts]
        _, selected_experts = torch.topk(scores, k=top_k, dim=-1)
        routing_weights = torch.gather(routing_weights, 1, selected_experts).type_as(router_logits)
    else:
        routing_weights, selected_experts = torch.topk(routing_weights, k=top_k, dim=-1)

    if norm_topk_prob:
        routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-6)

    routing_weights = routing_weights * routed_scaling_factor

    return {
        "router_logits": router_logits,
        "routing_weights": routing_weights,
        "selected_experts": selected_experts,
        "top_k": top_k,
        "num_experts": num_experts,
    }


# ---------------------------------------------------------------------------
# MoE experts (separate per-expert w1/w2/w3)
# ---------------------------------------------------------------------------

def run_experts(
    *,
    layer_idx: int,
    provider: SafetensorsWeights,
    router_input,  # [num_tokens, hidden_size]
    route: dict[str, Any],
):
    """Execute selected experts one at a time.

    Unlike Granite's stacked 3D expert tensors, LFM2 stores each expert's
    w1/w2/w3 as separate 2D tensors: feed_forward.experts.{j}.w1.weight etc.
    """
    import torch
    import torch.nn.functional as F

    selected_experts = route["selected_experts"]  # [num_tokens, top_k]
    routing_weights = route["routing_weights"]    # [num_tokens, top_k]
    top_k = route["top_k"]
    num_experts = route["num_experts"]
    num_tokens = router_input.shape[0]

    # Build expert → token mapping
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts)  # [num_tokens, top_k, num_experts]
    expert_mask = expert_mask.permute(2, 1, 0)  # [num_experts, top_k, num_tokens]
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    final_hidden = torch.zeros_like(router_input)
    selected_expert_ids = []
    cumulative_expert_weight_bytes = 0
    peak_expert_pair_bytes = 0

    for expert_idx in expert_hit:
        expert_idx = int(expert_idx[0])
        if expert_idx == num_experts:
            continue

        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = router_input[token_idx]  # [num_selected, hidden]

        w1_name = layer_tensor(layer_idx, f"feed_forward.experts.{expert_idx}.w1.weight")
        w3_name = layer_tensor(layer_idx, f"feed_forward.experts.{expert_idx}.w3.weight")
        w2_name = layer_tensor(layer_idx, f"feed_forward.experts.{expert_idx}.w2.weight")

        with provider.tensor(w1_name) as w1, \
             provider.tensor(w3_name) as w3, \
             provider.tensor(w2_name) as w2:

            expert_bytes = (w1.numel() + w3.numel() + w2.numel()) * BF16_BYTES
            cumulative_expert_weight_bytes += expert_bytes
            peak_expert_pair_bytes = max(peak_expert_pair_bytes, expert_bytes)
            selected_expert_ids.append(expert_idx)

            # Match HF's combined gate_up_proj matmul to avoid BF16 BLAS dispatch differences
            gate_up = torch.cat([w1, w3], dim=0)  # [2*intermediate, hidden]
            gate_up_out = F.linear(current_state, gate_up)  # [num_selected, 2*intermediate]
            gate, up = gate_up_out.chunk(2, dim=-1)
            hidden = F.silu(gate) * up
            output = F.linear(hidden, w2)         # [num_selected, hidden]

            # Scale by routing weights
            output = output * routing_weights[token_idx, top_k_pos, None]
            final_hidden.index_add_(0, token_idx, output.to(final_hidden.dtype))

    return {
        "moe_output": final_hidden,
        "selected_expert_ids": selected_expert_ids,
        "cumulative_expert_weight_bytes": cumulative_expert_weight_bytes,
        "peak_expert_pair_bytes": peak_expert_pair_bytes,
    }


# ---------------------------------------------------------------------------
# Dense MLP (layers 0-1)
# ---------------------------------------------------------------------------

def run_dense_mlp(
    *,
    layer_idx: int,
    provider: SafetensorsWeights,
    hidden,  # [batch, seq, hidden_size]
):
    """Run standard SwiGLU MLP for dense layers (0-1)."""
    import torch.nn.functional as F

    w1_name = layer_tensor(layer_idx, "feed_forward.w1.weight")
    w3_name = layer_tensor(layer_idx, "feed_forward.w3.weight")
    w2_name = layer_tensor(layer_idx, "feed_forward.w2.weight")

    with provider.tensor(w1_name) as w1, \
         provider.tensor(w3_name) as w3, \
         provider.tensor(w2_name) as w2:
        gate = F.linear(hidden, w1)
        up = F.linear(hidden, w3)
        return F.linear(F.silu(gate) * up, w2)


# ---------------------------------------------------------------------------
# Layer execution
# ---------------------------------------------------------------------------

def run_layer(
    *,
    layer_idx: int,
    hidden,
    model_file: Path,
    header: dict[str, Any],
    payload_start: int,
    inputs: dict[str, Any],
    config: dict[str, Any],
    scalars: dict[str, float],
    budget: WeightBudget | None = None,
):
    """No-KV path: recompute full prompt. Returns (hidden, stats)."""
    hidden, _, stats = run_layer_with_optional_state(
        layer_idx=layer_idx, hidden=hidden, kv_pair=None, conv_state=None,
        model_file=model_file, header=header, payload_start=payload_start,
        inputs=inputs, config=config, scalars=scalars, return_state=False, budget=budget,
    )
    return hidden, stats


def run_layer_with_optional_state(
    *,
    layer_idx: int,
    hidden,
    kv_pair: tuple[Any, Any] | None,
    conv_state: Any | None,
    model_file: Path,
    header: dict[str, Any],
    payload_start: int,
    inputs: dict[str, Any],
    config: dict[str, Any],
    scalars: dict[str, float],
    return_state: bool = True,
    budget: WeightBudget | None = None,
):
    """Execute one decoder layer.

    Returns (hidden, (kv_pair or None, conv_state or None), stats).

    For attention layers: produces KV cache.
    For conv layers: produces conv state.
    """
    import torch

    provider = SafetensorsWeights(model_file, header=header, payload_start=payload_start, budget=budget)

    layer_types = config.get("layer_types", [])
    is_attention = layer_types[layer_idx] == "full_attention" if layer_idx < len(layer_types) else False
    num_dense = int(config.get("num_dense_layers", 2))
    is_dense_mlp = layer_idx < num_dense

    top_k = int(config.get("num_experts_per_tok", 4))
    num_experts = int(config.get("num_experts", 32))
    use_expert_bias = bool(config.get("use_expert_bias", True))
    norm_topk_prob = bool(config.get("norm_topk_prob", True))
    routed_scaling_factor = float(config.get("routed_scaling_factor", 1.0))
    hidden_size = int(config.get("hidden_size", 2048))

    op_norm_name = layer_tensor(layer_idx, "operator_norm.weight")
    ffn_norm_name = layer_tensor(layer_idx, "ffn_norm.weight")

    loaded_nonexpert_bytes = 0
    new_kv_pair = None
    new_conv_state = None

    # --- Operator (conv or attention) ---
    residual = hidden
    with provider.tensor(op_norm_name) as op_norm_weight:
        loaded_nonexpert_bytes += op_norm_weight.numel() * BF16_BYTES
        hidden_norm = lfm2_rmsnorm(hidden, op_norm_weight, eps=scalars["norm_eps"])

    if is_attention:
        attn_output, new_kv_pair = run_attention(
            layer_idx=layer_idx, hidden=hidden_norm,
            model_file=model_file, header=header, payload_start=payload_start,
            inputs=inputs, config=config, scalars=scalars,
            kv_pair=kv_pair, budget=budget,
        )
        loaded_nonexpert_bytes += 0  # attention weights counted inside run_attention
    else:
        conv_output, new_conv_state = run_short_conv(
            layer_idx=layer_idx, hidden=hidden_norm,
            model_file=model_file, header=header, payload_start=payload_start,
            budget=budget, conv_state=conv_state,
        )
        attn_output = conv_output

    hidden = residual + attn_output  # no multiplier

    # --- Feed-forward (dense MLP or MoE) ---
    residual = hidden
    with provider.tensor(ffn_norm_name) as ffn_norm_weight:
        loaded_nonexpert_bytes += ffn_norm_weight.numel() * BF16_BYTES
        hidden_norm = lfm2_rmsnorm(hidden, ffn_norm_weight, eps=scalars["norm_eps"])

    if is_dense_mlp:
        ff_output = run_dense_mlp(layer_idx=layer_idx, provider=provider, hidden=hidden_norm)
        expert_stats = {
            "selected_expert_ids": [],
            "selected_expert_count": 0,
            "cumulative_expert_weight_bytes": 0,
            "peak_expert_pair_bytes": 0,
        }
    else:
        # MoE: router + experts
        router_input = hidden_norm.reshape(-1, hidden_size)
        gate_name = layer_tensor(layer_idx, "feed_forward.gate.weight")

        # Load expert_bias (F32) if needed
        expert_bias = None
        if use_expert_bias:
            bias_name = layer_tensor(layer_idx, "feed_forward.expert_bias")
            with provider.tensor_any(bias_name) as bias_weight:
                expert_bias = bias_weight.float()

        with provider.tensor(gate_name) as gate_weight:
            loaded_nonexpert_bytes += gate_weight.numel() * BF16_BYTES
            route = route_tokens(
                router_input, gate_weight,
                top_k=top_k, num_experts=num_experts,
                expert_bias=expert_bias,
                norm_topk_prob=norm_topk_prob,
                routed_scaling_factor=routed_scaling_factor,
                use_expert_bias=use_expert_bias,
            )

        expert_result = run_experts(
            layer_idx=layer_idx, provider=provider,
            router_input=router_input, route=route,
        )
        batch_size, q_len, _ = hidden.shape
        ff_output = expert_result["moe_output"].view(batch_size, q_len, hidden_size)
        expert_stats = {
            "selected_expert_ids": expert_result["selected_expert_ids"],
            "selected_expert_count": len(expert_result["selected_expert_ids"]),
            "cumulative_expert_weight_bytes": expert_result["cumulative_expert_weight_bytes"],
            "peak_expert_pair_bytes": expert_result["peak_expert_pair_bytes"],
        }

    hidden = residual + ff_output  # no multiplier

    # --- Stats ---
    stats = {
        "layer": layer_idx,
        "layer_type": "full_attention" if is_attention else "conv",
        "is_dense_mlp": is_dense_mlp,
        **expert_stats,
        "loaded_nonexpert_bytes": loaded_nonexpert_bytes,
    }
    if budget is not None:
        stats.update({
            "weight_budget_current_bytes": budget.current_resident_bytes,
            "weight_budget_peak_bytes": budget.peak_resident_bytes,
            "weight_budget_total_visited_bytes": budget.total_visited_bytes,
            "weight_budget_violation_count": len(budget.violations),
        })
    if return_state:
        stats["kv_key_shape"] = list(new_kv_pair[0].shape) if new_kv_pair else None
        stats["kv_value_shape"] = list(new_kv_pair[1].shape) if new_kv_pair else None
        stats["conv_state_shape"] = list(new_conv_state.shape) if new_conv_state is not None else None
        return hidden, (new_kv_pair, new_conv_state), stats
    return hidden, None, stats