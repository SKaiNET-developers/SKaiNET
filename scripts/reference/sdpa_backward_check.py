"""Numerical check of the scaled dot-product attention adjoint (#1385).

Evaluates the hand-derived backward of `TensorOps.scaledDotProductAttention`

    S = s * Q K^T + M        P = softmax(S)        O = P V
    dV = P^T dO              dP = dO V^T           dS = P * (dP - rowsum(dP * P))
    dQ = s * dS K            dK = s * dS^T Q       dM = dS (summed to the mask's shape)

in float64 and compares it with PyTorch autograd for every option of the forward: default
and custom scale, causal, key-padding lengths, sliding window with right context, a dense
additive mask, grouped-query attention and a KV-cache decode step. The structural rules are
those of `AttentionMask` (absolute query position `seqKV - seqQ + qi`, masked score `-1e30`).
Where PyTorch has the same option, the forward is also compared with
`torch.nn.functional.scaled_dot_product_attention`.

    python3 scripts/reference/sdpa_backward_check.py

The derivation is in docs/modules/ROOT/pages/explanation/theory/attention-backward.adoc.
"""
import math

import torch
import torch.nn.functional as F

MASKED = -1.0e30
torch.set_default_dtype(torch.float64)


def rules(batch, seq_q, seq_kv, causal, window, right_context, lengths):
    """AttentionMask.additive: 0 where query row qi attends to key ki, MASKED elsewhere. [batch, 1, seqQ, seqKV]."""
    m = torch.full((batch, 1, seq_q, seq_kv), MASKED)
    for b in range(batch):
        length = seq_kv if lengths is None else lengths[b]
        for qi in range(seq_q):
            abs_q = seq_kv - seq_q + qi
            first = max(0, abs_q - window + 1) if window > 0 else 0
            end = min(seq_kv, length)
            if causal:
                end = min(end, abs_q + 1)
            if window > 0:
                end = min(end, abs_q + right_context + 1)
            m[b, 0, qi, first:max(first, end)] = 0.0
    return m


def forward(q, k, v, mask, scale, structural):
    """The forward as defined, with K/V of a grouped-query call repeated over their query heads."""
    n_rep = q.shape[1] // k.shape[1]
    k_full = k.repeat_interleave(n_rep, dim=1)
    v_full = v.repeat_interleave(n_rep, dim=1)
    s = scale * q @ k_full.transpose(-1, -2) + structural
    if mask is not None:
        # a mask with one entry per K/V head applies to every query head of that group
        per_kv_head = mask.dim() == 4 and mask.shape[1] == k.shape[1] != q.shape[1]
        s = s + (mask.repeat_interleave(n_rep, dim=1) if per_kv_head else mask)
    p = torch.softmax(s, dim=-1)
    return p @ v_full, p, k_full, v_full


def backward(q, k, p, k_full, v_full, d_o, mask, scale):
    """The hand-derived adjoint. Returns dQ, dK, dV and dM (None without a dense mask)."""
    batch, heads, _, head_dim = q.shape
    kv_heads, seq_kv = k.shape[1], k.shape[2]
    n_rep = heads // kv_heads
    d_v_full = p.transpose(-1, -2) @ d_o
    d_p = d_o @ v_full.transpose(-1, -2)
    d_s = p * (d_p - (d_p * p).sum(-1, keepdim=True))
    d_q = scale * d_s @ k_full
    d_k_full = scale * d_s.transpose(-1, -2) @ q
    # grouped-query attention: one K/V head serves n_rep query heads, so its gradient is their sum
    d_k = d_k_full.reshape(batch, kv_heads, n_rep, seq_kv, head_dim).sum(2)
    d_v = d_v_full.reshape(batch, kv_heads, n_rep, seq_kv, head_dim).sum(2)
    d_m = None
    if mask is not None:
        d_m = d_s
        if mask.shape[1] == kv_heads != heads:  # per K/V head: sum the query heads of each group
            d_m = d_m.reshape(batch, kv_heads, n_rep, *d_s.shape[2:]).sum(2)
        for axis, size in enumerate(mask.shape):  # broadcast axes of the mask collect a sum
            if size == 1 and d_m.shape[axis] != 1:
                d_m = d_m.sum(axis, keepdim=True)
    return d_q, d_k, d_v, d_m


def check(label, batch, heads, kv_heads, seq_q, seq_kv, head_dim, scale=0.0, causal=False, window=0,
          right_context=0, lengths=None, mask_shape=None, torch_reference=None):
    q = torch.randn(batch, heads, seq_q, head_dim, requires_grad=True)
    k = torch.randn(batch, kv_heads, seq_kv, head_dim, requires_grad=True)
    v = torch.randn(batch, kv_heads, seq_kv, head_dim, requires_grad=True)
    mask = torch.randn(*mask_shape, requires_grad=True) if mask_shape else None
    w = torch.randn(batch, heads, seq_q, head_dim)  # loss = sum(O * w), so dO = w
    s = 1.0 / math.sqrt(head_dim) if scale == 0.0 else scale
    structural = rules(batch, seq_q, seq_kv, causal, window, right_context, lengths)

    out, p, k_full, v_full = forward(q, k, v, mask, s, structural)
    inputs = [q, k, v] + ([mask] if mask is not None else [])
    auto = torch.autograd.grad((out * w).sum(), inputs)
    hand = backward(q, k, p, k_full, v_full, w, mask, s)
    worst = max((a - h).abs().max().item() for a, h in zip(auto, hand))

    forward_note = ""
    if torch_reference is not None:
        forward_note = f"   forward vs torch {(out - torch_reference(q, k, v)).abs().max().item():.1e}"
    padded = ""
    if lengths is not None:
        beyond = max(max(auto[1][b, :, n:].abs().max().item(), auto[2][b, :, n:].abs().max().item())
                     for b, n in enumerate(lengths) if n < seq_kv)
        padded = f"   gradient of padded keys {beyond:.1e}"
    print(f"{label:<44} max |hand - autograd| = {worst:.1e}{forward_note}{padded}")
    assert worst < 1e-12, label
    return auto, hand


torch.manual_seed(1385)
sdpa = F.scaled_dot_product_attention
check("plain", 2, 2, 2, 3, 3, 4, torch_reference=lambda q, k, v: sdpa(q, k, v))
check("custom scale 0.5", 1, 2, 2, 3, 3, 4, scale=0.5, torch_reference=lambda q, k, v: sdpa(q, k, v, scale=0.5))
check("causal", 1, 2, 2, 4, 4, 4, causal=True, torch_reference=lambda q, k, v: sdpa(q, k, v, is_causal=True))
check("causal decode step (1 query, 5 cached keys)", 1, 2, 2, 1, 5, 4, causal=True,
      torch_reference=lambda q, k, v: sdpa(q, k, v))  # the last query sees every key
check("key padding, lengths [4, 2]", 2, 2, 2, 3, 4, 4, lengths=[4, 2])
check("sliding window 2", 1, 2, 2, 5, 5, 4, window=2)
check("sliding window 2, right context 1", 1, 2, 2, 5, 5, 4, window=2, right_context=1)
check("causal + window 3 + lengths [5, 3]", 2, 2, 2, 5, 5, 4, causal=True, window=3, lengths=[5, 3])
check("dense mask [1, 1, seqQ, seqKV]", 2, 2, 2, 3, 4, 4, mask_shape=(1, 1, 3, 4))
check("dense mask [batch, heads, seqQ, seqKV]", 2, 2, 2, 3, 4, 4, mask_shape=(2, 2, 3, 4))
check("grouped query, 4 heads on 2 K/V heads", 1, 4, 2, 3, 3, 4,
      torch_reference=lambda q, k, v: sdpa(q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1)))
check("grouped query + causal + scale 0.5", 1, 4, 2, 3, 3, 4, scale=0.5, causal=True)
check("grouped query + dense mask per K/V head", 1, 4, 2, 3, 3, 4, mask_shape=(1, 2, 3, 3))

# A row whose keys are all masked (key length 0): every score becomes exactly MASKED in floating
# point, P is uniform, and the formulas still return the derivative of the definition.
auto, hand = check("fully masked row, lengths [3, 0]", 2, 1, 1, 2, 3, 4, lengths=[3, 0])
print(f"  fully masked row: |dQ| of that row = {auto[0][1].abs().max().item():.1e} (not zero, see the note)")
