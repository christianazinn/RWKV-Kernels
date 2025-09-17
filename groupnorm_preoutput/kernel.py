import torch
import triton
import triton.language as tl


@triton.jit
def groupnorm_fusion_kernel(
    xx_ptr,
    ln_w_ptr,
    ln_b_ptr,
    r_ptr,
    k_ptr,
    r_k_ptr,
    v_ptr,
    g_ptr,
    output_ptr,
    T,
    H,
    N,
    eps: tl.constexpr,
    N_size: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    t_id = pid // H
    group_id = pid % H

    if t_id >= T:
        return

    base_offset = t_id * H * N + group_id * N
    group_base = group_id * N

    n_offs = tl.arange(0, N_size)
    
    xx_offs = base_offset + n_offs
    xx = tl.load(xx_ptr + xx_offs)

    nn_offs = group_base + n_offs
    ln_w = tl.load(ln_w_ptr + nn_offs)
    ln_b = tl.load(ln_b_ptr + nn_offs)

    # group norm
    xx_float = xx.to(tl.float32)
    xx_sum = tl.sum(xx_float, axis=0)
    mean = xx_sum / N

    # group variance
    xx_centered = xx_float - mean
    var_sum = tl.sum(xx_centered * xx_centered, axis=0)
    var = var_sum / N

    # normalize
    inv_std = 1.0 / tl.sqrt(var + eps)
    xx_norm = xx_centered * inv_std
    xx_norm = xx_norm * ln_w.to(tl.float32) + ln_b.to(tl.float32)

    r = tl.load(r_ptr + xx_offs)
    k = tl.load(k_ptr + xx_offs)
    v = tl.load(v_ptr + xx_offs)
    r_k = tl.load(r_k_ptr + nn_offs)

    # (r * k * r_k).sum(-1, keepdim=True) * v
    rkv_prod = r.to(tl.float32) * k.to(tl.float32) * r_k.to(tl.float32)
    rkv_sum = tl.sum(rkv_prod, axis=0)
    rkv_contribution = rkv_sum * v.to(tl.float32)

    xx_final = xx_norm + rkv_contribution

    g = tl.load(g_ptr + xx_offs)
    output = xx_final * g.to(tl.float32)

    tl.store(output_ptr + xx_offs, output.to(xx.dtype))


def groupnorm_fusion(
    xx: torch.Tensor,
    ln_w: torch.Tensor,
    ln_b: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    r_k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    H: int,
    N: int,
    eps: float = 64e-5
):
    """replaces everything after RWKV7_OP except matmul @ O_"""

    C = H * N
    if xx.dim() == 3:
        # generation: xx is [H, N, 1]
        assert xx.shape == (H, N, 1), f"Expected [H, N, 1] for generation, got {xx.shape}"
        T = 1
        xx = xx.view(T, C) 
        squeeze_output = True
    else:
        # prefill: xx is [T, C]
        T = xx.shape[0]
        squeeze_output = False

    assert xx.shape == (T, C)
    assert ln_w.shape == (C,)
    assert ln_b.shape == (C,)
    assert r_k.shape == (C,)

    output = torch.empty_like(xx)

    N_size = triton.next_power_of_2(N)
    num_blocks = T * H

    grid = (num_blocks,)

    groupnorm_fusion_kernel[grid](
        xx, ln_w, ln_b, r, k, r_k, v, g, output,
        T, H, N, eps, N_size
    )

    if squeeze_output:
        output = output.squeeze(0)

    return output
