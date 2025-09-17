import time
import torch
from kernel import groupnorm_fusion


@torch.jit.script
def torch_op(
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
    # taken straight out of albatross
    if xx.dim() == 3:
        # generation
        xx = torch.nn.functional.group_norm(xx.view(1,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps=eps).view(H*N)    
        xx = xx + ((r * k * r_k).view(H,N).sum(dim=-1, keepdim=True) * v.view(H,N)).view(H*N)
        return xx * g
    else:
        # prefill
        T = xx.shape[0]
        xx = torch.nn.functional.group_norm(xx.view(T,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps=eps).view(T,H*N)
        xx = xx + ((r * k * r_k).view(T,H,N).sum(dim=-1, keepdim=True) * v.view(T,H,N)).view(T,H*N)
        return xx * g


def make_inputs(H, N, T, dtype, device):
    C = H * N
    if T == 1:
        # generation
        xx = torch.randn(H, N, T, dtype=dtype, device=device)
        r = torch.randn(C, dtype=dtype, device=device)
        k = torch.randn(C, dtype=dtype, device=device)
        v = torch.randn(C, dtype=dtype, device=device)
        g = torch.randn(C, dtype=dtype, device=device)
    else:
        # prefill
        xx = torch.randn(T, C, dtype=dtype, device=device)
        r = torch.randn(T, C, dtype=dtype, device=device)
        k = torch.randn(T, C, dtype=dtype, device=device)
        v = torch.randn(T, C, dtype=dtype, device=device)
        g = torch.randn(T, C, dtype=dtype, device=device)

    ln_w = torch.randn(C, dtype=dtype, device=device)
    ln_b = torch.randn(C, dtype=dtype, device=device)
    r_k = torch.randn(C, dtype=dtype, device=device)

    return xx, ln_w, ln_b, r, k, r_k, v, g, H, N


def benchmark(H=8, N=64, T=1024, dtype=torch.float32, device="cuda", warmup=10, iters=100):
    inputs = make_inputs(H, N, T, dtype, device)

    # ============ warmup ============
    for _ in range(warmup):
        _ = groupnorm_fusion(*inputs)
        _ = torch_op(*inputs)
    torch.cuda.synchronize()

    # === Triton version timing ===
    t0 = time.time()
    for _ in range(iters):
        _ = groupnorm_fusion(*inputs)
    torch.cuda.synchronize()
    t_triton = (time.time() - t0) / iters * 1000

    # === PyTorch version timing ===
    t0 = time.time()
    for _ in range(iters):
        _ = torch_op(*inputs)
    torch.cuda.synchronize()
    t_torch = (time.time() - t0) / iters * 1000

    print(f"Input size: T={T}, H={H}, N={N}, C={H*N}")
    print(f"PyTorch  : {t_torch:.3f} ms/iter")
    print(f"Triton   : {t_triton:.3f} ms/iter  {t_torch / t_triton:.2f}x")

def validate_accuracy(H=8, N=64, T=1024, dtype=torch.float32, device="cuda"):
    inputs = make_inputs(H, N, T, dtype, device)

    output_triton = groupnorm_fusion(*inputs)
    output_torch = torch_op(*inputs)

    abs_diff = (output_triton - output_torch).abs()
    max_abs_err = abs_diff.max().item()
    mean_abs_err = abs_diff.mean().item()

    rel_diff = abs_diff / (output_torch.abs() + 1e-8)
    max_rel_err = rel_diff.max().item()
    mean_rel_err = rel_diff.mean().item()

    print(f"[GroupNorm+Output] max_abs_err={max_abs_err:.2e} mean_abs_err={mean_abs_err:.2e} max_rel_err={max_rel_err:.2e} mean_rel_err={mean_rel_err:.2e}")

if __name__ == "__main__":
    DTYPES = [torch.float16, torch.bfloat16, torch.float32]

    for DTYPE in DTYPES:
        print(f'########################## {DTYPE} ##########################')
        print(f"============= GROUPNORM OUTPUT FUSION BENCHMARK =============")

        print("--- Generation (T=1) - New 3D format [H, N, T]:")
        for H in [8, 16, 32, 40]:
            print(f"--- H={H}, N=64 (3D):")
            benchmark(H=H, N=64, T=1, dtype=DTYPE)
            print()

        print("--- Prefill (various T):")
        for T in [64, 256, 1024, 4096]:
            print(f"--- T={T}, H=40, N=64:")
            benchmark(H=40, N=64, T=T, dtype=DTYPE)
            print()

        print(f"\n=========== GROUPNORM OUTPUT FUSION CORRECTNESS CHECK =========")
        print("Testing generation (T=1) - New 3D:")
        validate_accuracy(H=8, N=64, T=1, dtype=DTYPE)

        print("Testing prefill (T=1024):")
        validate_accuracy(H=8, N=64, T=1024, dtype=DTYPE)
