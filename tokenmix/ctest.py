import torch
import token_mix_cuda
import time


def token_mix_ref(x, x_prev, x_r, x_w, x_k, x_v, x_a, x_g):
    xx = torch.cat((x_prev[0].unsqueeze(0), x[:-1, :])) - x
    x_prev[0] = x[-1, :]
    return (
        x + xx * x_r,
        x + xx * x_w,
        x + xx * x_k,
        x + xx * x_v,
        x + xx * x_a,
        x + xx * x_g,
    )


token_mix_ref_compiled = torch.compile(token_mix_ref)


def benchmark(fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        for _ in range(iters):
            result = fn()
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    took = torch.median(torch.tensor(times)) / iters

    return took.item(), 0


def run_benchmark():
    print("=" * 70)
    print(f"Token Mix Kernel Benchmark: CUDA vs PyTorch (with CUDAGraph)")
    print("=" * 70)

    atol = 1e-5

    for T in [1, 64, 256, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]:
        D = 1024

        x = torch.randn(T, D, device="cuda", dtype=torch.bfloat16)
        x_prev = torch.randn(1, D, device="cuda", dtype=torch.bfloat16)
        x_r = torch.randn(D, device="cuda", dtype=torch.bfloat16)
        x_w = torch.randn(D, device="cuda", dtype=torch.bfloat16)
        x_k = torch.randn(D, device="cuda", dtype=torch.bfloat16)
        x_v = torch.randn(D, device="cuda", dtype=torch.bfloat16)
        x_a = torch.randn(D, device="cuda", dtype=torch.bfloat16)
        x_g = torch.randn(D, device="cuda", dtype=torch.bfloat16)

        # ---------------------- correctness check ----------------------
        x_prev_ref = x_prev.clone()
        x_prev_cuda = x_prev.clone()

        ref_outs = token_mix_ref(x, x_prev_ref, x_r, x_w, x_k, x_v, x_a, x_g)
        cuda_outs = token_mix_cuda.token_mix(
            x, x_prev_cuda, x_r, x_w, x_k, x_v, x_a, x_g
        )

        for i, (ref, cuda) in enumerate(zip(ref_outs, cuda_outs)):
            if not torch.allclose(ref, cuda, atol=atol, rtol=1e-5):
                max_err = (ref - cuda).abs().max().item()
                print(f"WARNING: Mismatch at output {i}, max error: {max_err}")

        if not torch.allclose(x_prev_ref, x_prev_cuda, atol=atol, rtol=1e-2):
            max_err = (x_prev_ref - x_prev_cuda).abs().max().item()
            print(f"WARNING: x_prev mismatch, max error: {max_err}")

        # ---------------------- capture CUDA graphs ----------------------
        x_prev_torch_static = x_prev.clone()
        x_prev_cuda_static = x_prev.clone()

        for _ in range(3):
            token_mix_ref_compiled(x, x_prev_torch_static, x_r, x_w, x_k, x_v, x_a, x_g)
            token_mix_cuda.token_mix(
                x, x_prev_cuda_static, x_r, x_w, x_k, x_v, x_a, x_g
            )
        torch.cuda.synchronize()

        g_torch = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_torch):
            torch_outs = token_mix_ref_compiled(
                x, x_prev_torch_static, x_r, x_w, x_k, x_v, x_a, x_g
            )

        out_xr = torch.empty_like(x)
        out_xw = torch.empty_like(x)
        out_xk = torch.empty_like(x)
        out_xv = torch.empty_like(x)
        out_xa = torch.empty_like(x)
        out_xg = torch.empty_like(x)

        for _ in range(3):
            token_mix_cuda.token_mix_inplace(
                x,
                x_prev_cuda_static,
                x_r,
                x_w,
                x_k,
                x_v,
                x_a,
                x_g,
                out_xr,
                out_xw,
                out_xk,
                out_xv,
                out_xa,
                out_xg,
            )
        torch.cuda.synchronize()

        stream = torch.cuda.Stream()

        with torch.cuda.stream(stream):
            for _ in range(3):
                token_mix_cuda.token_mix_inplace(
                    x,
                    x_prev_cuda_static,
                    x_r,
                    x_w,
                    x_k,
                    x_v,
                    x_a,
                    x_g,
                    out_xr,
                    out_xw,
                    out_xk,
                    out_xv,
                    out_xa,
                    out_xg,
                )
            stream.synchronize()

            g_cuda = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_cuda, stream=stream):
                token_mix_cuda.token_mix_inplace(
                    x,
                    x_prev_cuda_static,
                    x_r,
                    x_w,
                    x_k,
                    x_v,
                    x_a,
                    x_g,
                    out_xr,
                    out_xw,
                    out_xk,
                    out_xv,
                    out_xa,
                    out_xg,
                )

            cuda_ms, cuda_avg_ms = benchmark(g_cuda.replay)

        # ---------------------- benchmark ----------------------
        torch_ms, torch_avg_ms = benchmark(g_torch.replay)
        # cuda_ms, cuda_avg_ms = benchmark(run_cuda_no_graph)
        speedup = torch_ms / cuda_ms

        print(
            f"T={T:6d}: PyTorch={torch_ms:.4f}ms, CUDA={cuda_ms:.4f}ms, Speedup={speedup:.2f}x"
        )

    print()


if __name__ == "__main__":
    torch.manual_seed(42)
    run_benchmark()
