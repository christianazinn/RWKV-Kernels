import torch
import layer_norm_cuda
import time


def layer_norm_ref(x, w, b, eps=1e-5):
    return torch.nn.functional.layer_norm(x, (x.size(-1),), w, b, eps)


layer_norm_ref_compiled = torch.compile(layer_norm_ref)


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

    return took.item()


if __name__ == "__main__":
    torch.manual_seed(42)
    
    print("=" * 70)
    print("LayerNorm Kernel Benchmark: CUDA vs PyTorch (bfloat16)")
    print("=" * 70)
    
    eps = 1e-5
    atol = 1e-2
    rtol = 1e-2
    
    for T in [1, 64, 256, 1024, 4096, 16384, 32768, 65536, 131072]:
        D = 2048
        
        x = torch.randn(T, D, device='cuda', dtype=torch.bfloat16)
        w = torch.randn(D, device='cuda', dtype=torch.bfloat16)
        b = torch.randn(D, device='cuda', dtype=torch.bfloat16)
        out = torch.empty_like(x)
        
        # ---------------------- correctness check ----------------------
        ref = layer_norm_ref(x, w, b, eps)
        layer_norm_cuda.layer_norm_inplace(x, w, b, out, eps)
        ref_compiled = layer_norm_ref_compiled(x, w, b, eps)

        if not torch.allclose(ref, out, atol=atol, rtol=rtol):
            max_err = (ref - out).abs().max().item()
            print(f"WARNING T={T}: max error {max_err}")

            diff = (out - ref).abs()
            idx = diff.argmax()
            row, col = idx // D, idx % D
            print(f"Max diff at [{row}, {col}]: cuda={out[row,col]}, torch={ref[row,col]}")
        
        # ---------------------- capture CUDA graphs ----------------------
        for _ in range(3):
            layer_norm_ref_compiled(x, w, b, eps)
        torch.cuda.synchronize()

        g_torch = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_torch):
            torch_outs = layer_norm_ref_compiled(x, w, b, eps)

        stream = torch.cuda.Stream()

        out_x = torch.empty_like(x)

        with torch.cuda.stream(stream):
            for _ in range(3):
                layer_norm_cuda.layer_norm_inplace(x, w, b, out, eps)
            stream.synchronize()

            g_cuda = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_cuda, stream=stream):
                layer_norm_cuda.layer_norm_inplace(x, w, b, out, eps)
            cuda_ms = benchmark(g_cuda.replay)

        torch_ms = benchmark(g_torch.replay)
        speedup = torch_ms / cuda_ms
        
        print(f"T={T:5d}, D={D}: PyTorch={torch_ms:.3f}ms, CUDA={cuda_ms:.3f}ms, Speedup={speedup:.2f}x")
    
    print("=" * 70)