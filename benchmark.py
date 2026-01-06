import torch
import time


def timed(fn, warmup=10, iters=100):
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


def benchmark(name, compiled_fn, cuda_fn,
              in_shapes: dict, in_kwargs: dict, out_shapes: dict,
              T_vals = [1, 64, 256, 1024, 4096, 16384, 32768, 65536, 131072],
              atol=1e-2, rtol=1e-2, seed=42):
    print("=" * 70)
    print(f"{name} Kernel Benchmark: CUDA vs PyTorch (bfloat16)")
    print("=" * 70)
    torch.manual_seed(seed)
    for T in T_vals:
        torch_ins = [
            torch.randn(shape_t(T), device='cuda', dtype=torch.bfloat16) for shape_t in in_shapes.values()
        ]
        cuda_ins = [x.clone() for x in torch_ins]
        cuda_outs = [
            torch.empty(shape_t(T), device='cuda', dtype=torch.bfloat16) for shape_t in out_shapes.values()
        ]

        # ---------------------- correctness check ----------------------
        ref_outs = compiled_fn(*torch_ins, **in_kwargs)
        if type(ref_outs) not in (list, tuple):
            ref_outs = (ref_outs,)
        cuda_fn(*cuda_ins, *cuda_outs, **in_kwargs)
        for i, (ref, out) in enumerate(zip(ref_outs, cuda_outs)):
            if not torch.allclose(ref, out, atol=atol, rtol=rtol):
                max_err = (ref - out).abs().max().item()
                print(f"[WARNING] Output {i} at T={T}: max error {max_err}")

        # ---------------------- capture CUDA graphs ----------------------
        # torch
        for _ in range(3):
            compiled_fn(*torch_ins, **in_kwargs)
        torch.cuda.synchronize()

        g_torch = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_torch):
            torch_outs = compiled_fn(*torch_ins, **in_kwargs)
        torch_ms = timed(g_torch.replay)

        # cuda
        stream = torch.cuda.Stream()
        cuda_outs_static = [torch.empty(shape_t(T), device='cuda', dtype=torch.bfloat16) for shape_t in out_shapes.values()]

        with torch.cuda.stream(stream):
            for _ in range(3):
                cuda_fn(*cuda_ins, *cuda_outs, **in_kwargs)
            stream.synchronize()

            g_cuda = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_cuda, stream=stream):
                cuda_fn(*cuda_ins, *cuda_outs_static, **in_kwargs)
            cuda_ms = timed(g_cuda.replay)
        
        speedup = torch_ms / cuda_ms
        print(f"T={T:5d}: PyTorch={torch_ms:.3f}ms, CUDA={cuda_ms:.3f}ms, Speedup={speedup:.2f}x")
    print("=" * 70)
