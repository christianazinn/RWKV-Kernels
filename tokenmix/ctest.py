import torch
import token_mix_cuda

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
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    times.sort()
    return times[len(times) // 2], sum(times) / len(times)


def run_benchmark():
    print("=" * 70)
    print(f"Token Mix Kernel Benchmark: CUDA vs PyTorch")
    print("=" * 70)
    
    atol = 1e-5
    
    for T in [1, 64, 256, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]:
        D = 1024
        
        x = torch.randn(T, D, device='cuda', dtype=torch.bfloat16)
        x_prev = torch.randn(1, D, device='cuda', dtype=torch.bfloat16)
        x_r = torch.randn(D, device='cuda', dtype=torch.bfloat16)
        x_w = torch.randn(D, device='cuda', dtype=torch.bfloat16)
        x_k = torch.randn(D, device='cuda', dtype=torch.bfloat16)
        x_v = torch.randn(D, device='cuda', dtype=torch.bfloat16)
        x_a = torch.randn(D, device='cuda', dtype=torch.bfloat16)
        x_g = torch.randn(D, device='cuda', dtype=torch.bfloat16)
        
        # ---------------------- correctness check ----------------------
        x_prev_ref = x_prev.clone()
        x_prev_cuda = x_prev.clone()
        
        ref_outs = token_mix_ref(x, x_prev_ref, x_r, x_w, x_k, x_v, x_a, x_g)
        cuda_outs = token_mix_cuda.token_mix(x, x_prev_cuda, x_r, x_w, x_k, x_v, x_a, x_g)
        
        for i, (ref, cuda) in enumerate(zip(ref_outs, cuda_outs)):
            if not torch.allclose(ref, cuda, atol=atol, rtol=1e-5):
                max_err = (ref - cuda).abs().max().item()
                print(f"WARNING: Mismatch at output {i}, max error: {max_err}")
        
        if not torch.allclose(x_prev_ref, x_prev_cuda, atol=atol, rtol=1e-2):
            max_err = (x_prev_ref - x_prev_cuda).abs().max().item()
            print(f"WARNING: x_prev mismatch, max error: {max_err}")
        
        # ---------------------- benchmark ----------------------
        def run_torch():
            xp = x_prev.clone()
            return token_mix_ref_compiled(x, xp, x_r, x_w, x_k, x_v, x_a, x_g)
        
        def run_cuda():
            xp = x_prev.clone()
            return token_mix_cuda.token_mix(x, xp, x_r, x_w, x_k, x_v, x_a, x_g)
        
        torch_ms, torch_avg_ms = benchmark(run_torch)
        cuda_ms, cuda_avg_ms = benchmark(run_cuda)
        speedup = torch_ms / cuda_ms
        speedup_avg = torch_avg_ms / cuda_avg_ms
        
        print(f"T={T:5d}, D={D}: PyTorch={torch_ms:.3f}ms, CUDA={cuda_ms:.3f}ms, Speedup={speedup:.2f}x")
        print(f"                Avg PyTorch={torch_avg_ms:.3f}ms, Avg CUDA={cuda_avg_ms:.3f}ms, Avg Speedup={speedup_avg:.2f}x")
    
    print()


if __name__ == "__main__":
    torch.manual_seed(42)
    run_benchmark()
