#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "../azinn.cuh"

#define BLOCK_T 64

__global__ void token_mix_kernel_bf16(
    const bf16x8* __restrict__ x,
    bf16x8* __restrict__ x_prev,
    const bf16x8* __restrict__ x_r,
    const bf16x8* __restrict__ x_w,
    const bf16x8* __restrict__ x_k,
    const bf16x8* __restrict__ x_v,
    const bf16x8* __restrict__ x_a,
    const bf16x8* __restrict__ x_g,
    bf16x8* __restrict__ out_xr,
    bf16x8* __restrict__ out_xw,
    bf16x8* __restrict__ out_xk,
    bf16x8* __restrict__ out_xv,
    bf16x8* __restrict__ out_xa,
    bf16x8* __restrict__ out_xg,
    int T, int D8
) {
    int t_block = blockIdx.x;
    int d8 = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (d8 >= D8) return;
    
    int t_start = t_block * BLOCK_T;
    int t_end = min(t_start + BLOCK_T, T);
    
    bf16x8 r = x_r[d8];
    bf16x8 w = x_w[d8];
    bf16x8 k = x_k[d8];
    bf16x8 v = x_v[d8];
    bf16x8 a = x_a[d8];
    bf16x8 g = x_g[d8];
    
    bf16x8 x_shifted;
    if (t_start == 0) {
        x_shifted = x_prev[d8];
    } else {
        x_shifted = x[(t_start - 1) * D8 + d8];
    }
    
    for (int t = t_start; t < t_end; t++) {
        int idx = t * D8 + d8;
        bf16x8 x_curr = x[idx];
        
        bf16x8 xx = bf16x8_sub(x_shifted, x_curr);
        
        out_xr[idx] = bf16x8_fma(x_curr, xx, r);
        out_xw[idx] = bf16x8_fma(x_curr, xx, w);
        out_xk[idx] = bf16x8_fma(x_curr, xx, k);
        out_xv[idx] = bf16x8_fma(x_curr, xx, v);
        out_xa[idx] = bf16x8_fma(x_curr, xx, a);
        out_xg[idx] = bf16x8_fma(x_curr, xx, g);
        
        x_shifted = x_curr;
    }
    
    // update x_prev inplace
    int last_t_block = (T - 1) / BLOCK_T;
    if (t_block == last_t_block) {
        x_prev[d8] = x_shifted;
    }
}

void token_mix_cuda(
    torch::Tensor x, torch::Tensor x_prev,
    torch::Tensor x_r, torch::Tensor x_w, torch::Tensor x_k,
    torch::Tensor x_v, torch::Tensor x_a, torch::Tensor x_g,
    torch::Tensor out_xr, torch::Tensor out_xw, torch::Tensor out_xk,
    torch::Tensor out_xv, torch::Tensor out_xa, torch::Tensor out_xg
) {
    int T = x.size(0);
    int D = x.size(1);
    int threads = 256;
    
    int D8 = D / 8;
    dim3 grid(div_up(T, BLOCK_T), div_up(D8, threads));
    dim3 block(threads);
    
    token_mix_kernel_bf16<<<grid, block>>>(
        reinterpret_cast<const bf16x8*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<bf16x8*>(x_prev.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bf16x8*>(x_r.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bf16x8*>(x_w.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bf16x8*>(x_k.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bf16x8*>(x_v.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bf16x8*>(x_a.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bf16x8*>(x_g.data_ptr<at::BFloat16>()),
        reinterpret_cast<bf16x8*>(out_xr.data_ptr<at::BFloat16>()),
        reinterpret_cast<bf16x8*>(out_xw.data_ptr<at::BFloat16>()),
        reinterpret_cast<bf16x8*>(out_xk.data_ptr<at::BFloat16>()),
        reinterpret_cast<bf16x8*>(out_xv.data_ptr<at::BFloat16>()),
        reinterpret_cast<bf16x8*>(out_xa.data_ptr<at::BFloat16>()),
        reinterpret_cast<bf16x8*>(out_xg.data_ptr<at::BFloat16>()),
        T, D8
    );
}
