#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>

#include "../azinn.cuh"

__global__ void layer_norm_kernel_bf16(
    const bf16x8* __restrict__ x,
    const bf16x8* __restrict__ w,
    const bf16x8* __restrict__ b,
    bf16x8* __restrict__ out,
    int T, int D8,
    float eps
) {
    int t = blockIdx.x;
    if (t >= T) return;
    
    int tid = threadIdx.x;
    
    __shared__ float smem[32];
    __shared__ float s_mean, s_inv_std;
    
    float vals[8];
    if (tid < D8) {
        bf16x8 xv = x[t * D8 + tid];
        float2 fa = __bfloat1622float2(xv.a);
        float2 fb = __bfloat1622float2(xv.b);
        float2 fc = __bfloat1622float2(xv.c);
        float2 fd = __bfloat1622float2(xv.d);
        vals[0] = fa.x; vals[1] = fa.y;
        vals[2] = fb.x; vals[3] = fb.y;
        vals[4] = fc.x; vals[5] = fc.y;
        vals[6] = fd.x; vals[7] = fd.y;
    } else {
        #pragma unroll
        for (int i = 0; i < 8; i++) vals[i] = 0.0f;
    }
    
    // welford's online
    float local_sum = vals[0] + vals[1] + vals[2] + vals[3] + 
                      vals[4] + vals[5] + vals[6] + vals[7];
    float local_sum_sq = vals[0]*vals[0] + vals[1]*vals[1] + vals[2]*vals[2] + vals[3]*vals[3] +
                         vals[4]*vals[4] + vals[5]*vals[5] + vals[6]*vals[6] + vals[7]*vals[7];
    
    float sum = block_reduce_sum(local_sum, smem);
    __syncthreads();
    float sum_sq = block_reduce_sum(local_sum_sq, smem);
    
    // mean, inv_std
    if (tid == 0) {
        float rcp_n = 1.0f / (D8 * 8.0f);
        float mean = sum * rcp_n;
        float var = sum_sq * rcp_n - mean * mean;
        s_mean = mean;
        s_inv_std = rsqrtf(var + eps);
    }
    __syncthreads();
    
    float mean = s_mean;
    float inv_std = s_inv_std;
    
    if (tid < D8) {
        bf16x8 wv = w[tid];
        bf16x8 bv = b[tid];
        
        float2 wa = __bfloat1622float2(wv.a), ba = __bfloat1622float2(bv.a);
        float2 wb = __bfloat1622float2(wv.b), bb = __bfloat1622float2(bv.b);
        float2 wc = __bfloat1622float2(wv.c), bc = __bfloat1622float2(bv.c);
        float2 wd = __bfloat1622float2(wv.d), bd = __bfloat1622float2(bv.d);
        
        float n0 = (vals[0] - mean) * inv_std;
        float n1 = (vals[1] - mean) * inv_std;
        float n2 = (vals[2] - mean) * inv_std;
        float n3 = (vals[3] - mean) * inv_std;
        float n4 = (vals[4] - mean) * inv_std;
        float n5 = (vals[5] - mean) * inv_std;
        float n6 = (vals[6] - mean) * inv_std;
        float n7 = (vals[7] - mean) * inv_std;
        
        bf16x8 result;
        result.a = __float22bfloat162_rn(make_float2(n0 * wa.x + ba.x, n1 * wa.y + ba.y));
        result.b = __float22bfloat162_rn(make_float2(n2 * wb.x + bb.x, n3 * wb.y + bb.y));
        result.c = __float22bfloat162_rn(make_float2(n4 * wc.x + bc.x, n5 * wc.y + bc.y));
        result.d = __float22bfloat162_rn(make_float2(n6 * wd.x + bd.x, n7 * wd.y + bd.y));
        
        out[t * D8 + tid] = result;
    }
}

void layer_norm_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    torch::Tensor out,
    float eps
) {
    int T = x.size(0);
    int D = x.size(1);
    int D8 = D / 8;
    
    dim3 grid(T);
    dim3 block(D8);
    
    layer_norm_kernel_bf16<<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const bf16x8*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bf16x8*>(w.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bf16x8*>(b.data_ptr<at::BFloat16>()),
        reinterpret_cast<bf16x8*>(out.data_ptr<at::BFloat16>()),
        T, D8, eps
    );
}