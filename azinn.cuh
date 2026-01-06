// pack for 128 bit contiguous memory access
// i can't believe reinterpret_cast 
struct bf16x8 {
    __nv_bfloat162 a, b, c, d;
};

__device__ __forceinline__ bf16x8 bf16x8_sub(bf16x8 x, bf16x8 y) {
    bf16x8 r;
    r.a = x.a - y.a;
    r.b = x.b - y.b;
    r.c = x.c - y.c;
    r.d = x.d - y.d;
    return r;
}

__device__ __forceinline__ bf16x8 bf16x8_fma(bf16x8 x, bf16x8 y, bf16x8 z) {
    bf16x8 r;
    // so for some reason using __hfma2 here causes incorrect results???
    // i think this is probably due to floating point associativity issues
    r.a = x.a + y.a * z.a;
    r.b = x.b + y.b * z.b;
    r.c = x.c + y.c * z.c;
    r.d = x.d + y.d * z.d;
    return r;
}

__host__ __device__ constexpr int div_up(int a, int b) {
    return (a + b - 1) / b;
}

// warp-level reduction
template<typename T> __device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// block-level reduction in smem
template<typename T> __device__ __forceinline__ T block_reduce_sum(T val, T* smem) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warp_reduce_sum(val);
    
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    
    // reduce across warps
    val = (threadIdx.x < blockDim.x / 32) ? smem[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    
    return val;
}