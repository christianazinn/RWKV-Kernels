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