import torch
import token_mix_cuda

from benchmark import benchmark


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


if __name__ == "__main__":
    D = 1024

    benchmark(
        "TokenMix",
        token_mix_ref_compiled,
        token_mix_cuda.token_mix_inplace,
        in_shapes={
            "x": lambda T: (T, D),
            "x_prev": lambda T: (1, D),
            "x_r": lambda T: (D,),
            "x_w": lambda T: (D,),
            "x_k": lambda T: (D,),
            "x_v": lambda T: (D,),
            "x_a": lambda T: (D,),
            "x_g": lambda T: (D,),
        },
        in_kwargs={},
        out_shapes={
            "out_xr": lambda T: (T, D),
            "out_xw": lambda T: (T, D),
            "out_xk": lambda T: (T, D),
            "out_xv": lambda T: (T, D),
            "out_xa": lambda T: (T, D),
            "out_xg": lambda T: (T, D),
        },
        atol=1e-2,
        rtol=1e-2,
    )
