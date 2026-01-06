import torch
import layer_norm_cuda

from benchmark import benchmark


def layer_norm_ref(x, w, b, eps=1e-5):
    return torch.nn.functional.layer_norm(x, (x.size(-1),), w, b, eps)


layer_norm_ref_compiled = torch.compile(layer_norm_ref)


if __name__ == "__main__":
    D = 2048

    benchmark(
        "LayerNorm",
        layer_norm_ref_compiled,
        layer_norm_cuda.layer_norm_inplace,
        in_shapes={"x": lambda T: (T, D), "w": lambda T: (D,), "b": lambda T: (D,)},
        in_kwargs={"eps": 1e-5},
        out_shapes={"out": lambda T: (T, D)},
        atol=1e-2,
        rtol=1e-2,
    )
