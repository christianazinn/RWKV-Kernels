#include <torch/extension.h>
#include <vector>

void token_mix_cuda(
    torch::Tensor x, torch::Tensor x_prev,
    torch::Tensor x_r, torch::Tensor x_w, torch::Tensor x_k,
    torch::Tensor x_v, torch::Tensor x_a, torch::Tensor x_g,
    torch::Tensor out_xr, torch::Tensor out_xw, torch::Tensor out_xk,
    torch::Tensor out_xv, torch::Tensor out_xa, torch::Tensor out_xg
);

std::vector<torch::Tensor> token_mix(
    torch::Tensor x,
    torch::Tensor x_prev,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g
) {
    auto T = x.size(0);
    auto D = x.size(1);
    
    auto opts = x.options();
    auto out_xr = torch::empty({T, D}, opts);
    auto out_xw = torch::empty({T, D}, opts);
    auto out_xk = torch::empty({T, D}, opts);
    auto out_xv = torch::empty({T, D}, opts);
    auto out_xa = torch::empty({T, D}, opts);
    auto out_xg = torch::empty({T, D}, opts);
    
    token_mix_cuda(x, x_prev, x_r, x_w, x_k, x_v, x_a, x_g,
                   out_xr, out_xw, out_xk, out_xv, out_xa, out_xg);
    
    return {out_xr, out_xw, out_xk, out_xv, out_xa, out_xg};
}

void token_mix_inplace(
    torch::Tensor x,
    torch::Tensor x_prev,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g,
    torch::Tensor out_xr,
    torch::Tensor out_xw,
    torch::Tensor out_xk,
    torch::Tensor out_xv,
    torch::Tensor out_xa,
    torch::Tensor out_xg
) {
    token_mix_cuda(x, x_prev, x_r, x_w, x_k, x_v, x_a, x_g,
                   out_xr, out_xw, out_xk, out_xv, out_xa, out_xg);
}

// Update PYBIND11_MODULE to add the new binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("token_mix", &token_mix, "token mixing forward bf16");
    m.def("token_mix_inplace", &token_mix_inplace, "token mixing forward bf16 (inplace outputs)");
}