#include <torch/extension.h>

void layer_norm_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    torch::Tensor out,
    float eps
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layer_norm_inplace", &layer_norm_cuda, "layer norm forward bf16 inplace",
          py::arg("x"), py::arg("w"), py::arg("b"), py::arg("out"), py::arg("eps") = 1e-5f);
}