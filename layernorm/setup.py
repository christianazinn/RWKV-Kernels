from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='layer_norm_cuda',
    ext_modules=[
        CUDAExtension('layer_norm_cuda', [
            'layernorm.cpp',
            'layernorm_kernel.cu',
        ])
    ],
    cmdclass={'build_ext': BuildExtension}
)

# python setup.py build_ext --inplace; python ctest.py