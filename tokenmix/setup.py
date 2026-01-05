from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='token_mix_cuda',
    ext_modules=[
        CUDAExtension('token_mix_cuda', [
            'tokenmix.cpp',
            'tokenmix_kernel.cu',
        ])
    ],
    cmdclass={'build_ext': BuildExtension}
)

# python setup.py build_ext --inplace; python ctest.py