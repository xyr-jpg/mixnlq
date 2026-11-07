"""Build the optional native CUDA Decode operator in-place."""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="mixnlq_cuda_ops",
    ext_modules=[
        CUDAExtension(
            name="mixnlq_cuda_ops",
            sources=[
                "csrc/mixnlq_cuda.cpp",
                "csrc/mixnlq_decode_cuda.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
