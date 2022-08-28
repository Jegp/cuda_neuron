from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='cuda_neuron',
      ext_modules=[cpp_extension.CUDAExtension('li_cpp', ['li_cuda.cpp', 'li_cuda_kernel.cu'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})