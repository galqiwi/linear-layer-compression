import os
from typing import Optional

import torch
from torch.utils.cpp_extension import load

CUDA_FOLDER = os.path.dirname(os.path.abspath(__file__))
CUDA_KERNEL = load(
    name="higgs_cuda",
    sources=[os.path.join(CUDA_FOLDER, "higgs.cpp"), os.path.join(CUDA_FOLDER, "higgs.cu")],
    extra_cflags=["-g"],
    extra_cuda_cflags=["-g", "-G", "-Xcompiler", "-O0", "-Xptxas -O0", "-lineinfo", "-O0"],
)

torch.library.define(
    "higgs::higgs2x256_matmat", "(Tensor input, Tensor codes, Tensor scales, Tensor bias) -> Tensor"
)

torch.library.impl("higgs::higgs2x256_matmat", "default", CUDA_KERNEL.higgs2x256_matmat)


@torch.library.impl_abstract("higgs::higgs2x256_matmat")
def higgs2x256_matmat_meta(input, codes, scales, bias):
    return torch.empty(input.shape[:-1] + (codes.shape[0],), device=input.device, dtype=input.dtype)
