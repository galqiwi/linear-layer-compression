import os
from typing import Optional

import torch
from torch.utils.cpp_extension import load

CUDA_FOLDER = os.path.dirname(os.path.abspath(__file__))
CUDA_KERNEL = load(
    name="higgs_cuda",
    sources=[os.path.join(CUDA_FOLDER, "higgs.cpp"), os.path.join(CUDA_FOLDER, "higgs.cu")],
)

torch.library.define(
    "higgs::higgs_matmat", "(Tensor input, Tensor codes, Tensor scales, Tensor bias) -> Tensor"
)

torch.library.impl("higgs::higgs_matmat", "default", CUDA_KERNEL.higgs_matmat)


@torch.library.impl_abstract("higgs::higgs_matmat")
def higgs_matmat_meta(input, codes, scales, bias):
    return torch.empty(input.shape[:-1] + (codes.shape[0],), device=input.device, dtype=input.dtype)
