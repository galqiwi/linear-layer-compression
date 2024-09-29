import math

import torch
from torch import nn
import torch.nn.functional as F

from fast_hadamard_transform import hadamard_transform

from aqlm.utils import _dequantize_weight

from higgs import CUDA_FOLDER


def pad_to_block(tensor, dims, blocksize):
    pad_dims = [0 for _ in range(2 * len(tensor.shape))]
    for dim in dims:
        size = tensor.shape[dim]
        next_multiple_of_block = ((size - 1) // blocksize + 1) * blocksize
        delta = next_multiple_of_block - size
        pad_dims[-2 * dim - 1] = delta
    
    return F.pad(tensor, pad_dims, "constant", 0)


class HiggsLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        higgs_d: int,
        higgs_n: int = 256,
        for_eval=False,
        bias=True,
        device=None,
        dtype=None,
    ):
        assert (
            higgs_n == 256 and higgs_d in [1, 2, 3, 4, 5, 6] # HIGGS
            or higgs_d == 1 and higgs_n in [4, 8, 16]        # FLUTE
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.hadamard_size = 1024
        self.higgs_d = higgs_d
        self.higgs_n = higgs_n
        self.for_eval = for_eval
        
        in_features = ((in_features - 1) // self.hadamard_size + 1) * self.hadamard_size
        num_hadamard_groups = in_features // self.hadamard_size
        post_hadamard_size = ((self.hadamard_size - 1) // higgs_d) * higgs_d + higgs_d
        in_features = ((in_features - 1) // post_hadamard_size + 1) * post_hadamard_size
        num_higgs_groups = in_features // higgs_d
        
        # CODES
        self.register_parameter(
            f'codes_{higgs_d}_{higgs_n}',
            nn.Parameter(
                torch.empty(
                    (out_features, num_higgs_groups),
                    device=device,
                    dtype=torch.uint8,
                ),
                requires_grad=False,
            )
        )
        self.codes = None
        
        # SCALES
        self.scales = nn.Parameter(
            torch.empty((out_features, num_hadamard_groups), **factory_kwargs), requires_grad=False
        )

        # BIAS
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs), requires_grad=False)
        else:
            self.register_parameter("bias", None)
    
    def forward(self, input):
        if self.codes is None:
            if self.for_eval:
                post_hadamard_size = ((1024 - 1) // self.higgs_d + 1) * self.higgs_d
                
                self.codes = _dequantize_weight(
                    self.__getattr__(f"codes_{self.higgs_d}_{self.higgs_n}")[:,:,None],
                    torch.load(f"../grids/EDEN{self.higgs_d}-{self.higgs_n}.pt").half()[None,:,None,:],
                )
                self.codes = self.codes.reshape(self.codes.shape[0], -1, post_hadamard_size)[...,:1024]
                self.codes *= self.scales[...,None]
                self.codes = self.codes.reshape(self.codes.shape[0], -1)
            else:
                self.codes = self.__getattr__(f"codes_{self.higgs_d}_{self.higgs_n}")
        input = pad_to_block(input, [-1], self.hadamard_size)
        input = hadamard_transform(
            input.reshape(input.shape[:-1] + (-1, self.hadamard_size)),
            scale=1/math.sqrt(self.hadamard_size),
        )
        input = input.reshape(input.shape[:-2] + (-1,))

        if self.for_eval:
            return nn.functional.linear(input, self.codes, self.bias)
        else:
            return torch.ops.higgs.higgs_matmat(
                input,
                self.codes,
                self.scales,
                self.bias,
            )


class HiggsMultiLinear(nn.Module):
    def __init__(
        self,
        multicodes,
        scales,
        bias,
    ):
        super().__init__()
        
        # CODES
        for (higgs_d, higgs_n), codes in multicodes.items():
            self.register_parameter(
                f'codes_{higgs_d}_{higgs_n}',
                nn.Parameter(
                    codes,
                    requires_grad=False,
                )
            )
        
        # SCALES
        self.scales = nn.Parameter(
            scales, requires_grad=False
        )

        # BIAS
        if bias:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.register_parameter("bias", None)
