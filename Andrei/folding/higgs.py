import math

import torch

import sys
sys.path.append("../")
from gptq.edenn import higgs_quantize_dequantize



def quantize_layer_higgs(w: torch.Tensor, dim: int, higgs_d: int, higgs_n: int):
    scales = torch.norm(w, dim=dim, keepdim=True) / math.sqrt(w.shape[dim])
    w /= scales
    
    w = w.reshape(w.shape[0], -1, higgs_d)
    for i in range(0, w.shape[0], 64):
        w[i:i+64] = higgs_quantize_dequantize(w[i:i+64].float(), higgs_d, higgs_n)[0].half()
    w = w.reshape(w.shape[0], -1)
    
    w *= scales
    
    return w
    