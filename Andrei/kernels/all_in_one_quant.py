import math
import sys
import os
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from tqdm import tqdm, trange
from fast_hadamard_transform import hadamard_transform


sys.path.append("..")
from gptq.edenn import higgs_quantize, pad_to_block
from linear import HiggsMultiLinear

DEV = torch.device('cuda')

def find_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def replace_submodule(module, submodule_path, new_submodule):
    submodule_names = submodule_path.split(".")
    for submodule in submodule_names[:-1]:
        module = getattr(module, submodule)
    setattr(module, submodule_names[-1], new_submodule)


@torch.no_grad()
def get_codes_and_scales(layer: nn.Linear, edenn_d: int, edenn_n: int):    
    weight = layer.weight.clone().float()
    # Pad to Hadamard transform size
    weight = pad_to_block(weight, [1], 1024)
    
    # Scale and Hadamard transform
    mult = weight.shape[1] // 1024
    weight = weight.reshape(-1, mult, 1024)
    scales = torch.linalg.norm(weight, axis=-1)
    weight = hadamard_transform(weight) / scales[:, :, None]
    
    # Pad to edenn_d and project
    weight = pad_to_block(weight, [2], edenn_d).reshape(weight.shape[0], mult, -1, edenn_d)

    # Quantize
    codes = torch.empty(weight.shape[:-1], device=weight.device, dtype=torch.uint8)

    for i in range(0, weight.shape[0], 64):
        codes[i:i+64] = higgs_quantize(weight[i:i+64], edenn_d, edenn_n)
        
    codes = codes.reshape(codes.shape[0], -1)
    
    return codes, scales / math.sqrt(1024)
    

@torch.no_grad()
def llama_multi_quantize(model, device):
    linear_layers = find_layers(model)
    
    for (name, layer) in tqdm(linear_layers.items(), desc="Quantizing linear layers..."):
        if "lm_head" in name:
            continue
        
        layer = layer.to(device)
        multicodes = {}
        scales = None
        for edenn_d in tqdm([1, 2, 3, 4, 5, 6], desc="Iterating dimensions", leave=False):
            codes, scales = get_codes_and_scales(layer, edenn_d, 256)
            multicodes[(edenn_d, 256)] = codes.cpu()
        for edenn_d in tqdm([1], desc="Iterating dimensions", leave=False):
            codes, scales = get_codes_and_scales(layer, edenn_d, 8)
            multicodes[(edenn_d, 8)] = codes.cpu()
        layer = layer.cpu()
        scales = scales.cpu().half()
        
        quantized_linear = HiggsMultiLinear(
            multicodes, scales, layer.bias
        )
        
        replace_submodule(model, name, quantized_linear)

    return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        'save_path', type=str,
        help="Dir to save checkpoint to"
    )
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="cpu")
    
    model = llama_multi_quantize(model, DEV)

    torch.save(
        model.state_dict(),
        os.path.join(args.save_path, args.model.split("/")[-1]) + ".pt",
    )
