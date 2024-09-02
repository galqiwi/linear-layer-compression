import argparse
import os
import sys
sys.path.insert(0, 'AQLM')

import time
import random
from tqdm.auto import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from src.aq import QuantizedWeight

torch.set_num_threads(16)
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# num_codebooks = 2
# nbits_per_codebook = 8
out_group_size = 1
in_group_size = 8
batch_size = 16384
beam_size = 1
sparsity_regularizer = 0
print_frequency = 10
scale_nbits = 0    # 0 means no scales, 16 means no compression;
codebook_values_nbits = 16  # less than 16 means we quantize codebooks as well
init_max_iter = 100
steps_per_epoch = 100

relative_mse_tolerance = 0.01

def get_loss_after_quantization(reference_weight, XTX, num_codebooks, nbits_per_codebook, relative_mse_tolerance = 0.01, verbose=False):
    def get_loss():
        delta_weight = (quantized_weight() - reference_weight).double()
        return (delta_weight @ XTX.double()).flatten() @ delta_weight.flatten() / len(delta_weight)
    
    quantized_weight = QuantizedWeight(
        XTX=XTX, reference_weight=reference_weight, num_codebooks=num_codebooks,
        nbits_per_codebook=nbits_per_codebook, scale_nbits=scale_nbits, 
        out_group_size=out_group_size, in_group_size=in_group_size,
        verbose=verbose, max_iter=init_max_iter,   # faster init, not tested
    )
    if verbose:
        print("AVG bits:", quantized_weight.estimate_nbits_per_parameter())
    opt = torch.optim.Adam(quantized_weight.parameters(), lr=1e-4, betas=(0.0, 0.95), amsgrad=True)

    previous_best_loss = float('inf')
    best_loss = float('inf')
    
    for epoch in range(1_000_000_000):
        done = False
        
        for step in range(steps_per_epoch):
            loss = get_loss()
        
            if not torch.isfinite(loss).item():
                raise ValueError(f"Quantization loss is {loss}")
    
            best_loss = min(loss.item(), best_loss)
    
            if step == 0:
                if loss.item() / previous_best_loss > (1.0 - relative_mse_tolerance):
                    if verbose:
                        print(f'{previous_best_loss = }')
                        print(f'{loss.item() = }')
                    return quantized_weight.estimate_nbits_per_parameter(), get_loss().item()
                    
                    done = True
                    break # early stopping; no updates after last epoch's beam search
                previous_best_loss = min(previous_best_loss, loss.item())
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            if verbose and ((epoch * steps_per_epoch + step) % print_frequency == 0):
                print(f"epoch={epoch}\tstep={step}\tloss={loss.item():.10f}\t")
    
        if done:
            break
        
        quantized_weight.beam_search_update_codes_(
            XTX, reference_weight, beam_size=beam_size,
            sparsity_regularizer=sparsity_regularizer, dim_rng=random.Random(),
            verbose=verbose,
        )

    return quantized_weight.estimate_nbits_per_parameter(), get_loss().item()


parser = argparse.ArgumentParser(description="Process input file paths.")
    
parser.add_argument('--xtx_path', type=str, required=True, help='Path to the XTX file.')
parser.add_argument('--weight_path', type=str, required=True, help='Path to the weight file.')
parser.add_argument('--num_codebooks', type=int, required=True, help='Number of codebooks.')
parser.add_argument('--nbits_per_codebook', type=int, required=True, help='Number of bits per codebook.')

args = parser.parse_args()

XTX = torch.load(args.xtx_path, map_location='cpu')
XTX = XTX.cuda().to(torch.float32)

reference_weight = torch.load(args.weight_path, map_location='cpu')
reference_weight = reference_weight.cuda().to(torch.float32)

random_weight = torch.randn(reference_weight.shape, device=reference_weight.device)
random_weight = (random_weight - random_weight.mean()) / random_weight.std()
random_weight = random_weight * reference_weight.std() + reference_weight.mean()

wbits, loss = get_loss_after_quantization(reference_weight, XTX, args.num_codebooks, args.nbits_per_codebook)
_, random_loss = get_loss_after_quantization(random_weight, XTX, args.num_codebooks, args.nbits_per_codebook)

print({
    'xtx_path': args.xtx_path,
    'weight_path': args.weight_path,
    'num_codebooks': args.num_codebooks,
    'nbits_per_codebook': args.nbits_per_codebook,
    'real_nbits_per_parameter': wbits,
    'wbits': wbits,
    'loss': loss,
    'random_loss': random_loss,
})
