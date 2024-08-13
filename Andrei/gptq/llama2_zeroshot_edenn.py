import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

import wandb
from tqdm import tqdm, trange

from edenn import edenn, pad_to_block, HadLinear
from fast_hadamard_transform import hadamard_transform


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
def quantize_linear_layer(layer: nn.Linear, hadamard_groupsize: int, edenn_d: int, edenn_n: int):
    weight = layer.weight.clone()
    
    # Pad to Hadamard transform size
    weight = pad_to_block(weight, [1], hadamard_groupsize)
    
    # Scale and Hadamard transform
    mult = weight.shape[1] // hadamard_groupsize
    weight = weight.reshape(-1, mult, hadamard_groupsize)
    scales = torch.linalg.norm(weight, axis=-1)
    weight = hadamard_transform(weight) / scales[:, :, None]
    weight = weight.reshape(-1, mult * hadamard_groupsize)
    
    # Pad to edenn_d and project
    real_num_columns = weight.shape[1]
    weight = pad_to_block(weight, [1], edenn_d)
    
    weight = weight.reshape(weight.shape[0], -1, edenn_d)
    for i in range(0, weight.shape[0], 128):
        weight[i:i+128], entorpy = edenn(weight[i:i+128], edenn_d, edenn_n)
    weight = weight.reshape(weight.shape[0], -1)[:,:real_num_columns]
    
    # Unscale
    weight = (weight.reshape(weight.shape[0], -1, hadamard_groupsize) * scales[:, :, None]).reshape(weight.shape[0], -1)
    
    return HadLinear(weight, hadamard_groupsize), entorpy
    

@torch.no_grad()
def llama_zeroshot(model, args, device):
    linear_layers = find_layers(model)
    
    for name, layer in tqdm(linear_layers.items(), desc="Quantizing linear layers..."):
        if "lm_head" in name:
            continue
        quantized_layer, entropy = quantize_linear_layer(layer.to(device), args.hadamard_groupsize, args.edenn_d, args.edenn_n)
        wandb.log({f"layer_entropy": entropy})
        replace_submodule(model, name, quantized_layer.cpu())
        
    return model
        

@torch.no_grad()
def llama_eval(model, dataloader, dev):
    print('Evaluating ...')

    nsamples = len(dataloader) 

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = []
    attention_masks = []
    position_ids = []

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps.append(inp)
            attention_masks.append(kwargs['attention_mask'])
            position_ids.append(kwargs['position_ids'])
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch.to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    for i in trange(len(layers), desc=f"Evaluating layer-by-layer..."):
        layer = layers[i].to(dev)
        for j in range(nsamples):
            inps[j] = layer(inps[j], attention_mask=attention_masks[j], position_ids=position_ids[j])[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i]
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = (dataloader[i].to(dev))[:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache
    
    return ppl.item()

@torch.no_grad()
def eval_grid(edenn_d: int, edenn_n: int):
    x = torch.empty((2**16, edenn_d), device=DEV, dtype=torch.float16).normal_()
    dequant, entropy = edenn(x, edenn_d, edenn_n)
    mse = (x - dequant).pow(2).mean().item()
    return mse, entropy / edenn_d


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        '--edenn-d', type=int,
        help='EDENN grid dimension'
    )
    parser.add_argument(
        '--edenn-n', type=int,
        help='EDENN grid size'
    )
    parser.add_argument(
        '--hadamard_groupsize', type=int, default=1024, choices=[64, 128, 256, 512, 1024, 2048, 4096],
        help='Groupsize to use for hadamard; default is 1024.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--seqlen',
        type=int, default=8192, help='Seq len for PPL evals.'
    )

    args = parser.parse_args()
    
    wandb.init(
        # set the wandb project where this run will be logged
        project="compression_horizon",
        
        # track hyperparameters and run metadata
        config=args,
        name=f"{args.model=},{args.hadamard_groupsize=},{args.edenn_d=},{args.edenn_n=},{args.seed=}",
    )
    
    mse, entropy = eval_grid(args.edenn_d, args.edenn_n)
    wandb.log({f"expected_mse": mse, "expected_entropy": entropy, "bitwidth": np.log2(args.edenn_n) / args.edenn_d, "edenn_d": args.edenn_d, "edenn_n": args.edenn_n})

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="cpu")
    model.seqlen = args.seqlen
    model.eval()


    model = llama_zeroshot(model, args, DEV)

    datasets = ['wikitext2'] 
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        ppl = llama_eval(model, testloader, DEV)
        wandb.log({f"{dataset}_PPL": ppl})
