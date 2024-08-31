import os

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

import wandb
from tqdm import tqdm, trange

from edenn import edenn, pad_to_block, HadLinear
from fast_hadamard_transform import hadamard_transform
from gptq import apply_gptq, get_accumulate_input_fn

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
    
def replace_empty(model: nn.Module, had_block_size: int):
    linear_layers = find_layers(model)
    for name, layer in tqdm(linear_layers.items(), desc="Replacing linear layers..."):
        if "lm_head" in name:
            continue
        
        replace_submodule(model, name, HadLinear(pad_to_block(layer.weight, [1], had_block_size)))


@torch.no_grad()
def quantize_linear_layer(layer: nn.Linear, hadamard_groupsize: int, edenn_d: int, edenn_n: int):
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
    for i in range(0, weight.shape[0], 64):
        weight[i:i+64], entorpy = edenn(weight[i:i+64], edenn_d, edenn_n)
    weight = weight.reshape(weight.shape[0], -1)[:,:real_num_columns]
    
    # Unscale
    weight = (weight.reshape(weight.shape[0], -1, hadamard_groupsize) * scales[:, :, None]).reshape(weight.shape[0], -1)
    
    return HadLinear(weight, hadamard_groupsize), entorpy
    

@torch.no_grad()
def llama_rtn(model, args, device):
    linear_layers = find_layers(model)
    
    for name, layer in tqdm(linear_layers.items(), desc="Quantizing linear layers..."):
        if "lm_head" in name:
            continue
        quantized_layer, entropy = quantize_linear_layer(layer.to(device), args.hadamard_groupsize, args.edenn_d, args.edenn_n)
        wandb.log({f"layer_entropy": entropy})
        replace_submodule(model, name, quantized_layer.cpu())
        
    return model


@torch.no_grad()
def llama_gptq(model, args, dataloader, dev):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    model.model.rotary_emb.inv_freq = model.model.rotary_emb.inv_freq.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = []
    outs = []
    attention_masks = []
    position_ids = []

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps.append(inp)
            outs.append(torch.zeros_like(inp))
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
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    for i in trange(len(layers), desc="Quantizing with GPTQ..."):
        layer = layers[i].to(dev)
        linear_layers = find_layers(layer)

        hessians = {name: None for name in linear_layers}
        num_samples = {name: 0 for name in linear_layers}
        handles = [
            linear_layers[name].register_forward_hook(
                get_accumulate_input_fn(name, hessians, num_samples)
            ) for name in linear_layers
        ]
        for j in trange(args.nsamples, leave=False, desc="Before pass..."):
            outs[j] = layer(inps[j], attention_mask=attention_masks[j], position_ids=position_ids[j])[0]
        for h in handles:
            h.remove()

        for name, linear in linear_layers.items():
            quantized_layer = apply_gptq(
                linear.weight.data, 2 * hessians[name] / num_samples[name],
                edenn_d=args.edenn_d, edenn_n=args.edenn_n,
                had_block_size=args.hadamard_groupsize,
            )
                
            quantized_linear = HadLinear(quantized_layer, args.hadamard_groupsize)
            replace_submodule(layer, name, quantized_linear)

        mse = 0
        for j in trange(args.nsamples, leave=False, desc="After pass..."):
            out = layer(inps[j], attention_mask=attention_masks[j], position_ids=position_ids[j])[0]
            mse += torch.nn.functional.mse_loss(outs[j][0], out[0]).item()
            inps[j] = out
        wandb.log({"obc_mse": mse})

        if any([inp.isnan().any() for inp in inps]):
            raise Exception("NaNs!")
        
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
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


def get_zero_shots(model, task_list = ('arc_easy',), num_fewshots=1):
    import lm_eval
    from lm_eval import evaluator

    lm_eval_model = lm_eval.models.huggingface.HFLM(
        pretrained=model,
    )

    tasks = lm_eval.tasks.get_task_dict(task_list)
    if num_fewshots != 1:
        # TODO: make fewshots properly
        for task_name in tasks:
            task = tasks[task_name]
            if isinstance(task, tuple):
                task = task[1]
            if task is None:
                continue
            task.config.num_fewshot = num_fewshots

    results = evaluator.evaluate(
        lm=lm_eval_model,
        task_dict=lm_eval.tasks.get_task_dict(task_list),
    )

    result_dict = {task_name: task_result['acc,none'] for task_name, task_result in results['results'].items()}
    result_err_dict = {f'{task_name}_err': task_result['acc_stderr,none'] for task_name, task_result in
                       results['results'].items()}
    result_dict = dict(list(result_dict.items()) + list(result_err_dict.items()))

    if num_fewshots != 1:
        result_dict = {f'{task_name}@{num_fewshots}': acc for task_name, acc in result_dict.items()}

    return result_dict


@torch.no_grad()
def eval_grid(edenn_d: int, edenn_n: int):
    x = torch.empty((2**16, edenn_d), device=DEV).normal_()
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
    parser.add_argument(
        '--method', type=str, choices=["rtn", "gptq"], default="gptq", help="Method to quantize with",
    )
    parser.add_argument(
        '--dataset', type=str, default='red', choices=['red'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=256,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--cache-dir', type=str, default=None,
        help='Models cache dir',
    )
    args = parser.parse_args()
    
    wandb.init(
        # set the wandb project where this run will be logged
        entity="rock-and-roll",
        project="edenn-gptq",
        
        # track hyperparameters and run metadata
        config=args,
        name=f"{args.model=},{args.hadamard_groupsize=},{args.edenn_d=},{args.edenn_n=},{args.seed=}",
    )
    
    mse, entropy = eval_grid(args.edenn_d, args.edenn_n)
    wandb.log({
        "model": args.model,
        "method": args.method,
        "dataset": args.dataset,
        "nsamples": args.nsamples,
        "expected_mse": mse,
        "expected_entropy": entropy,
        "bitwidth": np.log2(args.edenn_n) / args.edenn_d,
        "edenn_d": args.edenn_d,
        "edenn_n": args.edenn_n
    })

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="cpu")
    model.seqlen = args.seqlen
    model.eval()
    
    ckpt_name = f"{args.model}_{args.method}_{args.edenn_d}_{args.edenn_n}.pt"
    
    if args.cache_dir is not None and os.path.isfile(f"{args.cache_dir}/{ckpt_name}"):
        replace_empty(model, args.hadamard_groupsize)
        
        print(f"Using quantized model at {args.cache_dir}/{ckpt_name}")
        
        model.load_state_dict(torch.load(
            f"{args.cache_dir}/{ckpt_name}"
        ))
    else:
        match args.method:
            case "rtn":
                model = llama_rtn(model, args, DEV)
            case "gptq":
                dataloader, testloader = get_loaders(
                    args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
                )
                model = llama_gptq(model, args, dataloader, DEV)
            case _:
                raise Exception("AAA")
        
        if args.cache_dir is not None:
            ckpt_path = f"{args.cache_dir}/{ckpt_name}"
            last_slash_pos = ckpt_path.rfind("/")
            os.makedirs(ckpt_path[:last_slash_pos], exist_ok=True)
            torch.save(
                model.state_dict(),
                ckpt_path,
            )

    datasets = ['wikitext2'] 
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        ppl = llama_eval(model, testloader, DEV)
        wandb.log({f"{dataset}_PPL": ppl})
    
    model = model.to(DEV)
    wandb.log(get_zero_shots(model, task_list = ('mmlu',), num_fewshots=5))