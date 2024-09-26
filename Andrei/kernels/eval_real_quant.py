import os
import math
import sys
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from accelerate import load_checkpoint_and_dispatch

import wandb
from tqdm import tqdm, trange

from fast_hadamard_transform import hadamard_transform

from linear import HiggsLinear

DEV = torch.device('cuda')


def replace_with_higgs_linear(
    model,
    quantization_config=None,
    current_key_name=None,
    has_been_replaced=False,
):
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if isinstance(module, nn.Linear):
            # Check if the current key is not in the `linear_weights_not_to_quantize`
            if ".".join(current_key_name) in quantization_config:
                in_features = module.in_features
                out_features = module.out_features
                higgs_d, higgs_n = quantization_config[".".join(current_key_name)]


                model._modules[name] = HiggsLinear(
                    in_features,
                    out_features,
                    higgs_d,
                    higgs_n,
                    for_eval=True,
                    bias=module.bias is not None,
                    dtype=module.weight.dtype,
                )
                has_been_replaced = True

                # Store the module class in case we need to transpose the weight later
                model._modules[name].source_cls = type(module)
                # Force requires grad to False to avoid unexpected errors
                model._modules[name].requires_grad_(False)
        if len(list(module.children())) > 0:
            _, has_been_replaced = replace_with_higgs_linear(
                module,
                quantization_config=quantization_config,
                current_key_name=current_key_name,
                has_been_replaced=has_been_replaced,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced

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


def build_layerwise_edenn_config(
    edenn_d: Optional[int] = None,
    edenn_n: Optional[int] = None, 
    blockwise_edenn_config: Optional[list[int]] = None,
    layerwise_edenn_config: Optional[dict[str, int]] = None,
) -> list[(int, int)]:
    if layerwise_edenn_config is not None:
        assert edenn_d is None and blockwise_edenn_config is None
        return layerwise_edenn_config
    
    if blockwise_edenn_config is None:
        assert edenn_d is not None
        blockwise_edenn_config = [(edenn_d, edenn_n) for _ in range(32)]
    
    layer_names = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ]
    
    return {
        f"model.layers.{i}.{layer_name}": blockwise_edenn_config[i]
        for layer_name in layer_names
        for i in range(len(blockwise_edenn_config))
    }
    

def calculate_bitwidth(layerwise_edenn_config):
    total_params = 32 * (4096 * (2 * 4096 + 2 * 4096 * 8 / 32) + 3 * 4096 * 14336)
    
    total_bitwidth = 0
    for name, (edenn_d, edenn_n) in layerwise_edenn_config.items():
        if "q_proj" in name:
            total_bitwidth += math.log2(edenn_n) / edenn_d * 4096 * 4096 / total_params
        elif "k_proj" in name:
            total_bitwidth += math.log2(edenn_n) / edenn_d * 4096 * 4096 * 8 / 32 / total_params
        elif "v_proj" in name:
            total_bitwidth += math.log2(edenn_n) / edenn_d * 4096 * 4096 * 8 / 32 / total_params
        elif "o_proj" in name:
            total_bitwidth += math.log2(edenn_n) / edenn_d * 4096 * 4096 / total_params
        elif "gate_proj" in name:
            total_bitwidth += math.log2(edenn_n) / edenn_d * 4096 * 14336 / total_params
        elif "up_proj" in name:
            total_bitwidth += math.log2(edenn_n) / edenn_d * 4096 * 14336 / total_params
        elif "down_proj" in name:
            total_bitwidth += math.log2(edenn_n) / edenn_d * 4096 * 14336 / total_params
    return total_bitwidth


if __name__ == '__main__':
    import argparse
    sys.path.append("..")
    from gptq.datautils import get_loaders

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        '--edenn-d', type=int, default=None,
        help='EDENN grid dimension'
    )
    parser.add_argument(
        '--edenn-n', type=int, default=None,
        help='EDENN grid size'
    )
    parser.add_argument(
        '--blockwise', type=str, default=None,
        help='Blockwise edenn configs'
    )
    parser.add_argument(
        '--layerwise', type=str, default=None,
        help='Layerwise edenn configs'
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
        '--multiquant-ckpt-path',
        type=str, help='Multiquant checkpoint file'
    )
    args = parser.parse_args()
    
    if args.layerwise is not None:
        import ast
        args.layerwise = ast.literal_eval(args.layerwise)
    if args.blockwise is not None:
        import ast
        args.blockwise = ast.literal_eval(args.blockwise)

    wandb.init(
        # set the wandb project where this run will be logged
        entity="rock-and-roll",
        project="higgs-evals-newgrids" if args.layerwise is None else "higgs-evals-newgrids-layerwise",
        
        # track hyperparameters and run metadata
        config=args,
        name=f"{args.model}",
    )
    
    wandb.log({
        "model": args.model,
    })

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="cpu")
    model.seqlen = args.seqlen
    model.eval()
    
    layerwise_edenn_config = build_layerwise_edenn_config(
        args.edenn_d, args.edenn_n,
        args.blockwise,
        args.layerwise,
    )
    wandb.log({
        "layerwise_edenn_config": layerwise_edenn_config,
        "bitwidth": calculate_bitwidth(layerwise_edenn_config),
    })
    
    model, _ = replace_with_higgs_linear(
        model,
        layerwise_edenn_config,
    )
    
    model = load_checkpoint_and_dispatch(
        model,
        args.multiquant_ckpt_path,
    ).to("cuda")

    datasets = ['wikitext2'] 
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        ppl = llama_eval(model, testloader, DEV)
        wandb.log({f"{dataset}_PPL": ppl})
    
    # model = model.to(DEV)
    # wandb.log(get_zero_shots(model, task_list = ('mmlu',), num_fewshots=5))