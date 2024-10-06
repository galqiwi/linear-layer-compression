import torch

from transformers import AutoModelForCausalLM
from utils import do_eval_ppl
import torch
from af4 import get_af4_grid
import tqdm
import torch.nn as nn
from utils import get_module_by_path
from compressors import quantize_dequantize_weight
import argparse
from eval import get_zero_shots


def parse_args():
    parser = argparse.ArgumentParser(description='Quantize model weights and evaluate perplexity.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model.')
    parser.add_argument('--model_seqlen', type=int, default=8192, help='Sequence length of the model.')
    parser.add_argument('--block_size', type=int, default=64, help='Block size for quantization.')
    parser.add_argument('--eval_ppl', action='store_true', help='Evaluate perplexity after quantization.')
    parser.add_argument('--eval_mmlu', action='store_true', help='Evaluate mmlu@5 after quantization.')
    parser.add_argument(
        '--eval_zero_shots',
        action='store_true',
        help='Evaluate winogrande, piqa, hellaswag, arc_easy, and arc_challenge after quantization.'
    )

    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_grad_enabled(False)

    model_pt = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True, torch_dtype=torch.float16, device_map='cuda',
    )

    layers = sorted([
        name for name, value in
        model_pt.named_modules() if (
            isinstance(value, nn.Linear) and
            name.startswith('model.layers')
        )
    ])

    print('Getting af4 grid')
    codes = get_af4_grid(args.block_size).half()

    for layer in tqdm.tqdm(layers, desc='Quantizing weights'):
        linear = get_module_by_path(model_pt, layer)

        linear.weight.data = quantize_dequantize_weight(
            linear.weight,
            codes=codes.half(),
            block_size=args.block_size,
        ).cuda()

    if args.eval_ppl:
        ppl = do_eval_ppl(
            model_pt,
            model_path=args.model_path,
            model_seqlen=args.model_seqlen,
            device='cuda:0',
            offload_activations=True,
        )
        print(f'Wikitext2 PPL: {ppl}')

    if args.eval_mmlu:
        mmlu = get_zero_shots(model_pt, task_list=['mmlu'], num_fewshots=5)['mmlu@5']
        print(f'MMLU@5: {mmlu}')

    if args.eval_zero_shots:
        zero_shots = get_zero_shots(
            model_pt,
            task_list=['winogrande', 'piqa', 'hellaswag', 'arc_easy', 'arc_challenge'],
            num_fewshots=1,
        )
        print(zero_shots)


if __name__ == '__main__':
    main()
