from fast_hadamard_transform import hadamard_transform
from eval import *


class NoisyHadamarLinear(torch.nn.Module):
    def __init__(self, weight, bias, *, had_block_size = 2048, relative_mse = 0):
        super().__init__()

        weight = weight.detach().clone()
        if bias is not None:
            bias = bias.detach().clone()

        self.had_block_size = had_block_size

        self.out_features, self.in_features = weight.shape

        self.inner = torch.nn.Linear(self.in_features, self.out_features, bias=(bias is not None), dtype=weight.dtype,
                                     device=weight.device)

        assert self.in_features % self.had_block_size == 0, (self.in_features, self.had_block_size)
        weight = weight.reshape(self.out_features, self.in_features // self.had_block_size, self.had_block_size)
        weight = hadamard_transform(weight, scale=1 / (self.had_block_size ** 0.5))
        weight = weight.reshape(self.out_features, self.in_features)

        weight = weight + torch.randn_like(weight) * torch.norm(weight) * (relative_mse ** 0.5) / (weight.numel() ** 0.5)



        self.inner.weight.data = weight
        if bias is not None:
            self.inner.bias.data = bias

    def forward(self, input):
        input_shape = input.shape

        assert input.shape[-1] % self.had_block_size == 0

        input = input.reshape(-1, self.had_block_size)
        input = hadamard_transform(input, scale=1 / (self.had_block_size ** 0.5))
        input = input.reshape(input_shape)

        return self.inner(input)


def do_eval_ppl(
        model,
        model_path,
        model_seqlen,
        device='cuda:0',
        offload_activations=False,
):
    testloader = get_loaders(
        'wikitext2',
        seed=0,
        model_path=model_path,
        seqlen=model_seqlen,
        eval_mode=True,
        use_fast_tokenizer=False,
        trust_remote_code=False,
    )

    output = {}

    ppl = perplexity_eval(
        model,
        testloader,
        dataset_name='wikitext2',
        model_seqlen=model_seqlen,
        device=device,
        offload_activations=offload_activations,
    )
    output['wikitext2'] = ppl
    # make sure that the cache is released
    torch.cuda.empty_cache()

    return output


def get_module_by_path(model, path):
    if path == '':
        return model
    splitted = path.split('.', 1)
    if len(splitted) == 1:
        splitted.append('')
    next_name, suffix = splitted

    try:
        next_module = model[int(next_name)]
    except:
        next_module = getattr(model, next_name)

    return get_module_by_path(next_module, suffix)


def set_module_by_path(model, path, value):
    parts = path.split('.')
    prefix = '.'.join(parts[:-1])
    parent = get_module_by_path(model, prefix)
    setattr(parent, parts[-1], value)
