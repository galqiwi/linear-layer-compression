import gc
import torch
import typing
from tqdm.auto import tqdm
from hadamard import random_hadamard_matrix

def fuse_ln_linear(layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear], device) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double().to(device)
        linear.weight.data = (W_ * layernorm.weight.double().to(device)).to(linear_dtype).cpu()

        if hasattr(layernorm, 'bias'):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float64))
            linear.bias.data = linear.bias.data.double() + torch.matmul(W_, layernorm.bias.double())
            linear.bias.data = linear.bias.data.to(linear_dtype)
            
    # Update the layernorm
    layernorm.weight.data = torch.ones_like(layernorm.weight.data)
    if hasattr(layernorm, 'bias'):
        layernorm.bias.data = torch.zeros_like(layernorm.bias.data)
            
def fuse_layer_norms(model, device):
    
    # Embedding fusion
    W = model.model.embed_tokens
    W_ = W.weight.data.double().to(device)
    W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(device="cpu", dtype=W.weight.data.dtype)
    del W_
        
    layers = model.model.layers
    
    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in tqdm(layers, desc="Fusing layer norms"):
        # fuse the input layernorms into the linear layers
        fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj], device)    
        fuse_ln_linear(layer.input_layernorm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj], device)
                    
    
    fuse_ln_linear(model.model.norm, [model.lm_head], device)
    


def rotate_embeddings(model, Q: torch.Tensor, device) -> None:
    # Rotate the embeddings.
    W = model.model.embed_tokens
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(dtype=torch.float64).to(device)
    W.weight.data = torch.matmul(W_, Q.to(W_.device)).to(device="cpu", dtype=dtype)

    
def rotate_attention_inputs(layer, Q, device) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(dtype=torch.float64).to(device)
        W.weight.data = torch.matmul(W_, Q.to(W_.device)).to(device="cpu", dtype=dtype)

def rotate_attention_output(layer, Q, device) -> None:
    # Rotate output matrix of the self-attention layer.
    W = layer.self_attn.o_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(dtype=torch.float64).to(device)
    W.weight.data = torch.matmul(Q.to(W_.device).T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(dtype=torch.float64)
        W.bias.data = torch.matmul(Q.to(W_.device).T, b).to(device="cpu", dtype=dtype)

def rotate_mlp_input(layer, Q, device):
    # Rotate the MLP input weights.
    mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(dtype=torch.float64).to(device)
        W.weight.data = torch.matmul(W_, Q.to(W_.device)).to(device="cpu", dtype=dtype)
    
def rotate_mlp_output(layer, Q, device):
    # Rotate the MLP output weights and bias.
    W = layer.mlp.down_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(dtype=torch.float64).to(device)
    W.weight.data = torch.matmul(Q.to(W_.device).T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(dtype=torch.float64)
        W.bias.data = torch.matmul(Q.to(W_.device).T, b).to(device="cpu", dtype=dtype)

def rotate_head(model, Q: torch.Tensor, device) -> None:
    # Rotate the head.
    W = model.lm_head
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(dtype=torch.float64).to(device)
    W.weight.data = torch.matmul(W_, Q.to(W_.device)).to(device="cpu", dtype=dtype)


@torch.inference_mode()
def rotate_model(model, device):
    Q = random_hadamard_matrix(model.config.hidden_size, device)

    rotate_embeddings(model, Q, device)
    rotate_head(model, Q, device)
    gc.collect()
    torch.cuda.empty_cache()
    layers = model.model.layers
    for idx, layer in enumerate(tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputs(layers[idx], Q, device)
        rotate_attention_output(layers[idx], Q, device)
        rotate_mlp_input(layers[idx], Q, device)
        rotate_mlp_output(layers[idx], Q, device)
