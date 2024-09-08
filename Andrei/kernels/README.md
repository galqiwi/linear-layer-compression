### Produce multiquant checkpoint:

```bash
CUDA_VISIBLE_DEVICES=1 python all_in_one_quant.py meta-llama/Meta-Llama-3.1-8B /nfs/scistore19/alistgrp/apanfero/models/higgs
```

### Eval with constant BW

```bash
CUDA_VISIBLE_DEVICES=1 python eval_real_quant.py meta-llama/Meta-Llama-3.1-8B --edenn-n 256 --edenn-d 4 --multiquant-ckpt-path /nfs/scistore19/alistgrp/apanfero/models/higgs/Meta-Llama-3.1-8B.pt
```

`edenn-n` must be 256

### Eval with blockwise BW

```bash
CUDA_VISIBLE_DEVICES=1 python eval_real_quant.py meta-llama/Meta-Llama-3.1-8B --blockwise "[(2,256),(1,256),(2,256),(2,256),(2,256),(2,256),(2,256),(2,256),(2,256),(2,256),(2,256),(2,256),(2,256),(2,256),(2,256),(2,256),(2,256),(2,256),(2,256),(2,256),(2,256),(2,256),(2,256),(3,256),(3,256),(3,256),(3,256),(3,256),(3,256),(2,256),(2,256),(2,256)]" --multiquant-ckpt-path /nfs/scistore19/alistgrp/apanfero/models/higgs/Meta-Llama-3.1-8B.pt
```

32 pairs in the list

### Eval with layerwise BW

```bash
CUDA_VISIBLE_DEVICES=1 python eval_real_quant.py meta-llama/Meta-Llama-3.1-8B --layerwise '{"model.layers.0.self_attn.q_proj": (2, 256)}' --multiquant-ckpt-path /nfs/scistore19/alistgrp/apanfero/models/higgs/Meta-Llama-3.1-8B.pt
```

layers not in `--layerwise` are skipped
