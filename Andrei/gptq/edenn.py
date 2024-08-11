import numpy as np
import torch

import pathlib
grids_folder = pathlib.Path(__file__).parent.parent.resolve().joinpath("grids/")

GRIDS = {
}
# Read files in the folder and read grids in the EDEN{DIM}_{SIZE}.pt format
for file in grids_folder.iterdir():
    if file.suffix == ".pt":
        dim, size = map(int, file.stem[4:].split('-'))
        GRIDS[dim] = GRIDS.get(dim, {})
        GRIDS[dim][size] = torch.load(file)

GRID_NORMS = {k1: {k2: torch.linalg.norm(GRIDS[k1][k2], dim=1) ** 2 for k2 in v1.keys()} for k1, v1 in GRIDS.items()}


def entropy(idx):
    _, counts = torch.unique(idx, return_counts=True)
    return -torch.sum(counts / len(idx) * torch.log2(counts / len(idx)))

def edenn(x, dim, size):
    idx = torch.argmax(2 * x @ GRIDS[dim][size].T - GRID_NORMS[dim][size], dim=-1)
    return GRIDS[dim][size][idx], entropy(idx)
