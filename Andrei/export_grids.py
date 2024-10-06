import pathlib
from itertools import chain

import torch
from tqdm.auto import tqdm


if __name__ == "__main__":
    multidim_grids_folder = pathlib.Path(__file__).parent.resolve().joinpath("mult_dimensional_grids/")
    singledim_grids_folder = pathlib.Path(__file__).parent.resolve().joinpath("one_dim_1_1000/")
    my_grids_folder = pathlib.Path(__file__).parent.resolve().joinpath("grids/")
    
    for file in tqdm(sorted(chain(multidim_grids_folder.iterdir(), singledim_grids_folder.iterdir()))):
        try:
            filename = file.name
            edenn_n, edenn_d, _ = filename.split('_')
            edenn_n = int(edenn_n)
            edenn_d = int(edenn_d)
            
            with open(file, "r") as grid_file:
                grid_string = file.read_text()
        
            grid = torch.asarray([float(val) for val in grid_string.split()]).reshape(edenn_n + 1, edenn_d + 3)[:-1,1:edenn_d + 1].cuda()
            torch.save(grid, my_grids_folder.joinpath(f"EDEN{edenn_d}-{edenn_n}.pt"))
        except Exception as e:
            print(f"Skipping {edenn_d} {edenn_n}, {e}")
