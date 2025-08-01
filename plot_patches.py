import os
os.environ["USE_PYGEOS"] = "0"
import torch
import random
import argparse
import pickle as pk
import pytorch_lightning as pl
import lightning as L
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from utils import *
from HIST2ST import *
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from our_dataset import OUR_BUILDER
from spared.datasets import get_dataset
from spared.metrics import get_metrics
import anndata as ad
from datetime import datetime
from log_functions import *
from PIL import Image
import numpy as np

parser = get_main_parser()
args = parser.parse_args()
args_dict = vars(args)

#dataset = get_dataset(args.dataset, visualize=False)

path = "/home/dvegaa/ST_Diffusion/stDiff_Spared/processed_data/Visium_data/10xgenomic_mouse_brain_sagittal_posterior/2025-02-24-17-19-56/adata.h5ad"
adata = ad.read_h5ad(path)

adata_slide = adata[adata.obs["slide_id"]=="V1_Mouse_Brain_Sagittal_Posterior"]

coord =  pd.DataFrame(adata_slide.obsm['spatial'], columns=['x_coord', 'y_coord'])
# Get indices of sorted coordinates
coords_index = coord.index

# Extract all patches in sorted order
patches = [adata.obsm["patches_scale_1.0"][i].reshape(224, 224, 3) for i in coords_index]


def plot_patches(coord, patches, path):
    # Normalize coordinates to a grid
    x_unique = np.sort(coord["x_coord"].unique())  # Unique x values
    y_unique = np.sort(coord["y_coord"].unique())[::-1]  # Unique y values (reverse to keep order)

    x_to_idx = {val: i for i, val in enumerate(x_unique)}
    y_to_idx = {val: i for i, val in enumerate(y_unique)}

    # Define grid size
    grid_width = len(x_unique)
    grid_height = len(y_unique)

    # Create an empty canvas for reconstruction
    canvas = np.zeros((grid_height * 224, grid_width * 224, 3), dtype=np.uint8)

    # Place patches in the correct positions
    for (x, y), patch in zip(coord.values, patches):
        x_idx = x_to_idx[x]
        y_idx = y_to_idx[y]
        
        canvas[y_idx * 224: (y_idx + 1) * 224, x_idx * 224: (x_idx + 1) * 224] = patch

    # Plot the reconstructed image
    plt.figure(figsize=(10, 10))
    plt.imshow(canvas)
    plt.axis("off")
    plt.title("Reconstructed Image from Patches")

    # Save the reconstructed image
    Image.fromarray(canvas).save(path)


path = f"/home/dvegaa/Hist2ST/{args.dataset}_reconstructed_image.png"
#plot_patches(coord=coord, patches=patches, path=path)

def divide_patches(coords, patches):
    # Normalize coordinates to a grid
    x_unique = np.sort(coords["x_coord"].unique())  # Unique x values
    y_unique = np.sort(coords["y_coord"].unique())[::-1]  # Unique y values (reverse to keep order)

    x_to_idx = {val: i for i, val in enumerate(x_unique)}
    y_to_idx = {val: i for i, val in enumerate(y_unique)}

    # Define grid size
    grid_width = len(x_unique)
    grid_height = len(y_unique)

    # Split image into quadrants
    mid_x = grid_width // 2
    mid_y = grid_height // 2

    index_quadrants = {
        "Q1": [],  # Top-left
        "Q2": [],  # Top-right
        "Q3": [],  # Bottom-left
        "Q4": []   # Bottom-right
    }

    # Assign patches to quadrants
    for (x, y), patch, index in zip(coords.values, patches, coords_index):
        x_idx = x_to_idx[x]
        y_idx = y_to_idx[y]

        if x_idx < mid_x and y_idx < mid_y:
            index_quadrants["Q1"].append(index)
        elif x_idx >= mid_x and y_idx < mid_y:
            index_quadrants["Q2"].append(index)
        elif x_idx < mid_x and y_idx >= mid_y:
            index_quadrants["Q3"].append(index)
        else:
            index_quadrants["Q4"].append(index)

    return index_quadrants

index_quadrants = divide_patches(coords=coord, patches=patches)
breakpoint()

for i in index_quadrants.keys():
    coords_quadrant = coord.loc[index_quadrants[i]]
    # Extract all patches in sorted order
    patches_quadrants = [adata.obsm["patches_scale_1.0"][i].reshape(224, 224, 3) for i in index_quadrants[i]]
    # Create directories
    os.makedirs(f"patches_plots/{args.dataset}/{i}", exist_ok=True)
    path_quadrants = os.path.join(f"patches_plots/{i}", f"{i}_reconstructed_image.png")
    #plot_patches(coord=coords_quadrant, patches=patches_quadrants, path=path_quadrants)


"""
top_10_coords = coord_sorted.head(10)
coords_index = top_10_coords.index
top_10_patches = []

for i in coords_index:
    top_10_patches.append(adata.obsm["patches_scale_1.0"][i].reshape(224, 224, 3))

os.makedirs("patches_plots", exist_ok=True)

def save_array_as_image(array, filename):
    # Asegurar que el array es de tipo uint8 (0-255)
    array = np.array(array, dtype=np.uint8)
    
    # Convertir el array en imagen
    image = Image.fromarray(array)
    
    # Guardar la imagen
    image.save(filename)

for i in range(len(top_10_patches)):
    save_array_as_image(top_10_patches[i], f"/home/dvegaa/Hist2ST/patches_plots/{i}_output_image.png")

"""