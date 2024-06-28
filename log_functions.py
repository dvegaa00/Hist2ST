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
import plotly.express as px
import matplotlib.colors as colors
from torchvision.transforms import Normalize
from matplotlib import pyplot as plt
import matplotlib 
from tqdm import tqdm

#TODO: Documentación
def get_predictions(adata: ad.AnnData, args, model, layer='c_d_log1p', device="cuda")->None:
    """
    _summary_

    Args:
        model (_type_): model to run
        layer (str, optional): layer for prediction. Defaults to 'c_d_log1p'.
        data (int, optional): Dataset as Adata. 
        use_cuda (bool, optional): Whether to use cuda or not. Defaults to False.
    """
    # Set the X of the adata to the layer casted to float32
    adata.X = adata.layers[layer].astype(np.float32)
    slides_names = adata.obs["slide_id"].unique().tolist()

    #ID to Name list
    id2name = dict(enumerate(slides_names))

    #Our dataser builder
    custom_dataset = OUR_BUILDER(adata=adata, prediction=layer, patch_size=224, prune=args.prune, neighs=args.neighbor, id2name=id2name)

    # Get complete dataloader
    dataloader = DataLoader(custom_dataset, batch_size=1, num_workers=0, shuffle=False)
    # Define global variables
    glob_expression_pred = None
    glob_ids = None
    glob_ids = adata.obs['unique_id'].tolist() if glob_ids is None else glob_ids + adata.obs['unique_id'].tolist()

    # Set model to eval mode
    model=model.to(device)
    model.eval()

    # Get complete predictions
    with torch.no_grad():
        for patch, position, exp, adj, oris, sfs, centers, mask in tqdm(dataloader):
            patch, position, adj = patch.to(device), position.to(device), adj.to(device).squeeze(0)
            pred = model(patch, position, adj)[0]
            exp_pred = pred.cpu()
            gt = exp.squeeze().cpu().numpy()

            # Concat batch to get global predictions and IDs
            glob_expression_pred = exp_pred if glob_expression_pred is None else torch.cat((glob_expression_pred, exp_pred))

        # Handle delta prediction
        if 'deltas' in layer:
            mean_key = f'{layer}_avg_exp'.replace('deltas', 'log1p')
            means = torch.tensor(adata.var[mean_key], device=glob_expression_pred.device)
            glob_expression_pred = glob_expression_pred+means
        
        
        # Put complete predictions in a single dataframe
        pred_matrix = glob_expression_pred
        pred_df = pd.DataFrame(pred_matrix, index=glob_ids, columns=adata.var_names)
        pred_df = pred_df.reindex(adata.obs.index)

        # Log predictions to wandb
        wandb_df = pred_df.reset_index(names='sample')
        wandb.init()
        wandb.log({'predictions': wandb.Table(dataframe=wandb_df)})
        
        # Add layer to adata
        adata.layers[f'predictions,{layer}'] = pred_df

        
