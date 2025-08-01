import numpy as np
import pandas as pd
import torchvision
import json
import squidpy as sq
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.colors as colors
import os
import torch
import wandb
from tqdm import tqdm
import argparse
from spared.metrics import get_metrics
import anndata as ad


# Auxiliary function to use booleans in parser
str2bool = lambda x: (str(x).lower() == 'true')
str2intlist = lambda x: [int(i) for i in x.split(',')]
str2floatlist = lambda x: [float(i) for i in x.split(',')]
str2h_list = lambda x: [str2intlist(i) for i in x.split('//')[1:]]

# Function to get global parser
def get_main_parser():
    parser = argparse.ArgumentParser(description='Code for expression prediction using contrastive learning implementation.')
    # Dataset parameters #####################################################################################################################################################################
    parser.add_argument('--dataset',                    type=str,           default='10xgenomic_human_brain',   help='Dataset to use.')
    parser.add_argument('--prediction_layer',           type=str,           default='c_t_log1p',                help='The prediction layer from the dataset to use.')
    parser.add_argument('--division',                   type=int,           default=2,                help='In hiw many pieces to divide the slides')
    # Model parameters #######################################################################################################################################################################
    parser.add_argument('--sota',                       type=str,           default='pretrain',                 help='The name of the sota model to use. "None" calls main.py, "nn_baselines" calls nn_baselines.py, "pretrain" calls pretrain_backbone.py, and any other calls main_sota.py', choices=['None', 'pretrain', 'stnet', 'nn_baselines', "histogene"])
    parser.add_argument('--img_backbone',               type=str,           default='ShuffleNetV2',             help='Backbone to use for image encoding.', choices=['resnet', 'ConvNeXt', 'MobileNetV3', 'ResNetXt', 'ShuffleNetV2', 'ViT', 'WideResNet', 'densenet', 'swin'])
    parser.add_argument('--img_use_pretrained',         type=str2bool,      default=True,                       help='Whether or not to use imagenet1k pretrained weights in image backbone.')
    parser.add_argument('--pretrained_ie_path',         type=str,           default='None',                     help='Path of a pretrained image encoder model to start from the contrastive model.')
    parser.add_argument('--freeze_img_encoder',         type=str2bool,      default=False,                      help='Whether to freeze the image encoder. Only works when using pretrained model.')
    parser.add_argument('--act',                        type=str,           default='None',                     help='Activation function to use in the architecture. Case sensitive, options available at: https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity')
    parser.add_argument('--graph_operator',             type=str,           default='None',                     help='The convolutional graph operator to use. Case sensitive, options available at: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#convolutional-layers', choices=['GCNConv','SAGEConv','GraphConv','GATConv','GATv2Conv','TransformerConv', 'None'])
    parser.add_argument('--pos_emb_sum',                type=str2bool,      default=False,                      help='Whether or not to sum the nodes-feature with the positional embeddings. In case False, the positional embeddings are only concatenated.')
    parser.add_argument('--h_global',                   type=str2h_list,    default='//-1//-1//-1',             help='List of dimensions of the hidden layers of the graph convolutional network.')
    parser.add_argument('--pooling',                    type=str,           default='None',                     help='Global graph pooling to use at the end of the graph convolutional network. Case sensitive, options available at but must be a global pooling: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#pooling-layers')
    parser.add_argument('--dropout',                    type=float,         default=0,                        help='Dropout to use in the model to avoid overfitting.')
    # Train parameters #######################################################################################################################################################################
    parser.add_argument('--optim_metric',               type=str,           default='MSE',                      help='Metric that should be optimized during training.', choices=['PCC-Gene', 'MSE', 'MAE', 'Global'])
    parser.add_argument('--epochs',                     type=int,           default=350,                        help='Number of epochs to train de model.')
    parser.add_argument('--batch_size',                 type=int,           default=256,                        help='The batch size to train model.')
    parser.add_argument('--shuffle',                    type=str2bool,      default=True,                       help='Whether or not to shuffle the data in dataloaders.')
    parser.add_argument('--lr',                         type=float,         default=1e-3,                       help='Learning rate to use.')
    parser.add_argument('--optimizer',                  type=str,           default='Adam',                     help='Optimizer to use in training. Options available at: https://pytorch.org/docs/stable/optim.html It will just modify main optimizers and not sota (they have fixed optimizers).')
    parser.add_argument('--momentum',                   type=float,         default=0.9,                        help='Momentum to use in the optimizer if it receives this parameter. If not, it is not used. It will just modify main optimizers and not sota (they have fixed optimizers).')
    parser.add_argument('--average_test',               type=str2bool,      default=False,                      help='If True it will compute the 8 symmetries of an image during test and the prediction will be the average of the 8 outputs of the model.')
    parser.add_argument('--cuda',                       type=str,           default='0',                        help='CUDA device to run the model.')
    parser.add_argument('--exp_name',                   type=str,           default='None',                     help='Name of the experiment to save in the results folder. "None" will assign a date coded name.')
    parser.add_argument('--train',                      type=str2bool,      default=True,                       help='If true it will train, if false it only tests')
    parser.add_argument('--max_steps',                  type=int,           default=1000,                       help='Steps for training')
    parser.add_argument('--val_check_interval',         type=int,           default=10,                         help='Check in validation')
    #HIST2ST parameters
    parser.add_argument('--gpu', type=int, default=2, help='the id of gpu.')
    parser.add_argument('--fold', type=int, default=5, help='dataset fold.')
    parser.add_argument('--seed', type=int, default=12000, help='random seed.')
    parser.add_argument('--name', type=str, default='hist2ST', help='prefix name.')
    parser.add_argument('--logger', type=str, default='../logs/my_logs', help='logger path.')
    parser.add_argument('--bake', type=int, default=5, help='the number of augmented images.')
    parser.add_argument('--lamb', type=float, default=0.5, help='the loss coef of self-distillation.')
    parser.add_argument('--nb', type=str, default='F', help='zinb or nb loss.')
    parser.add_argument('--zinb', type=float, default=0.25, help='the loss coef of zinb.')
    parser.add_argument('--prune', type=str, default='NA', help='how to prune the edge:{"Grid","NA"}')
    parser.add_argument('--policy', type=str, default='mean', help='the aggregation way in the GNN .')
    parser.add_argument('--neighbor', type=int, default=4, help='the number of neighbors in the GNN.')
    parser.add_argument('--tag', type=str, default='5-7-2-8-4-16-32', 
                    help='hyper params: kernel-patch-depth1-depth2-depth3-heads-channel,'
                         'depth1-depth2-depth3 are the depth of Convmixer, Multi-head layer in Transformer, and GNN, respectively'
                         'patch is the value of kernel_size and stride in the path embedding layer of Convmixer'
                         'kernel is the kernel_size in the depthwise of Convmixer module'
                         'heads are the number of attention heads in the Multi-head layer'
                         'channel is the value of the input and output channel of depthwise and pointwise. ')

    ##########################################################################################################################################################################################

    return parser

def divide_patches_generalized(coords, n=2):

    # Obtener valores Ãºnicos y ordenados
    x_unique = np.sort(coords["x_coord"].unique())  
    y_unique = np.sort(coords["y_coord"].unique())[::-1]  

    # Mapear coordenadas a Ã­ndices
    x_to_idx = {val: i for i, val in enumerate(x_unique)}
    y_to_idx = {val: i for i, val in enumerate(y_unique)}

    # Definir tamaÃ±o de la cuadrÃ­cula
    grid_width = len(x_unique)
    grid_height = len(y_unique)

    # Calcular tamano de cada cuadrante
    step_x = grid_width // n
    step_y = grid_height // n

    # Diccionario para almacenar cuadrantes
    index_quadrants = {f"Q{r * n + c + 1}": [] for r in range(n) for c in range(n)}

    # Asignar parches a cuadrantes
    for (x, y), index in zip(coords.values, coords.index):
        x_idx = x_to_idx[x]
        y_idx = y_to_idx[y]

        # Determinar el cuadrante
        row = min(y_idx // step_y, n - 1)  # Evitar salir del rango por redondeo
        col = min(x_idx // step_x, n - 1)
        quadrant = f"Q{row * n + col + 1}"
        
        index_quadrants[quadrant].append(index)
    
    # Revisamos si todos los quadrantes tienen al menos un spot, sino los borramos
    index_quadrants = {k: v for k, v in index_quadrants.items() if v}
    # Revisamos si cada cuadrante queda con por lo menos 5 spots en el, si esto no ocurre justamos datos
    for key, indexes in list(index_quadrants.items()):
        if len(indexes) < 5:
            subimage_id = int(key.split('Q')[-1])
            # Con esto se obtiene el nuevo id
            # Evaluamos si tengo que sumarselo al siguiente cuadrante, si es el caso de es 'Q_n^2' entonces los añado en 'Q_n^2 - 1'
            new_subimage_id = subimage_id + 1 if subimage_id < n**2 else subimage_id - 1
            # Agregamos la info y luego hacemos sort
            index_quadrants[f'Q{new_subimage_id}'] += indexes
            index_quadrants[f'Q{new_subimage_id}'] = sorted(index_quadrants[f'Q{new_subimage_id}'])
            # Borramos la key con menos spots del umbral
            index_quadrants.pop(key, None)

    return index_quadrants


def get_divided_adata(adata, n=2):
    slides = adata.obs["slide_id"].unique().tolist()
    adatas_concat = []

    for slide in slides:
        adata_slide = adata[adata.obs["slide_id"]==slide]
        # Extract coords
        coord =  pd.DataFrame(adata_slide.obsm['spatial'], columns=['x_coord', 'y_coord'])
        coords_index = coord.index
        # Get indices per quadrant
        #index_quadrants = divide_slide(coords=coord)
        index_quadrants = divide_patches_generalized(coords=coord, n=n)
        # Get mini adatas
        for i in index_quadrants.keys():
            filtered_adata = adata_slide[index_quadrants[i]]
            filtered_adata.obs["slide_id"] = f"{slide}_{i}"
            adatas_concat.append(filtered_adata)
    
    merged_adata = ad.concat(adatas_concat, join="outer")
    return merged_adata

def train_simple(model, loader, criterion, optimizer, transforms):
    
    # Find the key of dataset.obsm that contains the patches
    patch_key = [k for k in loader.dataset._view_attrs_keys['obsm'] if 'patches' in k]
    # Assert that there is only one key
    assert len(patch_key) == 1, 'There should be only one key with patches in data.obsm'
    patch_key = patch_key[0]

    # Set model to train mode
    model.train()

    # for data in loader:
    # Rewrite the above for loop with tqdm
    for data in tqdm(loader, desc='Training'):
        # Get images from batch
        tissue_tiles = data.obsm[patch_key]
        w = round(np.sqrt(tissue_tiles.shape[1]/3))
        tissue_tiles = tissue_tiles.reshape((tissue_tiles.shape[0], w, w, -1))
        # Permute dimensions to be in correct order for normalization
        tissue_tiles = tissue_tiles.permute(0,3,1,2).contiguous()
        # Make transformations in tissue tiles
        tissue_tiles = tissue_tiles/255.
        # Transform tiles
        tissue_tiles = transforms(tissue_tiles)
        # Get output of the model
        expression_pred = model(tissue_tiles)
        
        # Get groundtruth of expression
        expression_gt = data.X

        # Compute expression MSE loss
        loss = criterion(expression_gt, expression_pred)
    
        # Do backwards
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test_simple_and_save_output(model, loader, criterion, transforms):
    
    # Find the key of dataset.obsm that contains the patches
    patch_key = [k for k in loader.dataset._view_attrs_keys['obsm'] if 'patches' in k]
    # Assert that there is only one key
    assert len(patch_key) == 1, 'There should be only one key with patches in data.obsm'
    patch_key = patch_key[0]

    # Initialize loss in 0.0
    loss = 0.

    # Global variables to be used
    glob_expression_gt = None 
    glob_expression_pred = None
    glob_mask = None
    glob_ids = None

    # Set model to eval mode
    model.eval()

    with torch.no_grad():
        for data in tqdm(loader, desc='Testing'):
            # Get images from batch
            tissue_tiles = data.obsm[patch_key]
            tissue_tiles = tissue_tiles.reshape((tissue_tiles.shape[0], round(np.sqrt(tissue_tiles.shape[1]/3)), round(np.sqrt(tissue_tiles.shape[1]/3)), -1))
            # Permute dimensions to be in correct order for normalization
            tissue_tiles = tissue_tiles.permute(0,3,1,2).contiguous()
            # Make transformations in tissue tiles
            tissue_tiles = tissue_tiles/255.
            tissue_tiles = transforms(tissue_tiles)

            # Get expression from batch
            expression_gt = data.X

            # Get the mask of the batch
            mask = torch.Tensor(data.layers['mask']).to(expression_gt.device)

            # Get output of the model
            # If tissue tiles is tuple then we will compute outputs of the 8 symmetries and then average them for prediction
            if isinstance(tissue_tiles, tuple):
                pred_list = [model(tissue_rot) for tissue_rot in tissue_tiles]
                pred_stack = torch.stack(pred_list)
                expression_pred = pred_stack.mean(dim=0)
            # If tissue tiles is not tuple then a single prediction is done with the original image
            else:
                expression_pred = model(tissue_tiles)

            # Concat batch to get global predictions and IDs
            glob_expression_gt = expression_gt if glob_expression_gt is None else torch.cat((glob_expression_gt, expression_gt))
            glob_expression_pred = expression_pred if glob_expression_pred is None else torch.cat((glob_expression_pred, expression_pred))
            glob_mask = mask if glob_mask is None else torch.cat((glob_mask, mask))
            glob_ids = data.obs['unique_id'].tolist() if glob_ids is None else glob_ids + data.obs['unique_id'].tolist()

            # Compute expression reconstruction loss and do backwards
            curr_loss = criterion(expression_gt, expression_pred)
            
            # Accumulate loss
            loss += curr_loss
    
    # Average loss
    loss = loss/loader.dataset.n_obs
    
    # If the adata object has a used mean attribute then we will use it to unnormalize the data
    general_adata = loader.dataset.adatas[0]
    if 'used_mean' in general_adata.var.keys():
        means = loader.dataset.adatas[0].var['used_mean'].values
        # Pass means to torch tensor in the same device as the model
        means = torch.tensor(means, device=glob_expression_gt.device)
        # Unnormalize data and predictions
        glob_expression_gt = glob_expression_gt+means
        glob_expression_pred = glob_expression_pred+means

    # Get metric dict 
    metric_dict = get_metrics(glob_expression_gt, glob_expression_pred, glob_mask.bool())
    
    # Declare an output dict of the model outputs and input
    output_dict = { 'expression': glob_expression_gt,
                    'img_reconstruction': glob_expression_pred,
                    'mask': glob_mask,
                    'ids': glob_ids}

    # Return losses
    return metric_dict, output_dict

def test_graph_and_save_output(model, loader, device):

    all_preds = []
    all_labels = []
    
    model.eval()

    for _, batch in enumerate(loader):
        batch.to(device)
        gnn_pred = model(batch)
        batch_pred = batch.predictions[batch.ptr[:-1]]
        pred = gnn_pred + batch_pred

        # Get labels
        layer = batch.y
        labels = layer[batch.ptr[:-1]]
        
        # Compute metrics
        all_preds.append(pred)
        all_labels.append(labels)

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Handle the case of predicting deltas
    if hasattr(batch, 'used_mean'):
        all_preds = all_preds + batch.used_mean[:batch.y.shape[1]]
        all_labels = all_labels + batch.used_mean[:batch.y.shape[1]]

    metrics = get_metrics(all_labels, all_preds)

    return metrics, all_labels, all_preds

# FIXME: This is a temporary function to test the model with the binary mask
def test_graph_and_save_output_w_mask(model, loader, glob_mask, device):

    all_preds = []
    all_labels = []
    
    model.eval()

    for _, batch in enumerate(loader):
        batch.to(device)
        gnn_pred = model(batch)
        batch_pred = batch.predictions[batch.ptr[:-1]]
        pred = gnn_pred + batch_pred

        # Get labels
        layer = batch.y
        labels = layer[batch.ptr[:-1]]
        
        # Compute metrics
        all_preds.append(pred)
        all_labels.append(labels)

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Handle the case of predicting deltas
    if hasattr(batch, 'used_mean'):
        all_preds = all_preds + batch.used_mean[:batch.y.shape[1]]
        all_labels = all_labels + batch.used_mean[:batch.y.shape[1]]

    metrics = get_metrics(all_labels, all_preds, glob_mask.to(device).bool())

    return metrics, all_labels, all_preds


def update_save_metric_df(metric_df, epoch, train_metric_dict, val_metric_dict, path):
    
    # Copy train and val metric dict
    train_dict = train_metric_dict.copy()
    val_dict = val_metric_dict.copy()

    # Remove keys that can not be stored in dataframe
    train_dict.pop('pearson_series', None)
    val_dict.pop('pearson_series', None)

    # Put train and val prefixes in dicts
    train_dict = {f'train_{key}': [val] for key, val in train_dict.items()}
    val_dict = {f'val_{key}': [val] for key, val in val_dict.items()}

    # Merge the two dictionaries
    df_dict = train_dict | val_dict

    # Log with wandb
    wandb.log({key: val[0] for key, val in df_dict.items()})

    # Add the epoch key at to the df_dict
    df_dict['epoch'] =  [epoch]

    # Get current dataframe
    curr_df = pd.DataFrame(df_dict)

    # Put epoch in the beginning of the dataframe
    epoch_col = curr_df.pop('epoch')
    curr_df.insert(0, 'epoch', epoch_col) 

    # If this is the first step then declare metric_df and save with header
    if metric_df is None:
        metric_df = curr_df
        metric_df.to_csv(path)
    # If not just append the last line to metric_df and save last line to csv
    else:
        metric_df = pd.concat([metric_df, curr_df], ignore_index=True)
        metric_df.iloc[[-1], :].to_csv(path, mode='a', header=None)

    # Define and refine printing string
    print_str = f'Epoch {epoch} '

    for key in sorted(df_dict):
        if key == 'epoch':
            continue
        if (key == 'train_mean_pearson') or (key == 'val_mean_pearson'):
            print_str = print_str + f'|{key} = {round(df_dict[key][0], 4)}'
        else:
            print_str = print_str + f'|{key} = {round(df_dict[key][0], 1)}'

    # Print progress in terminal
    print(print_str)

    return metric_df

def plot_metrics(metric_df_path, ref_mse_train, ref_mse_test, path):

    metric_df = pd.read_csv(metric_df_path, index_col=0)
    best_epoch_df = metric_df[metric_df.test_mean_pearson == metric_df.test_mean_pearson.max()]

    # Start figure
    fig, ax = plt.subplots(ncols=3, nrows=1)
    fig.set_size_inches((15,5))
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    
    # Plot MSE losses 
    img_loss_cols = metric_df.columns[['img_loss' in col for col in metric_df.columns]]
    exp_loss_cols = metric_df.columns[['exp_loss' in col for col in metric_df.columns]]
    # Plot img_losses and exp_losses
    metric_df.plot(x='epoch', y=img_loss_cols, style = ['--', '-'], ax=ax[0], c='k', legend=False, grid=True, ylabel='Reconstruction MSE')
    metric_df.plot(x='epoch', y=exp_loss_cols, style = ['--', '-'], ax=ax[0], c='r', legend=False, grid=True)
    # Plot lines with the nearest neighbor distance information
    range_list = [metric_df.epoch.min(), metric_df.epoch.max()]
    ax[0].plot(range_list, [ref_mse_train['median'], ref_mse_train['median']], '--', color='darkgreen', label='Mean MSE NN train')
    ax[0].plot(range_list, [ref_mse_test['median'], ref_mse_test['median']], '-', color='darkgreen', label='Mean MSE NN test')
    # Plot best performance dot
    ax[0].scatter(best_epoch_df.epoch.item(), best_epoch_df.test_img_loss.item(), s=20, c='b')
    ax[0].scatter(best_epoch_df.epoch.item(), best_epoch_df.test_exp_loss.item(), s=20, c='b')
    # Put text of best performance
    textstr = f'Best performance: \nEpoch: {best_epoch_df.epoch.item()}\nImg loss test = {round(best_epoch_df.test_img_loss.item(),1)}\nExp loss test = {round(best_epoch_df.test_exp_loss.item(),1)}'
    ax[0].text(0.03, 0.80, textstr, transform=ax[0].transAxes, bbox=props)
    # Define custom legend
    legend_elements = [
        Line2D([0], [0], color='k', ls='-', label='Test'),
        Line2D([0], [0], color='k', ls='--', label='Train'),
         Patch(facecolor='k', edgecolor=None, label='Img loss'),
         Patch(facecolor='r', edgecolor=None, label='Exp loss'),
         Patch(facecolor='darkgreen', edgecolor=None, label='Median MSE NN')]
    # Format axis
    ax[0].legend(handles=legend_elements, loc='upper right', fontsize=8)
    ax[0].set_title('Reconstruction Losses')

    # Plot SimCLR losses
    cl_loss_cols = metric_df.columns[['cl_loss' in col for col in metric_df.columns]]
    metric_df.plot(x='epoch', y=cl_loss_cols, style = ['--', '-'], ax=ax[1], c='k', legend=False, grid=True, ylabel='SimCLR Loss')
    ax[1].scatter(best_epoch_df.epoch.item(), best_epoch_df.test_cl_loss.item(), s=20, c='b')
    # Define best performance text
    textstr = f'Best performance: \nEpoch: {best_epoch_df.epoch.item()}\nSimCLR test = {round(best_epoch_df.test_cl_loss.item(),1)}'
    ax[1].text(0.03, 0.85, textstr, transform=ax[1].transAxes, bbox=props)
    ax[1].set_title('Contrastive Losses')
    ax[1].set_ylim([0, None])

    # Plot pearson correlation 
    pearson_cols = metric_df.columns[['mean_pearson' in col for col in metric_df.columns]]
    metric_df.plot(x='epoch', y=pearson_cols, style = ['--', '-'], ax=ax[2], c='k', legend=False, grid=True, ylabel='Pearson Correlation')
    ax[2].scatter(best_epoch_df.epoch.item(), best_epoch_df.test_mean_pearson.item(), s=20, c='b')
    # Define best performance text
    textstr = f'Best performance: \nEpoch: {best_epoch_df.epoch.item()}\nPearson test = {round(best_epoch_df.test_mean_pearson.item(),4)}'
    ax[2].text(0.03, 0.85, textstr, transform=ax[2].transAxes, bbox=props)
    ax[2].set_title('Expression Correlation Metric')

    for axes in ax:
        axes.spines[['right', 'top']].set_visible(False)
        axes.set_xlabel('Epochs')
        axes.set_xlim(range_list)


    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def update_plot_prediction_layer(visium_dataset, metric_dict, output_dict, path):
    
    # Get gene names and id list
    gene_names = metric_dict['pearson_series'].index
    id_list = output_dict['ids']
    # Declare the prediction dataframe
    prediction_df = pd.DataFrame(data=output_dict['img_reconstruction'], index=id_list, columns=gene_names)
    
    # Get the ID ordering of the Visium dataset
    correct_id_order = visium_dataset.data.obs_names

    # Sort prediction dataframe to be the same as the groundtruth
    prediction_df = prediction_df.loc[correct_id_order]

    # Update best prediction layer in visium_dataset
    visium_dataset.data.layers['best_prediction'] = prediction_df

    # Get the best 2 and worst 2 predicted genes
    best_2_genes = metric_dict['pearson_series'].nlargest(2).index.tolist()
    worst_2_genes = metric_dict['pearson_series'].nsmallest(2).index.tolist()

    # List of plotting genes
    plotting_genes = best_2_genes + worst_2_genes
    plotting_pearson = metric_dict['pearson_series'][plotting_genes].tolist()

    # Make figure
    fig, ax = plt.subplots(nrows=2, ncols=4)
    fig.set_size_inches(13, 6.5)

    # Cycle plotting
    for i in range(len(plotting_genes)):
        # Define title color
        tit_color = 'g' if i<2 else 'r'

        # Define the normalization to have the same color range in groundtruth and prediction
        gt_min, gt_max = visium_dataset.data[:, [plotting_genes[i]]].X.min(), visium_dataset.data[:, [plotting_genes[i]]].X.max() 
        pred_min, pred_max = prediction_df[plotting_genes[i]].min(), prediction_df[plotting_genes[i]].max()
        vmin, vmax = min([gt_min, pred_min]), max([gt_max, pred_max])
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

        # Plot the groundtruth
        sq.pl.spatial_scatter(visium_dataset.data, color=[plotting_genes[i]], ax=ax[0,i], norm=norm, cmap='jet')
        ax[0,i].set_title(f'{plotting_genes[i]} GT', color=tit_color)

        # Plot the prediction
        sq.pl.spatial_scatter(visium_dataset.data, color=[plotting_genes[i]], ax=ax[1,i], layer='best_prediction', norm=norm, cmap='jet')
        ax[1,i].set_title(f'{plotting_genes[i]} Pred: PCC $= {round(plotting_pearson[i],3)}$', color=tit_color)
    
    # Format figure
    for axis in ax.flatten():
        axis.set_xlabel('')
        axis.set_ylabel('')
    
    fig.suptitle('Best 2 (left) and Worst 2 (right) Predicted Genes', fontsize=20)
    fig.tight_layout()
    # Save plot 
    fig.savefig(path, dpi=300)
    plt.close(fig)

def tensor_2_np(tens):
    return tens.detach().cpu().numpy()

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True    

class EightSymmetry(object):
    """Returns a tuple of the eight symmetries resulting from rotation and reflection.
    
    This behaves similarly to TenCrop.
    This transform returns a tuple of images and there may be a mismatch in the number of inputs and targets your Dataset returns. See below for an example of how to deal with this.
    Example:
     transform = Compose([
         EightSymmetry(), # this is a tuple of PIL Images
         Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])) # returns a 4D tensor
     ])
    """
    # This class function was taken fron the original ST-Net repository at:
    # https://github.com/bryanhe/ST-Net/blob/43022c1cb7de1540d5a74ea2338a12c82491c5ad/stnet/transforms/eight_symmetry.py#L3
    def __call__(self, img):
        identity = lambda x: x
        ans = []
        for i in [identity, torchvision.transforms.RandomHorizontalFlip(1)]:
            for j in [identity, torchvision.transforms.RandomVerticalFlip(1)]:
                for k in [identity, torchvision.transforms.RandomRotation((90, 90))]:
                    ans.append(i(j(k(img))))
        return tuple(ans)

    def __repr__(self):
        return self.__class__.__name__ + "()"


# To test the code
if __name__=='__main__':
    hello = 0