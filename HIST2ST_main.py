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
from utils_sepal import *
from HIST2ST import *
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from our_dataset import OUR_BUILDER
from spared.datasets import get_dataset
from spared.metrics import get_metrics
import anndata as ad
from datetime import datetime
from log_functions import *

parser = get_main_parser()
args = parser.parse_args()
args_dict = vars(args)

args.cuda = os.environ["CUDA_VISIBLE_DEVICES"]
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Seed everything
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)  
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
kernel,patch,depth1,depth2,depth3,heads,channel=map(lambda x:int(x),args.tag.split('-'))
label=None


dataset = get_dataset(args.dataset)

# Split data to train val and test and creat an ID to Name list for the get item
if args.prediction_layer == 'noisy':
    # Copy the layer c_t_log1p to the layer noisy
    noisy_layer = dataset.adata.layers['c_t_log1p'].copy()
    # Get zero mask
    zero_mask = ~dataset.adata.layers['mask']
    # Zero out the missing values
    noisy_layer[zero_mask] = 0
    # Add the layer to the adata
    dataset.adata.layers['noisy'] = noisy_layer

train_split = dataset.adata[dataset.adata.obs["split"]=="train"]
train_slides = train_split.obs["slide_id"].unique().tolist()
slide_names = train_slides.copy()

val_split = dataset.adata[dataset.adata.obs["split"]=="val"]
val_slides = val_split.obs["slide_id"].unique().tolist()
slide_names += val_slides

if "test" in dataset.adata.obs["split"].unique().tolist():
    test_split = dataset.adata[dataset.adata.obs["split"]=="test"]
    test_slides = test_split.obs["slide_id"].unique().tolist()
    slide_names += test_slides
    
# ID to Name list
id2name = dict(enumerate(slide_names))

# Call our dataset builder
custom_dataset = OUR_BUILDER(adata=dataset.adata, prediction=args.prediction_layer, patch_size=224, prune=args.prune, neighs=args.neighbor, id2name=id2name)

# Split the dataset
train_ids=[key for key, val in id2name.items() if val in train_slides]
val_ids=[key for key, val in id2name.items() if val in val_slides]
if "test" in dataset.adata.obs["split"].unique().tolist():
    test_ids=[key for key, val in id2name.items() if val in test_slides]

trainset = torch.utils.data.Subset(custom_dataset, train_ids)
valset = torch.utils.data.Subset(custom_dataset, val_ids)
if "test" in dataset.adata.obs["split"].unique().tolist():    
    testset = torch.utils.data.Subset(custom_dataset, test_ids)

# Load data
train_loader = DataLoader(trainset, batch_size=1, num_workers=0, shuffle=True)
val_loader = DataLoader(valset, batch_size=1, num_workers=0, shuffle=True)
test_loader = None
if "test" in dataset.adata.obs["split"].unique().tolist():
    test_loader = DataLoader(testset, batch_size=1, num_workers=0, shuffle=False)

# Gene list (required as parameter in the model)
genes = dataset.adata.shape[1]

# Logger 
# If exp_name is None then generate one with the current time
if args.exp_name == 'None':
    args.exp_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# Start wandb configs  
wandb_logger = WandbLogger(
    project='spared_hist2st_sota',
    name=args.exp_name,
    log_model=False,
    entity="sepal_v2")
#TODO: cambiar sepal_v2 a Benckmark

# Get save path and create is in case it is necessary
save_path = os.path.join('results', args.exp_name)
os.makedirs(save_path, exist_ok=True)

# Save script arguments in json file
with open(os.path.join(save_path, 'script_params.json'), 'w') as f:
    json.dump(args_dict, f, indent=4)

# Declare model
model = Hist2ST(
    args=args, depth1=depth1, depth2=depth2, depth3=depth3,
    n_genes=genes, learning_rate=args.lr, label=label, 
    kernel_size=kernel, patch_size=patch,
    heads=heads, channel=channel, dropout=args.dropout,
    zinb=args.zinb, nb=args.nb=='T',
    bake=args.bake, lamb=args.lamb, 
    policy=args.policy, 
    fig_size = 224
)

# Define dict to know whether to maximize or minimize each metric
max_min_dict = {'PCC-Gene': 'max', 'PCC-Patch': 'max', 'MSE': 'min', 'MAE': 'min', 'R2-Gene': 'max', 'R2-Patch': 'max', 'Global': 'max'}
# Define checkpoint callback to save best model in validation
checkpoint_callback = ModelCheckpoint(
    dirpath=save_path,
    monitor=f'val_{args.optim_metric}', # Choose your validation metric
    save_top_k=0, # Save only the best model
    mode=max_min_dict[args.optim_metric], # Choose "max" for higher values or "min" for lower values
)

# Define the trainier and fit the model
trainer = L.Trainer(
    max_steps=args.max_steps, 
    accelerator="gpu", 
    devices=1,
    logger=wandb_logger,
    val_check_interval=args.val_check_interval,
    log_every_n_steps=args.val_check_interval,
    check_val_every_n_epoch=None,
    enable_progress_bar=True,
    enable_model_summary=True,
    callbacks=[checkpoint_callback]
    )

trainer.fit(model, train_loader, val_loader)
# Load the best model after training
model.load_state_dict(model.model_best_weights)
# Test model if there is a test dataloader
if not (test_loader is None):
    trainer.test(model, dataloaders=test_loader)

# Get global prediction layer and log final artifacts
get_predictions(adata=dataset.adata,
    args=args,
    model=model,
    layer=args.prediction_layer,
    device=device)

# Log prediction images
dataset.log_pred_image()
wandb.finish()

