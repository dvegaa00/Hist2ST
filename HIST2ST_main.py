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
from utils_sepal import *
from HIST2ST import *
from predict import *
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from our_dataset import OUR_BUILDER
from spared.datasets import get_dataset
from spared.metrics import get_metrics

#torch.cuda.empty_cache() 
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
#CUDA_LAUNCH_BLOCKING = "1"
parser = get_main_parser()
args = parser.parse_args()

args.cuda = os.environ["CUDA_VISIBLE_DEVICES"]
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

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

#Split data to train val and test and creat an ID to Name list for the get item
train_split = dataset.adata[dataset.adata.obs["split"]=="train"]
train_slides = train_split.obs["slide_id"].unique().tolist()
slide_names = train_slides.copy()

val_split = dataset.adata[dataset.adata.obs["split"]=="val"]
val_slides = val_split.obs["slide_id"].unique().tolist()
slide_names += val_slides

test_split = dataset.adata[dataset.adata.obs["split"]=="test"]
test_slides = test_split.obs["slide_id"].unique().tolist()
slide_names += test_slides

#ID to Name list
id2name = dict(enumerate(slide_names))

#Our dataser builder
custom_dataset = OUR_BUILDER(adata=dataset.adata, prediction=args.prediction_layer, patch_size=224, prune=args.prune, neighs=args.neighbor, id2name=id2name)

#Split the dataset
train_ids=[key for key, val in id2name.items() if val in train_slides]
val_ids=[key for key, val in id2name.items() if val in val_slides]
test_ids=[key for key, val in id2name.items() if val in test_slides]

trainset = torch.utils.data.Subset(custom_dataset, train_ids)
valset = torch.utils.data.Subset(custom_dataset, val_ids)
testset = torch.utils.data.Subset(custom_dataset, test_ids)

"""
trainset = OUR_BUILDER(adata=train_split, prediction=args.prediction_layer, patch_size=224, prune=args.prune, neighs=args.neighbor, id2name=id2name)
valset = OUR_BUILDER(adata=val_split, prediction=args.prediction_layer, patch_size=224, prune=args.prune, neighs=args.neighbor, id2name=id2name)
testset = OUR_BUILDER(adata=test_split, prediction=args.prediction_layer, patch_size=224, prune=args.prune, neighs=args.neighbor, id2name=id2name)
"""

#Load data
train_loader = DataLoader(trainset, batch_size=1, num_workers=0, shuffle=True)
val_loader = DataLoader(valset, batch_size=1, num_workers=0, shuffle=True)
test_loader = DataLoader(testset, batch_size=1, num_workers=0, shuffle=False)

gene_list = dataset.adata.var["gene_symbol"].tolist()
genes = len(gene_list)
 
"""
trainset = pk_load(args.fold,'train',False,args.data,neighs=args.neighbor, prune=args.prune)
train_loader = DataLoader(trainset, batch_size=1, num_workers=0, shuffle=True)
testset = pk_load(args.fold,'test',False,args.data,neighs=args.neighbor, prune=args.prune)
test_loader = DataLoader(testset, batch_size=1, num_workers=0, shuffle=False)
label=None
if args.fold in [5,11,17,23,26,30] and args.data=='her2st':
    label=testset.label[testset.names[0]]

genes=785
if args.data=='cscc':
    args.name+='_cscc'
    genes=171

log_name=''
if args.zinb>0:
    if args.nb=='T':
        args.name+='_nb'
    else:
        args.name+='_zinb'
    log_name+=f'-{args.zinb}'
if args.bake>0:
    args.name+='_bake'
    log_name+=f'-{args.bake}-{args.lamb}'
log_name=f'{args.fold}-{args.name}-{args.tag}'+log_name+f'-{args.policy}-{args.neighbor}'
logger = TensorBoardLogger(
    args.logger, 
    name=log_name
)
print(log_name)
"""

log_name=f'{args.name}-{args.dataset}-{args.lr}'
logger = TensorBoardLogger(
    args.logger, 
    name=log_name
)

model = Hist2ST(
    depth1=depth1, depth2=depth2, depth3=depth3,
    n_genes=genes, learning_rate=args.lr, label=label, 
    kernel_size=kernel, patch_size=patch,
    heads=heads, channel=channel, dropout=args.dropout,
    zinb=args.zinb, nb=args.nb=='T',
    bake=args.bake, lamb=args.lamb, 
    policy=args.policy, 
    fig_size = 224
)

trainer = pl.Trainer(
    max_steps=args.max_steps, 
    accelerator="gpu", 
    devices=2,
    logger=logger,
    check_val_every_n_epoch=2,
    enable_progress_bar=True)

trainer.fit(model, train_loader, test_loader)
torch.save(model.state_dict(),f"./model/{args.dataset}-Hist2ST.ckpt")
# model.load_state_dict(torch.load(f"./model/{args.fold}-Hist2ST{'_cscc' if args.data=='cscc' else ''}.ckpt"),)
breakpoint()
pred, gt = test(model, test_loader,'cuda')
mask = test_split.layers["mask"]
metrics = get_metrics(gt, pred, maks)
#R=get_R(pred,gt)[0]
#print('Pearson Correlation:',np.nanmean(R))
#clus,ARI=cluster(pred,label)
#print('ARI:',ARI)