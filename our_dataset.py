import os
import glob
import torch
import torchvision
import numpy as np
import scanpy as sc
import pandas as pd 
import scprep as scp
import anndata as ad
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import ImageFile, Image
from utils import read_tiff, get_data
from graph_construction import calcADJ
from collections import defaultdict as dfd
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

class OUR_BUILDER(torch.utils.data.Dataset):
    """Some Information about ViT_SKIN"""
    def __init__(self,adata:ad.AnnData,prediction="c_d_log1p",patch_size=224,prune='NA',neighs=4,id2name=[]):
        super(OUR_BUILDER, self).__init__()
        
        self.adata = adata
        self.prediction = prediction
        self.patch_size = patch_size
        self.id2name = id2name
        
        names = self.adata.obs["slide_id"].unique().tolist()
        gene_list = self.adata.var["gene_symbol"].tolist() #opción 2: adata.var["gene_ids"].tolist()

        print('Loading data...')
        
        #define dictionaries of batch informatio
        self.exp_dict = {}
        self.ori_dict = {}
        self.center_dict = {}
        self.loc_dict = {}
        self.counts_dict = {}
        self.adj_dict = {}
        self.patch_dict = {}
        
        for n in names: 
            adata_slide = self.adata[self.adata.obs["slide_id"]==n]
            #calculate all batch information
            self.exp_dict[n] = adata_slide.layers[prediction] #expression prediction
            self.ori_dict[n] = adata_slide.layers["counts"] #original expresion
            self.center_dict[n] = adata_slide.obsm["spatial"] #pixels x and y
            x = adata_slide.obs["array_col"]
            y = adata_slide.obs["array_row"]
            self.loc_dict[n] = np.stack((x, y), axis=-1) #center (x,y)
            self.counts_dict[n] = self.ori_dict[n].sum(1)
            self.counts_dict[n] =  self.counts_dict[n] / np.median(self.counts_dict[n]) #counts (original matrix expression) divided by median
            self.adj_dict[n] = calcADJ(self.loc_dict[n],neighs,pruneTag=prune) #adjacency of patches
            #change patche dimensions
            flat_patches = adata_slide.obsm["patches_scale_1.0"] #patches
            patches = flat_patches.reshape((-1,self.patch_size, self.patch_size,3))
            self.patch_dict[n] = patches

    def __getitem__(self, index):
        ID=self.id2name[index]
        exps = self.exp_dict[ID]
        oris = self.ori_dict[ID]
        sfs = self.counts_dict[ID]
        adj = self.adj_dict[ID]
        centers = self.center_dict[ID]
        loc = self.loc_dict[ID]
        patches = self.patch_dict[ID]
        
        positions = torch.LongTensor(loc)
        exps = torch.Tensor(exps)
        oris = torch.Tensor(oris)
        sfs = torch.Tensor(sfs)
        adj = torch.Tensor(adj)
        centers = torch.Tensor(centers)
        patches = torch.FloatTensor(patches)
        patches = patches.permute(0,3,1,2)
        
        #for p in patches:
        
        data=[patches, positions, exps, adj, oris, sfs, centers]
        return data
    
    def __len__(self):
        return len(self.exp_dict)
"""               
    def get_img(self,adata:ad.AnnData):
        path = adata.uns["spatial"][name]["metadata"]["source_image_path"]
        im = Image.open(path)
        return im

    def get_cnt(self,adata:ad.AnnData):
        list_df = adata.layers["counts"]
        df = pd.DataFrame(list_df, columns=self.gene_list)
        return df

    def get_pos(self,adata:ad.AnnData):
        list_df = adata.obsm["spatial"]
        df = pd.DataFrame(list_df, columns=["pixel_x", "pixel_y"])
        df["x"] = adata.obs["array_col"]
        df["y"] = adata.obs["array_row"]
        x = df['pixel_x'].values
        y = df['pixel_y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
   
        id = np.arange(0,len(x))
        df['id'] = id
        return df

    def get_meta(self,adata:ad.AnnData,gene_list=None):
        cnt = self.get_cnt(adata)
        pos = self.get_pos(adata)
        meta = cnt.join(pos.set_index('id'),how='inner')

        return meta
"""