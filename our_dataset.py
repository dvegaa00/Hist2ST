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
from graph_construction import calcADJ
from collections import defaultdict as dfd
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

class OUR_BUILDER(torch.utils.data.Dataset):
    """This class load our Data and returns the batch that the model was build to receive"""
    def __init__(self,adata:ad.AnnData,prediction="c_d_log1p",patch_size=224,prune='NA',neighs=4,id2name={}):
        super(OUR_BUILDER, self).__init__()
        """Receives an AnnData and generates a dictionary to save the patches, positions, prediction layer, adjancency matrxi
            counts of the original expression, cpunts divided by the median, center coordinates and mask
            
            Args: 
                adata (AnnData): adata from data
                prediction (str): layer used for prediction
                patch_size (int): the size of the patch defined in the adata
                prune (str): grid used for the adjancency matrxi (NA: take all defined neighbors)
                neighs (int): neighbors considered in adjancency matrix
                id2name (dict): dictionary to pass from an ID to the name of the slide
            Returns:
                __get_item__ 
                
        """
        self.adata = adata
        self.prediction = prediction
        self.patch_size = patch_size
        self.id2name = id2name
        
        names = self.adata.obs["slide_id"].unique().tolist()
        # Gene_list = self.adata.var["gene_symbol"].tolist() #opción 2: adata.var["gene_ids"].tolist()

        print('Loading data...')
        
        # Define dictionaries of batch information
        self.exp_dict = {}
        self.ori_dict = {}
        self.center_dict = {}
        self.loc_dict = {}
        self.counts_dict = {}
        self.adj_dict = {}
        self.patch_dict = {}
        self.mask_dict = {}
        
        # Get information from all the slides and save in dictionary
        for n in names: 
            adata_slide = self.adata[self.adata.obs["slide_id"]==n]
            # Calculate all batch information
            # Expression prediction layer
            self.exp_dict[n] = adata_slide.layers[prediction] 
            # Original expresion
            self.ori_dict[n] = adata_slide.layers["counts"] 
            # Pixels x and y 
            self.center_dict[n] = adata_slide.obsm["spatial"] 
            x = adata_slide.obs["array_col"]
            y = adata_slide.obs["array_row"]
            # Center (x,y)
            self.loc_dict[n] = np.stack((x, y), axis=-1) 
            self.counts_dict[n] = self.ori_dict[n].sum(1)
            # Counts (original matrix expression) divided by median
            self.counts_dict[n] =  self.counts_dict[n] / np.median(self.counts_dict[n]) 
            # Adjacency matrix of patches
            self.adj_dict[n] = calcADJ(self.loc_dict[n],neighs,pruneTag=prune) 
            # Obtain patch and unflatten it
            #FIXME: robusto a otras escalas
            flat_patches = adata_slide.obsm["patches_scale_1.0"] #patches
            patches = flat_patches.reshape((-1,self.patch_size, self.patch_size,3))
            self.patch_dict[n] = patches
            # Obtain mask
            self.mask_dict[n] = adata_slide.layers['mask']

    def __getitem__(self, index):
        """Summary: get item in dataloader
        """
        # Define slide based on ID and save all batch information
        ID=self.id2name[index]
        exps = self.exp_dict[ID]
        oris = self.ori_dict[ID]
        sfs = self.counts_dict[ID]
        adj = self.adj_dict[ID]
        centers = self.center_dict[ID]
        loc = self.loc_dict[ID]
        patches = self.patch_dict[ID]
        mask = self.mask_dict[ID]
        
        # Convert variables to torch type
        positions = torch.LongTensor(loc)
        exps = torch.Tensor(exps)
        oris = torch.Tensor(oris)
        sfs = torch.Tensor(sfs)
        adj = torch.Tensor(adj)
        centers = torch.Tensor(centers)
        patches = torch.FloatTensor(patches)
        patches = patches.permute(0,3,1,2)
        
        # Save data to return as a single batch
        data=[patches, positions, exps, adj, oris, sfs, centers, mask]
        return data
    
    def __len__(self):
        return len(self.exp_dict)
