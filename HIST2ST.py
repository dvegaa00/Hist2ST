import torch
import numpy as np
import pytorch_lightning as pl
import lightning as L
import torchvision.transforms as tf
from gcn import *
from NB_module import *
from transformer import *
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from copy import deepcopy as dcp
from collections import defaultdict as dfd
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
import anndata as ann
from sklearn.cluster import KMeans
from spared.metrics import get_metrics
from torch.distributed import rpc

class convmixer_block(nn.Module):
    def __init__(self,dim,kernel_size):
        super().__init__()
        self.dw=nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                nn.BatchNorm2d(dim),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                nn.BatchNorm2d(dim),
                nn.GELU(),
        )
        self.pw=nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )
        
    def forward(self,x):
        x=self.dw(x)+x
        x=self.pw(x)
        return x
class mixer_transformer(nn.Module):
    def __init__(self,channel=32, kernel_size=5, dim=1024,
                 depth1=2, depth2=8, depth3=4, 
                 heads=8, dim_head=64, mlp_dim=1024, dropout = 0.,
                 policy='mean',gcn=True
                ):
        super().__init__()
        self.layer1=nn.Sequential(
            *[convmixer_block(channel,kernel_size) for i in range(depth1)],
        )
        self.layer2=nn.Sequential(*[attn_block(dim,heads,dim_head,mlp_dim,dropout) for i in range(depth2)])
        self.layer3=nn.ModuleList([gs_block(dim,dim,policy,gcn) for i in range(depth3)])
        self.jknet=nn.Sequential(
            nn.LSTM(dim,dim,2),
            SelectItem(0),
        )
        self.down=nn.Sequential(
            nn.Conv2d(channel,channel//8,1,1),
            nn.Flatten(),
        )
    def forward(self,x,ct,adj):
        x=self.down(self.layer1(x))
        g=x.unsqueeze(0)
        g=self.layer2(g+ct).squeeze(0)
        jk=[]
        for layer in self.layer3:
            g=layer(g,adj)
            jk.append(g.unsqueeze(0))
        g=torch.cat(jk,0)
        g=self.jknet(g).mean(0)
        return g
class ViT(nn.Module):
    def __init__(self, channel=32,kernel_size=5,dim=1024, 
                 depth1=2, depth2=8, depth3=4, 
                 heads=8, mlp_dim=1024, dim_head = 64, dropout = 0.,
                 policy='mean',gcn=True
                ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.transformer = mixer_transformer(
            channel, kernel_size, dim, 
            depth1, depth2, depth3, 
            heads, dim_head, mlp_dim, dropout,
            policy,gcn,
        )

    def forward(self,x,ct,adj):
        x = self.dropout(x)
        x = self.transformer(x,ct,adj)
        return x

class Hist2ST(L.LightningModule):
    def __init__(self, args, learning_rate=1e-5, fig_size=112, label=None, 
                 dropout=0.2, n_pos=128, kernel_size=5, patch_size=7, n_genes=785, 
                 depth1=2, depth2=8, depth3=4, heads=16, channel=32, 
                 zinb=0, nb=False, bake=0, lamb=0, policy='mean', 
                ):
        super().__init__()
        # self.save_hyperparameters()
        dim=(fig_size//patch_size)**2*channel//8
        self.model_best_weights = None
        self.save_hyperparameters()
        self.state_dict()
        self.args = args
        
        self.learning_rate = learning_rate
        
        self.nb=nb
        self.zinb=zinb
        
        self.bake=bake
        self.lamb=lamb
        
        # Auxiliary variables to log best metrics
        self.best_metrics = None
        min_max_metric_dict = {'PCC-Gene': 'max', 'PCC-Patch': 'max', 'MSE': 'min', 'MAE': 'min', 'R2-Gene': 'max', 'R2-Patch': 'max', 'Global': 'max'}
        self.metric_objective = min_max_metric_dict[self.args.optim_metric]
        
        # Define outputs of the train, validation and test step
        self.validation_step_outputs = []
        self.training_step_outputs = []
        self.test_step_outputs = []
        
        self.label=label
        self.patch_embedding = nn.Conv2d(3,channel,patch_size,patch_size)
        self.x_embed = nn.Embedding(n_pos,dim)
        self.y_embed = nn.Embedding(n_pos,dim)
        self.vit = ViT(
            channel=channel, kernel_size=kernel_size, dim=dim, heads=heads,
            depth1=depth1,depth2=depth2, depth3=depth3, dim_head=n_pos,
            mlp_dim=dim, dropout = dropout, policy=policy, gcn=True,
        )
        self.channel=channel
        self.patch_size=patch_size
        self.n_genes=n_genes
        if self.zinb>0:
            if self.nb:
                self.hr=nn.Linear(dim, n_genes)
                self.hp=nn.Linear(dim, n_genes)
            else:
                self.mean = nn.Sequential(nn.Linear(dim, n_genes), MeanAct())
                self.disp = nn.Sequential(nn.Linear(dim, n_genes), DispAct())
                self.pi = nn.Sequential(nn.Linear(dim, n_genes), nn.Sigmoid())
        if self.bake>0:
            self.coef=nn.Sequential(
                nn.Linear(dim,dim),
                nn.ReLU(),
                nn.Linear(dim,1),
            )
        self.gene_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_genes),
        )
        self.tf=tf.Compose([
            tf.RandomGrayscale(0.1),
            tf.RandomRotation(90),
            tf.RandomHorizontalFlip(0.2),
        ])
    def forward(self, patches, centers, adj, aug=False):
        B,N,C,H,W=patches.shape
        patches=patches.reshape(B*N,C,H,W)
        patches = self.patch_embedding(patches)
        centers_x = self.x_embed(centers[:,:,0])
        centers_y = self.y_embed(centers[:,:,1])
        ct=centers_x + centers_y
        h = self.vit(patches,ct,adj)
        x = self.gene_head(h)
        extra=None
        if self.zinb>0:
            if self.nb:
                r=self.hr(h)
                p=self.hp(h)
                extra=(r,p)
            else:
                m = self.mean(h)
                d = self.disp(h)
                p = self.pi(h)
                extra=(m,d,p)
        if aug:
            h=self.coef(h)
        return x,extra,h
    def aug(self,patch,center,adj):
        bake_x=[]
        for i in range(self.bake):
            new_patch=self.tf(patch.squeeze(0)).unsqueeze(0)
            x,_,h=self(new_patch,center,adj,True)
            bake_x.append((x.unsqueeze(0),h.unsqueeze(0)))
        return bake_x
    def distillation(self,bake_x):
        new_x,coef=zip(*bake_x)
        coef=torch.cat(coef,0)
        new_x=torch.cat(new_x,0)
        coef=F.softmax(coef,dim=0)
        new_x=(new_x*coef).sum(0)
        return new_x
    def training_step(self, batch, batch_idx):
        patch, center, exp, adj, oris, sfs, *_, mask = batch
        adj=adj.squeeze(0)
        exp=exp.squeeze(0)
        mask = mask.squeeze(0)
        pred,extra,h = self(patch, center, adj)
        
        mse_loss = F.mse_loss(pred, exp)
        #self.log('mse_loss', mse_loss,on_epoch=True, prog_bar=True, logger=True)
        bake_loss=0
        if self.bake>0:
            bake_x=self.aug(patch,center,adj)
            new_pred=self.distillation(bake_x)
            bake_loss+=F.mse_loss(new_pred,pred)
            #self.log('bake_loss', bake_loss,on_epoch=True, prog_bar=True, logger=True)
        zinb_loss=0
        if self.zinb>0:
            if self.nb:
                r,p=extra
                zinb_loss = NB_loss(oris.squeeze(0),r,p)
            else:
                m,d,p=extra
                zinb_loss = ZINB_loss(oris.squeeze(0),m,d,p,sfs.squeeze(0))
            #self.log('zinb_loss', zinb_loss,on_epoch=True, prog_bar=True, logger=True)
            
        loss=mse_loss+self.zinb*zinb_loss+self.lamb*bake_loss
        #Log train loss
        train_log_dict = {'train_loss': loss}
        self.log_dict(train_log_dict, on_step=True)
        
        self.training_step_outputs.append((pred, exp, mask))
        
        return loss
    
    def on_train_epoch_end(self):
        # Unpack the list of tuples
        glob_expression_pred, glob_expression_gt, glob_mask = zip(*self.training_step_outputs)
        # Concatenate outputs along the sample dimension
        glob_expression_pred, glob_expression_gt, glob_mask = torch.cat(glob_expression_pred), torch.cat(glob_expression_gt), torch.cat(glob_mask)

        # Get metrics and log
        metrics = get_metrics(glob_expression_gt, glob_expression_pred, glob_mask)
        # Put train prefix in metric dict
        metrics = {f'train_{k}': v for k, v in metrics.items()}
        self.log_dict(metrics, on_epoch=True)
        # Free memory
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        patch, center, exp, adj, oris, sfs, *_, mask = batch
        def cluster(pred,cls):
            sc.pp.pca(pred)
            sc.tl.tsne(pred)
            kmeans = KMeans(n_clusters=cls, init="k-means++", random_state=0).fit(pred.obsm['X_pca'])
            pred.obs['kmeans'] = kmeans.labels_.astype(str)
            p=pred.obs['kmeans'].to_numpy()
            return p
        
        exp=exp.squeeze(0)
        mask = mask.squeeze(0)
        pred,extra,h = self(patch, center, adj.squeeze(0))
        if self.label is not None:
            adata=ann.AnnData(pred.squeeze().cpu().numpy())
            idx=self.label!='undetermined'
            cls=len(set(self.label))
            x=adata[idx]
            l=self.label[idx]
            predlbl=cluster(x,cls-1)
            #self.log('nmi',nmi_score(predlbl,l))
            #self.log('ari',ari_score(predlbl,l))
        
        loss = F.mse_loss(pred.squeeze(0), exp)
        #self.log('valid_loss', loss,on_epoch=True, prog_bar=True, logger=True)
        self.validation_step_outputs.append((pred, exp, mask))
        
        return loss

    def on_validation_epoch_end(self):

        # Unpack the list of tuples
        glob_expression_pred, glob_expression_gt, glob_mask = zip(*self.validation_step_outputs)
        # Concatenate outputs along the sample dimension
        glob_expression_pred, glob_expression_gt, glob_mask = torch.cat(glob_expression_pred), torch.cat(glob_expression_gt), torch.cat(glob_mask)

        # Get metrics and log
        metrics = get_metrics(glob_expression_gt, glob_expression_pred, glob_mask)
        # Put val prefix in metric dict
        metrics = {f'val_{k}': v for k, v in metrics.items()}

        # Auxiliar metric dict with a changed name to facilitate things. aux_metrics is not necesarily representing best metrics.
        aux_metrics = {f'best_{k}': v for k, v in metrics.items()}
        # Log best metrics
        if self.best_metrics is None:
            self.best_metrics = aux_metrics
        else:
            # Define metric name
            metric_name = f'best_val_{self.args.optim_metric}'
            # Determine if we got a new best model (robust to minimization or maximization of any metric)
            got_best_min = (self.metric_objective == 'min') and (aux_metrics[metric_name] < self.best_metrics[metric_name])
            got_best_max = (self.metric_objective == 'max') and (aux_metrics[metric_name] > self.best_metrics[metric_name])
            # If we got a new best model, save it and log the metrics in wandb
            if got_best_min or got_best_max:
                self.best_metrics = aux_metrics
                self.model_best_weights = self.state_dict()
        
        # Log metrics and best metrics in each validation step
        self.log_dict({**metrics, **self.best_metrics})
        
        # Free memory
        self.validation_step_outputs.clear()         

    def test_step(self, batch):
        patch, center, exp, adj, oris, sfs, *_, mask = batch
        
        adj=adj.squeeze(0)
        exp = exp.squeeze(0)
        mask = mask.squeeze(0)
        pred,extra,h = self(patch, center, adj) 

        self.test_step_outputs.append((pred, exp, mask))
        
        return pred, exp, mask

    def on_test_epoch_end(self):
        
        # Unpack the list of tuples
        glob_expression_pred, glob_expression_gt, glob_mask = zip(*self.test_step_outputs)
        # Concatenate outputs along the sample dimension
        glob_expression_pred, glob_expression_gt, glob_mask = torch.cat(glob_expression_pred), torch.cat(glob_expression_gt), torch.cat(glob_mask)

        # Get metrics and log
        metrics = get_metrics(glob_expression_gt, glob_expression_pred, glob_mask)
        # Put test prefix in metric dict
        metrics = {f'test_{k}': v for k, v in metrics.items()}
        self.log_dict(metrics, on_epoch=True)
        # Free memory
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optim=torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        StepLR = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.9)
        optim_dict = {'optimizer': optim, 'lr_scheduler': StepLR}
        return optim_dict
