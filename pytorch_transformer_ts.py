# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import copy
import math
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')
from torch.cuda.amp import autocast, GradScaler
from dataset import *
from torch.utils.data import DataLoader

from model import *

import os
import logging


class TransformerModel(nn.Module):
    def __init__(
        self,
        d_feat: int = 20,
        d_model: int = 64,
        batch_size: int = 8192,
        nhead: int = 2,
        num_layers: int = 2,
        dropout: float = 0,
        n_epochs=100,
        lr=0.0001,
        metric="",
        early_stop=5,
        loss="mse",
        optimizer="adam",
        reg=1e-3,
        n_jobs=10,
        GPU=0,
        seed=19990914,
        window_len=21,
        ddp=True,
        step=5,
        model_path='',
        save_path='',
        tokensize=100,
        **kwargs
    ):
        super(TransformerModel, self).__init__()
        # set hyper-parameters.
        self.model_path = model_path
        self.save_path = save_path

        self.tokensize = tokensize

        self.d_model = d_model  
        self.window_len = window_len
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.n_jobs = n_jobs
        self.ddp = ddp
        # os.environ["CUDA_VISIBLE_DEVICES"] = self.ddp
        self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed
        
        self.step = step
        self.fitted = False

        self.scaler = GradScaler()
        self.K = torch.cuda.device_count()
        
        self.log()
        parameter_string = f'd_model:{d_model}_windowlen:{window_len}_batch_size:{batch_size}_nhead:{nhead}_num_layers:{num_layers}_dropout:{dropout}_n_epochs:{n_epochs}_lr:{lr}_metric:{metric}_early_stop:{early_stop}_loss:{loss}_optimizer:{optimizer}_reg:{reg}_n_jobs:{n_jobs}_GPU:{GPU}_seed:{seed}_step:{step}_model_path:{model_path}_save_path:{save_path}'
        self.logger.info(parameter_string)

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.model = Model2(stock_feat=158, future_feat=12, d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout,tokensize=self.tokensize)

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.reg)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.reg)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))


        if self.model_path:        
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device)) # 多卡报错解决
            print("Loading pretrained model Done...")
        
        if self.ddp:
             # number of GPUs
            self.batch_size *= self.K
            print('ddp : torch.cuda.device_count:',self.K, self.batch_size)
            
            self.model = nn.DataParallel(self.model, device_ids=list(range(self.K)))
        self.model.to(self.device)

    def log(self):
        # Initialize the logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        # Create a file handler and set the formatter
        file_handler = logging.FileHandler(os.path.join(self.save_path, 'log.txt'))
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        loss = (pred.float() - label.float()) ** 2
        return torch.mean(loss)

    @staticmethod
    def pearson_correlation(x, y):
        """
        Compute the Pearson correlation coefficient between
        two tensors while preserving gradient information.
        """
        mean_x = torch.mean(x)
        mean_y = torch.mean(y)
        xm = x - mean_x
        ym = y - mean_y
        r_num = torch.sum(xm * ym)
        r_den = torch.sqrt(torch.sum(xm ** 2) * torch.sum(ym ** 2))
        r = r_num / r_den

        # To avoid division by zero, in case of zero variance
        r = torch.where(torch.isnan(r), torch.zeros_like(r), r)
        return r
    
    def loss_fn(self, pred0,pred1,mask, label, var0=None,var1=None):

        # uni = ins.unique()
        # losses = 0
        # for u in uni:
        #     mask = (ins == u).bool()
        #     losses += -self.pearson_correlation(pred[mask.flatten()], label[mask.flatten()])
        # return losses
        # ic = self.ic_loss((pred0+pred1)/2,label,mask)
        ic0 = self.ic_loss(pred0,label,mask)
        ic = ic0
        # return - ic 
        pred0 = pred0[~mask]
        pred1 = pred1[~mask]

        var0 = var0[~mask]
        var1 = var1[~mask]
        # print('pred0.shape',pred0.shape,)
        # print('pred_var0.shape',pred_var0.shape,)

        bs,window,ins,pdim = label.size()
        # print('label[:,-1,:].reshape(bs,ins).shape',label[:,-1,:].reshape(bs,ins).shape,)
        label = label[:,-1,:].reshape(bs,ins)[~mask].to(self.device).float()
        # print('label.shape',label.shape,)

        mask = ~torch.isnan(label)
        # mask = torch.ones_like(label).bool()

        if self.loss == "mse":
            # pred = (pred0+pred1)/2
            return self.mse(pred0[mask], label[mask])

        elif self.loss == "ic":
            pred = (pred0+pred1)/2
            return -self.pearson_correlation(pred[mask], label[mask])

        elif self.loss == "DIY":
            pred = (pred0+pred1)/2
            loss_ = torch.mean((pred0[mask]-label[mask])**2/(2*torch.exp(var0[mask])) + var0[mask]/2) +\
                    torch.mean((pred1[mask]-label[mask])**2/(2*torch.exp(var1[mask])) + var1[mask]/2) +\
                    self.mse(var0[mask],var1[mask])  - ic
                    # - self.pearson_correlation(pred[mask], label[mask])
                    # 
        elif self.loss == "22":
            # pred = (pred0+pred1)/2
            loss_ = torch.mean((pred0[mask]-label[mask])**2/(2*torch.exp(var0[mask])) + var0[mask]/2) +\
                    -ic0
                    # - self.pearson_correlation(pred0[mask], label[mask])
            return loss_
        elif self.loss == "21":
            # pred = (pred0+pred1)/2
            loss_ = torch.mean((pred0[mask]-label[mask])**2/(2*torch.exp(var0[mask]))) +\
                        -ic0
                    # - self.pearson_correlation(pred0[mask], label[mask])
            return loss_
        elif self.loss == "20":
            # pred = (pred0+pred1)/2
            loss_ = torch.mean((pred0[mask]-label[mask])**2/(2*var0[mask]+1e-7) ) +\
                        -ic0
                    # - self.pearson_correlation(pred0[mask], label[mask])
                    # - self.pearson_correlation(pred[mask], label[mask])
            return loss_
        else:
            pred = (pred0+pred1)/2
            return 10*self.mse(pred[mask], label[mask])-self.pearson_correlation(pred[mask], label[mask])

    def train_epoch(self, ds):
        
        self.model.train()

        scores = []
        losses = []
        # graph_down,graph_up,graph_idx = ds.graph_down.repeat(self.K if self.ddp else 1 , 1,1),\
        #                         ds.graph_up.repeat(self.K if self.ddp else 1 , 1,1),\
        #                         ds.graph_idx
        ins_stock,ins_future = ds.ins
        ins_stock,ins_future = torch.tensor(ins_stock, dtype=torch.long).repeat(self.K if self.ddp else 1 , 1,1).to(self.device),\
                                torch.tensor(ins_future, dtype=torch.long).repeat(self.K if self.ddp else 1 , 1,1).to(self.device)


        data_loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=True,num_workers=64)#,pin_memory=True)
        for data in data_loader:
            feature_future, feature_future_mask, feature_stock, feature_stock_mask,label,label_mask = data
            
            feature_future, feature_future_mask, feature_stock, feature_stock_mask,label,label_mask = \
                    feature_future.float(), feature_future_mask, feature_stock.float(), feature_stock_mask,label.float().to(self.device),label_mask
            feature_future = feature_future.masked_fill(torch.isnan(feature_future), 0.0)
            feature_stock = feature_stock.masked_fill(torch.isnan(feature_stock), 0.0)

            self.train_optimizer.zero_grad()
            # print('feature_future.isnan().sum()',feature_future.isnan().sum())
            # print('feature_stock.isnan().sum()',feature_stock.isnan().sum())
            with autocast():

                pred0,pred1,pred_mask,pred_var0,pred_var1 = self.model(feature_future.to(self.device), feature_future_mask.to(self.device), \
                                            feature_stock.to(self.device), feature_stock_mask.to(self.device),\
                                                (),\
                                                (ins_stock,ins_future)  )
                
                mask = (pred_mask | label_mask[:,-1,:].squeeze().to(self.device))
                if self.loss == "DIY" or self.loss == "ic":
                    score = self.metric_fn((pred0+pred1)/2, label.float(),mask)
                else:
                    score = self.metric_fn(pred0, label.float(),mask)
                # print('mask.shape',mask.shape,)
                # print('pred0.shape',pred0.shape,)
                # print('pred_var0.shape',pred_var0.shape,)
                # pred0 = pred0[~mask]
                # pred1 = pred1[~mask]

                # pred_var0 = pred_var0[~mask]
                # pred_var1 = pred_var1[~mask]
                # # print('pred0.shape',pred0.shape,)
                # # print('pred_var0.shape',pred_var0.shape,)
                
                # bs,window,ins,pdim = label.size()
                # # print('label[:,-1,:].reshape(bs,ins).shape',label[:,-1,:].reshape(bs,ins).shape,)
                # label = label[:,-1,:].reshape(bs,ins)[~mask].to(self.device)
                # # print('label.shape',label.shape,)
                loss = self.loss_fn(pred0, pred1, mask,label,pred_var0,pred_var1)
                
                # raise
            # print('(pred0+pred1)/2.isnan().sum()',((pred0+pred1)/2).isnan().sum())
            # print('label.isnan().sum()',label.isnan().sum())
            # if loss.isnan(): 
            #     raise
            # print('loss:',loss.item())
            losses.append(loss.item())
            scores.append(score)
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 1.0)
            self.scaler.step(self.train_optimizer)
            self.scaler.update()

        return np.mean(losses), self.score_cal(scores)

    def test_epoch(self, ds,dl=True):
        self.model.eval()

        scores = []
        losses = []
        outcome = []

        # graph_down,graph_up,graph_idx = ds.graph_down.to(self.device).repeat(self.K if self.ddp else 1,1,1),\
        #                         ds.graph_up.to(self.device).repeat(self.K if self.ddp else 1,1,1),\
        #                         ds.graph_idx
        ins_stock,ins_future = ds.ins
        ins_stock,ins_future = torch.tensor(ins_stock, dtype=torch.long).repeat(self.K if self.ddp else 1 , 1,1).to(self.device),\
                                torch.tensor(ins_future, dtype=torch.long).repeat(self.K if self.ddp else 1 , 1,1).to(self.device)
        # att_fs = []
        data_loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False, drop_last=dl,num_workers=64)#10,pin_memory=True)
        vars = []
        for data in data_loader:
            feature_future, feature_future_mask, feature_stock, feature_stock_mask,label,label_mask  = data

            feature_future, feature_future_mask, feature_stock, feature_stock_mask,label,label_mask = \
                    feature_future.float(), feature_future_mask, feature_stock.float(), feature_stock_mask,label.float().to(self.device),label_mask
            ins_stock,ins_future = ins_stock.to(self.device), ins_future.to(self.device)
            feature_future = feature_future.masked_fill(torch.isnan(feature_future), 0.0)
            feature_stock = feature_stock.masked_fill(torch.isnan(feature_stock), 0.0)
            # print('feature_future.isnan().sum()',feature_future.isnan().sum())
            # print('feature_stock.isnan().sum()',feature_stock.isnan().sum())
            with torch.no_grad():
                with autocast():

                    pred0,pred1,pred_mask,pred_var0,pred_var1 = self.model(feature_future.to(self.device), feature_future_mask.to(self.device), \
                                                feature_stock.to(self.device), feature_stock_mask.to(self.device),\
                                                    (),  #(graph_down,graph_up,graph_idx),\
                                                (ins_stock,ins_future)  )
                    # print('pred_var0.shape',pred_var0.shape)
                    pred_ori = ((pred0+pred1)/2).clone().flatten()
                    mask = (pred_mask | label_mask[:,-1,:].squeeze().to(self.device))
                    score = self.metric_fn((pred0+pred1)/2, label.float(),mask)
                    # pred0 = pred0[~mask]
                    # pred1 = pred1[~mask]
                    # pred_var0 = pred_var0[~mask]
                    # pred_var1 = pred_var1[~mask]
                    vars.append(((pred_var0[~mask]+pred_var1[~mask])/2).detach().cpu().numpy())
                    # pred = (pred0+pred1)/2
                    # bs,window,ins,pdim = label.size()
                    # label = label[:,-1,:].reshape(bs,ins)[~mask].to(self.device)


                    loss = self.loss_fn(pred0,pred1, mask, label.float(),pred_var0,pred_var1)
                    # score = self.metric_fn(pred, label.float())

                # if loss.isnan(): 
                #     raise
                # if score.isnan(): raise
                # att_fs.append(att_f.detach().cpu().numpy())
                losses.append(loss.item())
                # scores.append(score.item())
                scores.append(score)
                outcome.append(pred_ori.detach().cpu().numpy())
        # np.save(self.save_path+'/att_f.npy',np.vstack(att_fs))
        # np.save(self.save_path+'/att_s.npy',att_s.detach().cpu().numpy())
        # scores = np.mean(scores)
        scores = self.score_cal(scores)
        outcome = np.hstack(outcome)
        vars = np.vstack(vars).flatten()
        np.save(self.save_path+'/vars.npy',vars)
        print(scores)
        # print('len of label',outcome.shape[0])
        
        return np.mean(losses), scores, outcome

    def fit(self, dataset):

        train_ds, valid_ds,test_ds = dataset
        best_score = np.inf
        best_ic = -np.inf
        stop_steps = 0
        train_loss = 0
        best_epoch = 0
        self.fitted = True
        losses = []
        ### stage2 pretrain with stocks
        print('training......')
        for step in range(self.n_epochs):
            train_loss, train_score = self.train_epoch(train_ds)
            val_loss, val_score, _ = self.test_epoch(valid_ds)
            test_loss, test_score, _ = self.test_epoch(test_ds)
            losses.append(   (train_loss,val_loss,test_loss))
            print("Epoch%d:" % step," IC  --  train %.6f, valid %.6f, test %.6f" % (train_score['IC'], val_score['IC'], test_score['IC'] )," ||| MSE --  train %.6f, valid %.6f, test %.6f" % (train_loss, val_loss, test_loss))
            self.logger.info(f"Epoch{step}: IC  --  train {train_score['IC']:.6f}, valid {val_score['IC']:.6f}, test {test_score['IC']:.6f} ||| MSE --  train {train_loss:.6f}, valid {val_loss:.6f}, test {test_loss:.6f}")
            if val_score['IC'] > best_ic:
                best_ic = val_score['IC']
                best_ict = test_score['IC']
                best_score = test_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    # if self.loss =='mse':
                    print("early stop")
                    break
                    # else:
                    #     self.loss ='mse'
                    #     print("stage 1 early stop")
                    #     stop_steps = 0
        np.save(self.save_path+'/losses.npy',np.array(losses))
        print(f"best score: %.6lf || best IC %.6lf || test IC %.6lf || @ %d " % (best_score['IC'], best_ic,best_ict, best_epoch))
        self.logger.info(f"best score: {best_score} || best IC {best_ic} || test IC {best_ict} || @ {best_epoch} ")
        self.model.load_state_dict(best_param)
        if self.ddp:
            torch.save(self.model.module.state_dict(), self.save_path+f'/model_v{round(best_ic,3)}_t{round(best_ict,3)}.model')
        else:
            torch.save(self.model.state_dict(), self.save_path+'/model.model')
        if self.use_gpu:
            torch.cuda.empty_cache()
        print(best_score)
        return best_score

    def predict(self, valid_ds):
        stock_loss, stock_score, pred_stock = self.test_epoch(valid_ds,dl=False)
        print("IC  --  Test loss %.6f, True IC %.6f" % (stock_loss, stock_score['IC']))
        print(stock_score)
        ddf = self.groupic(pred_stock,valid_ds)
        return pred_stock,valid_ds, stock_score,ddf
    
    def groupic(self,pred_future,test_ds):
        # try:

        f = np.hstack([np.array(list(range(test_ds.idxs_stock[-1][-1])))\
                    [test_ds.idxs_stock[idx][0]:test_ds.idxs_stock[idx][1]]\
                        .reshape(test_ds.window_len,test_ds.num_stock,-1)[-1].flatten()\
                                                        for idx in range(len(test_ds))])


        df = test_ds.outdf.iloc[f,:]
        print(df.shape,pred_future.shape,test_ds.label[f].shape)
        df['pred'] = pred_future
        df['label'] = test_ds.label[f]
        print(df['pred'].corr(df['label']))
        
        return df
    
    def ic_loss(self,pred,label_,mask):
        label=label_[:,-1,:,:].squeeze()
        if label.shape.__len__()<2:
            label=label.unsqueeze(0)
        bs,_= pred.shape
        ic = 0
        for i in range(bs):
            ic += self.pearson_correlation(pred[i][~mask[i]], label[i][~mask[i]])
            # rank_ic = self.spearman_correlation(pred[i][~mask[i]], label[i][~mask[i]]).item()
        return ic/bs
    
    def metric_fn(self, pred, label_,mask):
        # print('pred.shape, label_.shape,mask.shape',pred.shape, label_.shape,mask.shape)
        # label=label_
        
        label=label_[:,-1,:,:].squeeze()
        if label.shape.__len__()<2:
            label=label.unsqueeze(0)
        bs,_= pred.shape
        ics=[]
        rics=[]
        for i in range(bs):
            ic = self.pearson_correlation(pred[i][~mask[i]], label[i][~mask[i]]).item()
            rank_ic = self.spearman_correlation(pred[i][~mask[i]], label[i][~mask[i]]).item()
            # if not np.isnan(ic):
            ics.append(ic)
            rics.append(rank_ic)
        # return {
        #     "IC": np.mean(ics),
        #     "Rank IC": np.mean(rics),
        # }
        return {
            "IC": ics,
            "Rank IC": rics,
        }
        
        # mask = ~torch.isnan(label)
        # print(mask.shape,pred.shape,label.shape)
        # pred_masked = pred[mask]
        # label_masked = label[mask]
        # ic = self.pearson_correlation(pred_masked, label_masked)
        # rank_ic = self.spearman_correlation(pred_masked, label_masked)
        # return {
        #     "IC": ic.item(),
        #     "Rank IC": rank_ic.item(),
        # }

    @staticmethod
    def pearson_correlation(x, y):
        """
        Compute the Pearson correlation coefficient between
        two tensors while preserving gradient information.
        """
        mean_x = torch.mean(x)
        mean_y = torch.mean(y)
        xm = x - mean_x
        ym = y - mean_y
        r_num = torch.sum(xm * ym)
        r_den = torch.sqrt(torch.sum(xm ** 2) * torch.sum(ym ** 2))
        r = r_num / r_den

        # To avoid division by zero, in case of zero variance
        r = torch.where(torch.isnan(r), torch.zeros_like(r), r)
        return r

    def spearman_correlation(self, pred, label):
        pred_ranked = torch.argsort(torch.argsort(pred)).float()
        label_ranked = torch.argsort(torch.argsort(label)).float()
        return self.pearson_correlation(pred_ranked, label_ranked)
    
    def score_cal(self,list_score):
        from functools import reduce
        # print(list_score[0])
        # print(list_score[1])
        ics = list(reduce(lambda x,y:x+y,[z['IC'] for z in list_score]))
        rics = list(reduce(lambda x,y:x+y,[z['Rank IC'] for z in list_score]))

        # ics = [x['IC'] for x in list_score]
        # rics = [x['Rank IC'] for x in list_score]


        ans = {}
        ans['IC'] = np.mean(ics)
        ic_std = np.std(ics)

        
        ans['Rank IC'] = np.mean(rics)
        rank_ic_std = np.std(rics)
    
        ans['IR'] = ans['IC']/ic_std
        ans['rankIR'] = ans['Rank IC']/rank_ic_std
        return ans
    