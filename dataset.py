from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import polars as pl

import pandas as pd
import math
import numpy as np

import torch

def idx_df(df,trading_days,window_len,step,name_date):
    instruments = df['instrument'].unique().tolist()
    num_day = len(trading_days)
    num_ins = len(instruments)


    df_ = pd.DataFrame({
        name_date: pd.Series(trading_days*num_ins).sort_values().values,
        'instrument': instruments*num_day})
    
    df__ = df_.merge(df,how='left',on=[name_date,'instrument']).sort_values([name_date,'instrument']).set_index([name_date,'instrument'])#.reset_index(drop=1)

    array_ = list(range(0,num_ins*num_day,num_ins))
    array = list(zip(array_[:-1],array_[1:]))

    idxs = list(range(0,len(array),step))
    idxs = [(array[i][0],array[i+window_len][0]) for i in list(idxs[:-math.ceil(window_len/step)])]

    return df__,idxs,num_ins,instruments,df_


def subg(ins2id,future_ins,stock_ins,graph_down,graph_up):
    graph_idx = np.array(list(map(lambda x:ins2id[x.upper()], future_ins)) + list(map(lambda x:ins2id[x[2:]], stock_ins)))
    gd = torch.tensor(graph_down[graph_idx,:][:,graph_idx]).float()
    gu = torch.tensor(graph_up[graph_idx,:][:,graph_idx]).float()
    return gd,gu,graph_idx

class TSDataset(Dataset):
    def __init__(self, future,stock,name_label='Ref($close, -2) / Ref($close, -1) - 1',name_date='datetime',window_len=10,step=5,ins2idall={}):

        self.window_len = window_len
        self.step = step

        trading_days = pd.concat([future[name_date],stock[name_date]],axis=0).unique().tolist()
        self.trading_days = stock[name_date].unique().tolist()

        self.df_future,self.idxs_future,self.num_future,self.future_ins,outdf = idx_df(future,trading_days,window_len,step,name_date)
        self.df_stock,self.idxs_stock,self.num_stock,self.stock_ins,self.outdf = idx_df(stock,trading_days,window_len,step,name_date)

        ins2id = np.load('/home/qiyiyan/linsida_www/data/map_ins2id.npy',allow_pickle=True).item()
        graph_down = np.load('/home/qiyiyan/linsida_www/data/adj_downstream.npz')['arr_0']
        graph_up = np.load('/home/qiyiyan/linsida_www/data/adj_upstream.npz')['arr_0']
        self.graph_down,self.graph_up,self.graph_idx = subg(ins2id,self.future_ins,self.stock_ins,graph_down,graph_up)

        self.label = self.df_stock[name_label].values
        self.df_stock = self.df_stock.drop(name_label,axis=1)
        self.ins2idall = ins2idall
        ins_stock, ins_future = list(map(lambda x:ins2idall.get(x,-1) ,self.stock_ins )),list(map(lambda x:ins2idall.get(x,-1) ,self.future_ins ))
        self.ins = (torch.tensor(ins_stock, dtype=torch.long), torch.tensor(ins_future, dtype=torch.long))
    def __len__(self):
        return len(self.idxs_stock)

    def get_index(self,idx):
        return self.df_stock.index.values[self.idxs_stock[idx][0]:self.idxs_stock[idx][1]].reshape(self.window_len,self.num_stock,-1)[-1,:,2]

        pass

    def __getitem__(self, idx):

        feature_future = self.df_future.values[self.idxs_future[idx][0]:self.idxs_future[idx][1]].reshape(self.window_len,self.num_future,-1)
        feature_future_mask = (np.isnan(feature_future).sum(axis=-1)!=0)
        
        feature_stock = self.df_stock.values[self.idxs_stock[idx][0]:self.idxs_stock[idx][1]].reshape(self.window_len,self.num_stock,-1)
        feature_stock_mask = (np.isnan(feature_stock).sum(axis=-1)!=0)

        label = self.label[self.idxs_stock[idx][0]:self.idxs_stock[idx][1]].reshape(self.window_len,self.num_stock,-1)
        label_mask = (np.isnan(label))
        
        return feature_future, feature_future_mask, feature_stock, feature_stock_mask,label,label_mask#,self.ins[0],self.ins[1]