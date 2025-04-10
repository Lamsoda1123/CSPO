import pandas as pd
import math
import numpy as np
import torch 
import math
import torch.nn as nn

from dataset import TSDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from pytorch_transformer_ts import TransformerModel

from utils import log_dir, copy_py_files

if not torch.cuda.is_available():
    raise


path = './data/freq_day/csi500_Alpha158'
df_stock_train = pd.read_parquet(path+'/train_day_close.parquet.gzip').reset_index()
df_stock_train['datetime'] = df_stock_train['datetime'].astype('str')

df_stock_valid = pd.read_parquet(path+'/valid_day_close.parquet.gzip').reset_index()
df_stock_valid['datetime'] = df_stock_valid['datetime'].astype('str')

df_stock_test = pd.read_parquet(path+'/test_day_close.parquet.gzip').reset_index()
df_stock_test['datetime'] = df_stock_test['datetime'].astype('str')

df_future_all = pd.read_parquet('./data/freq_day/future_day_2006_weighted_.parquet')
df_future_all['datetime'] = df_future_all['datetime'].astype('str')

print(df_stock_train['datetime'].min(),df_stock_train['datetime'].max(),df_stock_valid['datetime'].min(),df_stock_valid['datetime'].max(),df_stock_test['datetime'].min(),df_stock_test['datetime'].max())
df_future_train = df_future_all[(df_future_all['datetime']>=df_stock_train['datetime'].min()) & (df_future_all['datetime']<=df_stock_train['datetime'].max())]
df_future_valid = df_future_all[(df_future_all['datetime']>=df_stock_valid['datetime'].min()) & (df_future_all['datetime']<=df_stock_valid['datetime'].max())]
df_future_test = df_future_all[(df_future_all['datetime']>=df_stock_test['datetime'].min()) & (df_future_all['datetime']<=df_stock_test['datetime'].max())]

instrument_all = np.unique(np.hstack([df_stock_train['instrument'].unique(),df_stock_valid['instrument'].unique(),df_stock_test['instrument'].unique(),df_future_all['instrument'].unique()]))
ins2idx = dict(zip(instrument_all,range(len(instrument_all))))

import itertools


win_lens = [21]
dropouts = [0,0.3,0.5]
num_layers_list = [8]
lrs = [1e-5]
regs = [1e-3]
losses = ['DIY'] #['mse','ic','DIY',''][::-1]
d_models = [128,256,512]

parameter_combinations = list(itertools.product(win_lens, dropouts, num_layers_list,lrs,regs,losses,d_models))

for win_len, dropout, num_layers,lr,reg,loss,d_model in parameter_combinations:
    print(win_len, dropout, num_layers,lr,reg,loss,d_model,int(8 / (num_layers / 4) / (win_len / 10) / (d_model / 256)))
    save_path = log_dir()
    copy_py_files('./', save_path)

    train_ds = TSDataset(df_future_train, df_stock_train, step=1, window_len=win_len,ins2idall=ins2idx)
    valid_ds = TSDataset(df_future_valid, df_stock_valid, step=1, window_len=win_len,ins2idall=ins2idx)
    test_ds = TSDataset(df_future_test, df_stock_test, step=1, window_len=win_len,ins2idall=ins2idx)

    model = TransformerModel(d_feat = 44,
                             d_model = d_model,
                             batch_size =1,
                             nhead = 8,
                             num_layers = num_layers,
                             dropout = dropout,
                             n_epochs=400,
                             lr=lr,
                             metric="",
                             early_stop=3,
                             loss=loss,
                             optimizer="adam",
                             reg=reg,
                             GPU=0,
                             ddp=1,
                             seed=20240704,
                             save_path=save_path,
                             tokensize=len(instrument_all),
                             model_path='/home/qiyiyan/linsida_www/log/241022_1336/model_v0.107_t0.055.model' if d_model==256 else ''
                             )
    
    model.fit([train_ds, valid_ds, test_ds])        
    pred_stock,valid_ds, stock_score,ddf = model.predict(test_ds)  
    ddf.to_parquet(save_path+f'/valid_{win_len, dropout, num_layers,lr,reg,loss,d_model}.parquet')

