import torch 
import math
import torch.nn as nn
import torch.nn.functional as F


from decoderGT import TransformerEncoderD,Transformer,TransformerA
from loraModel import loraTransformerEncoder

class Model(nn.Module):
    def __init__(self, stock_feat=6, future_feat=6, d_model=8, nhead=4, num_layers=2, dropout=0.5,tokensize=0):
        super(Model, self).__init__()
        self.stock_mapping = nn.Linear(stock_feat, d_model)
        self.future_mapping = nn.Linear(future_feat, d_model)

        self.stock_encoder = Transformer(d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout)
        self.future_encoder = Transformer(d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout)

        self.stock_std_encoder = Transformer(d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout)
        self.stock_std_predictor = nn.Linear(d_model, 1)

        self.instrument_pe = nn.Embedding(tokensize,d_model)
        self.relation_encoder = TransformerEncoderD(num_blocks=num_layers, hidden_dim=d_model, num_heads=nhead, dropout=dropout)
        self.stock_relation_encoder = Transformer(d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout,pe=False)
        # self.stock_relation_encoder.register_attention_hook()
        # print('register ok')
        self.predictor = nn.Linear(d_model, 1)
        
        
    def forward(self, feature_future, feature_future_mask, feature_stock, feature_stock_mask,gs=None,inss=(None,None)):
        
        # feature [bs, window, ins, F] 
        # basic mapping
        feature_stock,feature_future = self.initprocess(feature_stock,feature_future)

        # feature [bs, window, ins, F]
        # Future series feature extrator
        feature_future,feature_future_mask,bs, window, ins, dim = self.preprocess(feature_future,feature_future_mask)
        feature_future = self.future_encoder(feature_future, feature_future_mask).reshape(bs,ins,window,dim)
        feature_future,feature_future_mask = self.afterprocess(feature_future,feature_future_mask,bs,ins,dim,window)

        # Stock series predictor
        feature_stock,feature_stock_mask,bs, window, ins, dim = self.preprocess(feature_stock,feature_stock_mask)
        feature_stock_ = self.stock_encoder(feature_stock, feature_stock_mask).reshape(bs,ins,window,dim)

        feature_std_stock = self.stock_std_encoder(feature_stock, feature_stock_mask).reshape(bs,ins,window,dim)
        feature_std_stock = self.stock_std_predictor(feature_std_stock)
        
        feature_stock,feature_stock_mask = self.afterprocess(feature_stock_,feature_stock_mask,bs,ins,dim,window)

        # print('feature_stock, feature_future1',feature_stock.shape, feature_future.shape)

        # final step fully connect GT
        feature_stock, feature_future = feature_stock + self.instrument_pe(inss[0]), feature_future + self.instrument_pe(inss[1])
        feature_stock, feature_future = feature_stock.squeeze(), feature_future.squeeze()
        if feature_stock.shape.__len__()<3:
            feature_stock, feature_future = feature_stock.reshape(1,feature_stock.shape[0],feature_stock.shape[1]), \
            feature_future.reshape(1,feature_future.shape[0],feature_future.shape[1])
        # print('feature_stock, feature_future2',feature_stock.shape, feature_future.shape)
        feature_stock_fr = self.relation_encoder(feature_stock, feature_future)
        feature_stock = nn.functional.leaky_relu(feature_stock_fr + feature_stock)
        feature_stock = self.stock_relation_encoder(feature_stock, feature_stock_mask)
    
        feature_stock = self.predictor(feature_stock).reshape(bs,ins) 

        return  feature_stock,feature_stock_mask,feature_std_stock
    
    def initprocess(self,feature_stock,feature_future):

        feature_future = nn.functional.layer_norm(feature_future.transpose(1, 3), (feature_future.shape[1],)).transpose(1, 3)
        feature_future = nn.functional.leaky_relu(self.future_mapping(feature_future))
        feature_stock = nn.functional.layer_norm(feature_stock.transpose(1, 3), (feature_stock.shape[1],)).transpose(1, 3)
        feature_stock = nn.functional.leaky_relu(self.stock_mapping(feature_stock))
        return feature_stock,feature_future

    def preprocess(self,feature_future,feature_future_mask):

        bs, window, ins, dim  = feature_future.size()
        feature_future = feature_future.transpose(1, 2).reshape(bs*ins,window,dim)
        feature_future_mask = feature_future_mask.transpose(1, 2).reshape(bs*ins,window)
        feature_future[feature_future_mask]=0
        return feature_future,feature_future_mask,bs, window, ins, dim
    
    def afterprocess(self,feature_future,feature_future_mask,bs,ins,dim,window):

        feature_future_mask = feature_future_mask.reshape(bs,ins,window)
        feature_future[feature_future_mask]=0
        feature_future = feature_future[:,:,-1,:].reshape(bs,ins,dim)
        feature_future_mask = feature_future_mask[:,:,-1].reshape(bs,ins)
        return feature_future,feature_future_mask

class loraModel(nn.Module):
    def __init__(self, stock_feat=6, future_feat=6, d_model=8, nhead=4, num_layers=2, dropout=0.5,tokensize=0):
        super(loraModel, self).__init__()
        self.stock_mapping = nn.Linear(stock_feat, d_model)
        self.future_mapping = nn.Linear(future_feat, d_model)

        self.stock_encoder = Transformer(d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout)
        # self.future_encoder = Transformer(d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout)
        self.future_encoder = loraTransformerEncoder(d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout)


        self.stock_std_encoder = Transformer(d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout)
        self.stock_std_predictor = nn.Linear(d_model, 1)

        self.instrument_pe = nn.Embedding(tokensize,d_model)
        self.instrument_pe2 = nn.Sequential(nn.Embedding(tokensize,16),nn.Linear(16,d_model))
        self.relation_encoder = TransformerEncoderD(num_blocks=num_layers, hidden_dim=d_model, num_heads=nhead, dropout=dropout)
        self.stock_relation_encoder = Transformer(d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout,pe=False)

        self.predictor = nn.Linear(d_model, 1)
        
        
    def forward(self, feature_future, feature_future_mask, feature_stock, feature_stock_mask,gs=None,inss=(None,None)):
        
        # feature [bs, window, ins, F] 
        # basic mapping
        feature_stock,feature_future = self.initprocess(feature_stock,feature_future)

        # feature [bs, window, ins, F]
        # Future series feature extrator
        feature_future,feature_future_mask,bs, window, ins, dim = self.preprocess(feature_future,feature_future_mask)

        feature_future = self.future_encoder(feature_future, feature_future_mask).reshape(bs,ins,window,dim) + \
                            self.stock_encoder(feature_future, feature_future_mask).reshape(bs,ins,window,dim).detach()

        feature_future,feature_future_mask = self.afterprocess(feature_future,feature_future_mask,bs,ins,dim,window)

        # Stock series predictor
        feature_stock,feature_stock_mask,bs, window, ins, dim = self.preprocess(feature_stock,feature_stock_mask)
        feature_stock_ = self.stock_encoder(feature_stock, feature_stock_mask).reshape(bs,ins,window,dim)

        feature_std_stock = self.stock_std_encoder(feature_stock, feature_stock_mask).reshape(bs,ins,window,dim)
        feature_std_stock = self.stock_std_predictor(feature_std_stock)
        
        feature_stock,feature_stock_mask = self.afterprocess(feature_stock_,feature_stock_mask,bs,ins,dim,window)

        # final step fully connect GT
        feature_stock, feature_future = feature_stock + self.instrument_pe(inss[0]), feature_future + self.instrument_pe(inss[1])
        feature_stock, feature_future = feature_stock.squeeze(), feature_future.squeeze()
        
        feature_stock_fr = self.relation_encoder(feature_stock, feature_future)
        feature_stock = nn.functional.leaky_relu(feature_stock_fr + feature_stock)
        feature_stock = self.stock_relation_encoder(feature_stock, feature_stock_mask)
    
        feature_stock = self.predictor(feature_stock).reshape(bs,ins) 

        return  feature_stock,feature_stock_mask,feature_std_stock
    
    def initprocess(self,feature_stock,feature_future):

        feature_future = nn.functional.layer_norm(feature_future.transpose(1, 3), (feature_future.shape[1],)).transpose(1, 3)
        feature_future = nn.functional.leaky_relu(self.future_mapping(feature_future))
        feature_stock = nn.functional.layer_norm(feature_stock.transpose(1, 3), (feature_stock.shape[1],)).transpose(1, 3)
        feature_stock = nn.functional.leaky_relu(self.stock_mapping(feature_stock))
        return feature_stock,feature_future

    def preprocess(self,feature_future,feature_future_mask):

        bs, window, ins, dim  = feature_future.size()
        feature_future = feature_future.transpose(1, 2).reshape(bs*ins,window,dim)
        feature_future_mask = feature_future_mask.transpose(1, 2).reshape(bs*ins,window)
        feature_future[feature_future_mask]=0
        return feature_future,feature_future_mask,bs, window, ins, dim
    
    def afterprocess(self,feature_future,feature_future_mask,bs,ins,dim,window):

        feature_future_mask = feature_future_mask.reshape(bs,ins,window)
        feature_future[feature_future_mask]=0
        feature_future = feature_future[:,:,-1,:].reshape(bs,ins,dim)
        feature_future_mask = feature_future_mask[:,:,-1].reshape(bs,ins)
        return feature_future,feature_future_mask




class Model2(nn.Module):
    def __init__(self, stock_feat=6, future_feat=6, d_model=8, nhead=4, num_layers=2, dropout=0.5,tokensize=0):
        super(Model2, self).__init__()

        self.var_gt = Model(stock_feat, future_feat, d_model, nhead, num_layers, dropout,tokensize)
        self.mu_gt = Model(stock_feat, future_feat, d_model, nhead, num_layers, dropout,tokensize)
        self.var_gt1 = Model(stock_feat, future_feat, d_model, nhead, num_layers, dropout,tokensize)
        self.mu_gt1 = Model(stock_feat, future_feat, d_model, nhead, num_layers, dropout,tokensize)

    def forward(self, feature_future, feature_future_mask, feature_stock, feature_stock_mask,gs=None,inss=(None,None)):

        # print('feature_future, feature_future_mask, feature_stock, feature_stock_mask',feature_future.shape, feature_future_mask.shape, feature_stock.shape, feature_stock_mask.shape)
        # feature_std_stock, _ = self.var_gt(feature_future, feature_future_mask, feature_stock, feature_stock_mask,gs,inss)
        # print('feature_future, feature_future_mask, feature_stock, feature_stock_mask',feature_future.shape, feature_future_mask.shape, feature_stock.shape, feature_stock_mask.shape)
        feature_stock0, feature_stock_mask0, feature_std_stock = self.mu_gt(feature_future, feature_future_mask, feature_stock, feature_stock_mask,gs,inss)

        # print('feature_future, feature_future_mask, feature_stock, feature_stock_mask',feature_future.shape, feature_future_mask.shape, feature_stock.shape, feature_stock_mask.shape)
        # feature_std_stock1, _ = self.var_gt1(feature_future, feature_future_mask, feature_stock, feature_stock_mask,gs,inss)
        # print('feature_future, feature_future_mask, feature_stock, feature_stock_mask',feature_future.shape, feature_future_mask.shape, feature_stock.shape, feature_stock_mask.shape)
        feature_stock1, _, feature_std_stock1 = self.mu_gt1(feature_future, feature_future_mask, feature_stock, feature_stock_mask,gs,inss)

        # future_att_score = (future_att_score1+future_att_score2)/2
        # stock_att_score = (stock_att_score1+stock_att_score2)/2

        return feature_stock0,feature_stock1 ,feature_stock_mask0,feature_std_stock,feature_std_stock1#,future_att_score,stock_att_score1