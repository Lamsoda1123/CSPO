import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

class loraLinear(nn.Module):
    def __init__(self, in_dim,out_dim,rank=16):
        super(loraLinear, self).__init__()
        self.adp = nn.Linear(in_dim,rank)
        self.dec = nn.Linear(rank,out_dim)

        for layer in [self.adp, self.dec]:
            xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.)

    def forward(self,x):
        x = self.adp(x)
        x = self.dec(x)
        return x
    

class loraMultiHeadAttention(nn.Module):
    def __init__(self, num_heads, hidden_dim):
        super(loraMultiHeadAttention, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim should be divisible by num_heads"
        
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        # Weight matrices for query, key, and value projections
        self.W_q = loraLinear(hidden_dim, hidden_dim)
        self.W_k = loraLinear(hidden_dim, hidden_dim)
        self.W_v = loraLinear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)

        xavier_uniform_(self.W_o.weight)
        if self.W_o.bias is not None:
            nn.init.constant_(self.W_o.bias, 0.)

    def forward(self, query, key, value):
        batch_size = query.size(0)
        
        # Linear projections for query, key, value
        q = self.W_q(query)
        k = self.W_k(key)
        v = self.W_v(value)

        # Split heads and reshape for multi-head attention (batch_size, num_heads, seq_len, head_dim)
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = self.head_dim ** 0.5
        attn_output = self.scaled_dot_product_attention(q, k, v, scale)

        # Concatenate the heads (batch_size, seq_len, hidden_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)

        # Final linear projection
        output = self.W_o(attn_output)
        return output

    def scaled_dot_product_attention(self, query, key, value, scale):
        # Compute attention scores (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1)) / scale

        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Compute output (batch_size, num_heads, seq_len, head_dim)
        output = torch.matmul(attention_weights, value)
        return output

class loraTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super(loraTransformerBlock, self).__init__()
        self.attention = loraMultiHeadAttention(num_heads, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, stock):
        attn_output = self.attention(stock, stock, stock)
        stock = self.layer_norm1(stock + self.dropout(attn_output))
        ff_output = self.feed_forward(stock)
        output = self.layer_norm2(stock + self.dropout(ff_output))
        return output

class loraTransformerEncoder(nn.Module):
    def __init__(self,d_model=256, nhead=8, num_layers=8, dropout=0.1):
        super(loraTransformerEncoder, self).__init__()
        self.blocks = nn.ModuleList([loraTransformerBlock(d_model, nhead, dropout) for _ in range(num_layers)])

    def forward(self, stock,mask):
        for block in self.blocks:
            stock = block(stock)
        return stock


