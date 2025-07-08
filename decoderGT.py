import torch 
import math
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # [T, N, F]
        return x + self.pe[: x.size(0), :]

class Transformer(nn.Module):
    def __init__(self, d_model=8, nhead=4, num_layers=2, dropout=0.5,pe=True):
        super(Transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(d_model)
        self.pe = pe

    def forward(self, src,src_mask,ins=None):
        # src [N, T, F] --> [T, N, F], [60, 512, 8]
        src = src.transpose(1, 0)  # not batch first
        # print(src.min(),src.max())
        # nanper(src,'src')
        if self.pe:
            src = self.pos_encoder(src)
        # nanper(src,'src pe')
        # print(src.min(),src.max())
        output = self.transformer_encoder(src)#,  src_key_padding_mask=src_mask)  # [60, 512, 8]
        # nanper(output,'output')
        return output.transpose(1,0)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, hidden_dim):
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim should be divisible by num_heads"
        
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        # Weight matrices for query, key, and value projections
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)

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

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(num_heads, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, stock, future):
        attn_output = self.attention(stock, future, future)
        stock = self.layer_norm1(stock + self.dropout(attn_output))
        ff_output = self.feed_forward(stock)
        output = self.layer_norm2(stock + self.dropout(ff_output))
        return output

class TransformerEncoderD(nn.Module):
    def __init__(self, num_blocks, hidden_dim, num_heads, dropout):
        super(TransformerEncoderD, self).__init__()
        self.blocks = nn.ModuleList([TransformerBlock(hidden_dim, num_heads, dropout) for _ in range(num_blocks)])

    def forward(self, stock, future):
        for block in self.blocks:
            stock = block(stock, future)
        return stock
