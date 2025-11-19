import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RelativePositionBias(nn.Module):
    """Learnable relative position bias for attention"""
    def __init__(self, num_heads, max_distance=128):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        # Embeddings for relative positions
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(2 * max_distance + 1, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
    def forward(self, seq_len):
        # Create relative position matrix
        positions = torch.arange(seq_len, device=self.relative_position_bias_table.device)
        relative_positions = positions[None, :] - positions[:, None]  # [seq_len, seq_len]
        
        # Clip to max_distance
        relative_positions = torch.clamp(
            relative_positions, -self.max_distance, self.max_distance
        )
        # Shift to make all indices positive
        relative_positions = relative_positions + self.max_distance
        
        # Get biases: [seq_len, seq_len, num_heads]
        biases = self.relative_position_bias_table[relative_positions]
        # Reshape to [num_heads, seq_len, seq_len]
        return biases.permute(2, 0, 1)


class MultiHeadAttentionWithRPE(nn.Module):
    """Multi-head attention with relative position encoding"""
    def __init__(self, d_model, num_heads, dropout=0.1, max_distance=128):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Relative position bias
        self.relative_position_bias = RelativePositionBias(num_heads, max_distance)
        
    def forward(self, x):
        B, N, C = x.shape  # batch, seq_len, d_model
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        
        # Add relative position bias
        rel_pos_bias = self.relative_position_bias(N)  # [num_heads, N, N]
        attn = attn + rel_pos_bias.unsqueeze(0)  # Broadcast over batch
        
        # Apply softmax
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class TransformerBlockWithRPE(nn.Module):
    """Transformer block with RPE"""
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1, max_distance=128):
        super().__init__()
        
        self.attention = MultiHeadAttentionWithRPE(
            d_model, num_heads, dropout, max_distance
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention with residual
        x = x + self.attention(self.norm1(x))
        # Feedforward with residual
        x = x + self.ffn(self.norm2(x))
        return x
