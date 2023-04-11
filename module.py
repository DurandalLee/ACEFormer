# -*- coding: utf-8 -*-
from pretreatment import *
import torch.nn.functional as f

class PosFeedForwardNet(nn.Module, ABC):
    def __init__(self, embed_dim: int, forward_dim: int, dropout: float, activation: str = "relu"):
        super(PosFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=forward_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=forward_dim, out_channels=embed_dim, kernel_size=1)
        self.activation = f.relu if activation == "relu" else f.gelu
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs: torch.tensor):
        output = self.conv1(inputs.transpose(-1, -2))
        output = self.drop(self.activation(output))
        output = self.conv2(output).transpose(-1, -2)
        output = self.drop(output)
        return output

class Distilling(nn.Module, ABC):
    def __init__(self, distill_in: int):
        super(Distilling, self).__init__()
        self.conv = nn.Conv1d(in_channels=distill_in, out_channels=distill_in, kernel_size=3, padding=1, padding_mode="circular")
        self.normal = nn.BatchNorm1d(distill_in)
        self.activation = nn.ELU()
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, distill_input):
        conv_out = self.conv(distill_input.permute(0, 2, 1))
        norm_out = self.normal(conv_out)
        norm_out = self.activation(norm_out)
        pool_out = self.max_pool(norm_out.transpose(1, 2))
        return pool_out


class CrossLayer(nn.Module, ABC):
    def __init__(self, attention, embed_dim: int, forward_dim: int, dropout: float = 0.1, activation: str = "relu"):
        super(CrossLayer, self).__init__()
        self.attention = attention
        self.feedforward = PosFeedForwardNet(embed_dim, forward_dim, dropout, activation)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.tensor, x_cross: torch.tensor):
        attn_out = self.attention(x, x_cross, x_cross)
        x = x + self.drop(attn_out)
        x = self.norm1(x)

        feed_out = self.feedforward(attn_out)
        x = x + feed_out
        x = self.norm2(x)

        return x
