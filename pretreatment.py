# -*- coding: utf-8 -*-
from abc import ABC
import math
import torch
import torch.nn as nn


class ExpandEmbedding(nn.Module, ABC):
    def __init__(self, in_dim, out_dim):
        super(ExpandEmbedding, self).__init__()
        self.ExpandConv = nn.Conv1d(in_channels=in_dim, out_channels=out_dim,
                                    kernel_size=3, padding=1, padding_mode='circular')

    def forward(self, x: torch.tensor):
        x = self.ExpandConv(x.transpose(1, 2)).transpose(1, 2)
        return x


class PositionalEmbedding(nn.Module, ABC):
    def __init__(self, out_dim, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, out_dim).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, out_dim, 2).float() * (-math.log(10000.0) / out_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return self.pe.to(x.device)[:, :x.shape[1]]


class StampEmbedding(nn.Module, ABC):
    def __init__(self, out_dim):
        super(StampEmbedding, self).__init__()
        embed = nn.Embedding

        self.day_embedding = embed(32, out_dim)
        self.week_embedding = embed(6, out_dim)
        self.month_embedding = embed(13, out_dim)

    def forward(self, month, weekday, day):
        month_x = self.month_embedding(month)
        week_x = self.week_embedding(weekday)
        day_x = self.day_embedding(day)

        return day_x + week_x + month_x


class DataEmbedding(nn.Module, ABC):
    def __init__(self, in_dim: int, out_dim: int, drop: float = 0.1):
        super(DataEmbedding, self).__init__()

        self.expand_emb = ExpandEmbedding(in_dim, out_dim)
        self.position_emb = PositionalEmbedding(out_dim)
        self.stamp_emb = StampEmbedding(out_dim)
        self.dropout = nn.Dropout(p=drop)

    def forward(self, x: torch.tensor, time: torch.tensor):
        x = self.expand_emb(x) + self.position_emb(x)
        stamp = self.stamp_emb(time[:, :, 0], time[:, :, 1], time[:, :, 2])
        emb = self.dropout(x + stamp)

        return emb
