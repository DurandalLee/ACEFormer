# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import numpy as np
from abc import ABC


class ProbabilityAttention(nn.Module, ABC):
    def __init__(self, embed_dim: int, n_heads: int, kq_dim: int = None, v_dim: int = None, factor: int = 5, scale: float = None):
        super(ProbabilityAttention, self).__init__()

        if embed_dim < n_heads:
            raise Exception("embedding dimension must greater than heads number.")

        self.kq_dim = kq_dim or (embed_dim // n_heads)
        self.v_dim = v_dim or (embed_dim // n_heads)
        self.n_heads = n_heads

        self.matrix_Q = nn.Linear(embed_dim, self.kq_dim * n_heads)
        self.matrix_K = nn.Linear(embed_dim, self.kq_dim * n_heads)
        self.matrix_V = nn.Linear(embed_dim, self.v_dim * n_heads)

        self.factor = factor
        self.scale = scale or 1.0 / math.sqrt(self.kq_dim)

        self.fully_con = nn.Linear(self.v_dim * n_heads, embed_dim)

    def __probability_matrix__(self, q_vector: torch.tensor, k_vector: torch.tensor):
        batch, heads, q_unit, _, = q_vector.shape
        _, _, k_unit, _, = k_vector.shape
        rand_num = self.factor * np.ceil(np.log(k_unit)).astype('int').item()
        top_num = self.factor * int(np.log(q_unit))
        rand_num = rand_num if rand_num < k_unit else k_unit
        top_num = top_num if top_num < q_unit else q_unit

        keys_expand = k_vector.unsqueeze(-3).expand(-1, -1, q_unit, -1, -1)
        rand_index = torch.randint(k_unit, (q_unit, rand_num))
        keys_random = keys_expand[:, :, torch.arange(q_unit).unsqueeze(1), rand_index, :]
        qk_prob = torch.matmul(q_vector.unsqueeze(-2), keys_random.transpose(-2, -1)).squeeze()

        discrete = qk_prob.max(-1)[0] - qk_prob.sum(-1) / q_unit
        top_index = discrete.topk(top_num, sorted=False)[1]

        q_import = q_vector[
                   torch.arange(batch)[:, None, None],
                   torch.arange(heads)[None, :, None],
                   top_index, :]

        qk_import = torch.matmul(q_import, k_vector.transpose(-2, -1)) * self.scale

        return qk_import, top_index

    def forward(self, queries: torch.tensor, keys: torch.tensor, values: torch.tensor):
        batch, q_unit, _ = queries.shape
        _, k_unit, _ = keys.shape
        _, v_unit, _ = values.shape
        heads = self.n_heads

        q_vector = self.matrix_Q(queries).view(batch, q_unit, heads, -1).transpose(1, 2)
        k_vector = self.matrix_K(keys).view(batch, k_unit, heads, -1).transpose(1, 2)
        v_vector = self.matrix_V(values).view(batch, v_unit, heads, -1).transpose(1, 2)

        # scores_top:[batch, heads, unit, top_num]
        scores_top, top_index = self.__probability_matrix__(q_vector, k_vector)
        v_coupling = v_vector.cumsum(dim=-2)

        scores_top = torch.softmax(scores_top, dim=-1)
        v_coupling[
            torch.arange(batch)[:, None, None],
            torch.arange(heads)[None, :, None],
            top_index, :
        ] = torch.matmul(scores_top, v_vector)

        # (batch, unit, embedding)
        prob_output = self.fully_con(v_coupling.transpose(1, 2).reshape(batch, v_unit, -1))

        return prob_output.contiguous()


class FullAttention(nn.Module, ABC):
    def __init__(self, embed_dim: int, n_heads: int, kq_dim: int = None, v_dim: int = None, factor: int = 5, scale: float = None):
        super(FullAttention, self).__init__()
        if embed_dim < n_heads:
            raise Exception("embedding dimension must greater than heads number.")

        self.kq_dim = kq_dim or (embed_dim // n_heads)
        self.v_dim = v_dim or (embed_dim // n_heads)
        self.n_heads = n_heads
        self.factor = factor

        self.matrix_Q = nn.Linear(embed_dim, self.kq_dim * n_heads)
        self.matrix_K = nn.Linear(embed_dim, self.kq_dim * n_heads)
        self.matrix_V = nn.Linear(embed_dim, self.v_dim * n_heads)

        self.scale = scale or 1.0 / math.sqrt(self.kq_dim)

        self.fully_con = nn.Linear(self.v_dim * n_heads, embed_dim)

    def forward(self, queries: torch.tensor, keys: torch.tensor, values: torch.tensor):
        batch, unit_q, _ = queries.shape
        _, unit_v, _ = values.shape
        heads = self.n_heads

        q_vector = self.matrix_Q(queries).view(batch, unit_q, heads, -1).transpose(1, 2)
        k_vector = self.matrix_K(keys).view(batch, unit_v, heads, -1).transpose(1, 2)
        v_vector = self.matrix_V(values).view(batch, unit_v, heads, -1).transpose(1, 2)

        scores = torch.matmul(q_vector, k_vector.transpose(-1, -2)) * self.scale

        scores = nn.Softmax(dim=-1)(scores)
        z_vector = torch.matmul(scores, v_vector)
        full_output = self.fully_con(z_vector.transpose(1, 2).reshape(batch, unit_q, -1))

        return full_output.contiguous()