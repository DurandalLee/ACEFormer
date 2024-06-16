# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from math import sqrt


class ProbabilityAttention(nn.Module):
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
        self.scale = scale or 1.0 / sqrt(self.kq_dim)

        self.fully_con = nn.Linear(self.v_dim * n_heads, embed_dim)

    def __probability_matrix__(self, q_vector: torch.tensor, k_vector: torch.tensor):
        batch, heads, q_unit, _, = q_vector.shape
        _, _, k_unit, _, = k_vector.shape
        rand_num = self.factor * np.ceil(np.log(k_unit)).astype('int').item()
        top_num = self.factor * np.ceil(np.log(q_unit)).astype('int').item()
        # 随机选择天数
        rand_num = rand_num if rand_num < k_unit else k_unit
        # 选择相关度的前几
        top_num = top_num if top_num < q_unit else q_unit

        # k矩阵扩展，扩展方式为q_unit个k矩阵
        keys_expand = k_vector.unsqueeze(-3).expand(-1, -1, q_unit, -1, -1)
        # k矩阵的随机天数数据
        rand_index = torch.randint(k_unit, (q_unit, rand_num))
        # 根据随机天数数组获取q_unit个k随机矩阵
        keys_random = keys_expand[:, :, torch.arange(q_unit).unsqueeze(1), rand_index, :]
        # q矩阵中第i天与第i个k随机矩阵矩阵相乘
        qk_prob = torch.matmul(q_vector.unsqueeze(-2), keys_random.transpose(-2, -1)).squeeze()        

        # 矩阵离散度（最大值与平均值差值），判断需要选择Q矩阵的下标
        discrete = qk_prob.max(-1)[0] - qk_prob.sum(-1) / q_unit
        top_index = discrete.topk(top_num, sorted=False)[1]

        # 截取Q矩阵中影响大的部分
        q_import = q_vector[ torch.arange(batch)[:, None, None], torch.arange(heads)[None, :, None], top_index, :]

        # 重点Q矩阵与K矩阵相乘
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
        # 增加v矩阵耦合度
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


class FullAttention(nn.Module):
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

        self.scale = scale or 1.0 / sqrt(self.kq_dim)

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
    

## Nonstationary_Transformer
class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class DSAttention(nn.Module):
    '''De-stationary Attention'''

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(1).unsqueeze(1)  # B x 1 x 1 x 1
        delta = 0.0 if delta is None else delta.unsqueeze(1).unsqueeze(1)  # B x 1 x 1 x S

        # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)
        
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn