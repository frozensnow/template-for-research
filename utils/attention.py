import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask


# The numbers of Q, K, V vectors are the same!
class FullAttention(nn.Module):
    """
    General Multi-Head Self-Attention Mechanism
    """
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None,
                 mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        """

        :param d_model: the original dimension of Q, K, V vectors
        :param n_heads: the number of heads
        :param d_keys: the dimension of keys calculating in one-head
        :param d_values: the dimension of values calculating in one-head
        :param mask_flag: whether mask the future
        :param scale: attention calculation factor
        :param attention_dropout: rate dropout
        :param output_attention: whether output the attention
        """
        super(FullAttention, self).__init__()
        # attention settings
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

        # multi-head settings
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        """

        :param queries: (batch_size x Q_num x vec_dimension)
        :param keys: (batch_size x K_num x vec_dimension)
        :param values: (batch_size x K_num x vec_dimension)
        :param attn_mask: designated mask matrix (default up-triangular mask matrix)

        :return: Features after Attention calculations , Attention matrix (optional)
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        scale = self.scale or 1. / sqrt(queries.shape[-1])
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device).mask
            scores.masked_fill_(attn_mask, -np.inf)

        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", attn, values)

        out = V.contiguous()
        out = out.view(B, L, -1)

        if self.output_attention:
            return self.out_projection(out), attn
        else:
            return self.out_projection(out), None






