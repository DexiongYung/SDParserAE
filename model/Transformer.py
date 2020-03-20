import math
import numpy as np
import torch
from torch import nn

from const import *


class Transformer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, input_pad_idx: int, output_pad_idx: int, d_model: int = 16,
                 num_head: int = 4, num_e_layer: int = 6,
                 num_d_layer: int = 6, ff_dim: int = 64, drop_out: float = 0.1):
        '''
        Args:
            input_dim: Size of the vocab of the input
            output_dim: Size of the vocab for output
            num_head: Number of heads in mutliheaded attention models
            num_e_layer: Number of sub-encoder layers
            num_d_layer: Number of sub-decoder layers
            ff_dim: Dimension of feedforward network in mulihead models
            d_model: The dimension to embed input and output features into
            drop_out: The drop out percentage
        '''
        super(Transformer, self).__init__()
        self.transformer = nn.Transformer(d_model, num_head, num_e_layer, num_d_layer, ff_dim, drop_out)
        self.dec_position_encoding = PositionalEncoding(d_model)
        self.enc_position_encoding = PositionalEncoding(d_model)
        self.dec_embed = nn.Embedding(output_dim, d_model, padding_idx=output_pad_idx)
        self.enc_embed = nn.Embedding(input_dim, d_model, padding_idx=input_pad_idx)
        self.fc1 = nn.Linear(d_model, output_dim)
        self.softmax = nn.Softmax(dim=2)
        self.to(DEVICE)

    def forward(self, src: torch.Tensor, trg: torch.Tensor, src_mask: torch.Tensor = None,
                trg_mask: torch.Tensor = None):
        embedded_src = self.enc_position_encoding(self.enc_embed(src))
        embedded_trg = self.dec_position_encoding(self.dec_embed(trg))
        output = self.transformer.forward(embedded_src, embedded_trg, src_mask, trg_mask)
        return self.softmax(self.fc1(output))


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()
