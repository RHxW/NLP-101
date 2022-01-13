import math
import torch
from torch import nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1)  # 所有位置
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.pe = pe  # 每个位置的位置编码

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, ntoken, d_model, nhead, d_hid, nlayers, dropout):
        """

        :param ntoken: size of vocabulary
        :param d_model: model/embedding dimension
        :param nhead: number of heads in Multihead self-attention layer
        :param d_hid: hidden layer dimension(feedforward network)
        :param nlayers:
        :param dropout:
        """
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)  # 单层
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)  # 多个encoder layer堆一起
        self.encoder = nn.Embedding(ntoken, d_model)

        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

    def forward(self, x, mask):
        x = self.encoder(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x, mask)
        output = self.decoder(output)

        return output