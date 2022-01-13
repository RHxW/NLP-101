import torch

from model import EncoderRNN, DecoderRNN, AttnDecoderRNN
from data import MTDataset

def test(lang1, lang2, weight_path, decoder_with_attn=False, shuffle=True):
    pass