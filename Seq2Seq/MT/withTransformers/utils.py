import torch


def generate_square_subsequent_mask(sz):
    # mask = (torch.triu(torch.ones((sz, sz)))==1).transpose(0, 1)
    mask = torch.tril(torch.ones((sz, sz), dtype=torch.long))