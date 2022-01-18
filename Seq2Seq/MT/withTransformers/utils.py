import torch


def generate_square_subsequent_mask(sz):
    # mask = (torch.triu(torch.ones((sz, sz)))==1).transpose(0, 1)
    mask = torch.tril(torch.ones((sz, sz), dtype=torch.long) == 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0))
    return mask


def create_mask(src, tgt, pad_idx=1):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len)) != 0

    src_padding_mask = (src == pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask