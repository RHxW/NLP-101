import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from model import Seq2SeqTransformer
from dataset import TrainDataset

def train(src_language, tgt_language, device, lr):
    train_dataset = TrainDataset(src_language, tgt_language)
    vocab_transform = train_dataset.vocab_transform
    SRC_VOCAB_SIZE = len(vocab_transform[src_language])
    TGT_VOCAB_SIZE = len(vocab_transform[tgt_language])
    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 512
    BATCH_SIZE = 128
    NUM_ENCODER_LAYER = 3
    NUM_DECODER_LAYER = 3
    dropout = 0.1

    transformer = Seq2SeqTransformer(
        num_encoder_layers=NUM_ENCODER_LAYER,
        num_decoder_layers=NUM_DECODER_LAYER,
        emb_size=EMB_SIZE,
        nhead=NHEAD,
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        dim_feedforward=FFN_HID_DIM,
        dropout=dropout
    )
    transformer = transformer.to(device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=train_dataset.PAD_IDX)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True
    )

    for i, data in enumerate(dataloader):
        print(i)
        print(data)


if __name__ == '__main__':
    src_language = 'de'
    tgt_language = 'en'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 1e-3

    train(src_language, tgt_language, device, lr)