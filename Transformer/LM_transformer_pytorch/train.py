import math

import torch

from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from model import TransformerModel

train_iter = WikiText2(split='train')
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])


def data_process(raw_text_iter):
    data = []
    for item in raw_text_iter:
        tok = tokenizer(item)
        voc = vocab(tok)
        data.append(torch.tensor(voc, dtype=torch.long))

    # data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


train_iter, val_iter, test_iter = WikiText2()
train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)


def make_batch(data, batch_size):
    seq_len = data.size(0) // batch_size
    data = data[:seq_len * batch_size]
    data = data.view(batch_size, seq_len).t().contiguous()
    return data


batch_size = 20
eval_batch_size = 10
train_data = make_batch(train_data, batch_size)
val_data = make_batch(val_data, eval_batch_size)
test_data = make_batch(test_data, eval_batch_size)

bptt = 35  # subdivides the source data into chunks of length bptt


def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    # target = source[i + 1:i + 1 + seq_len].reshape(-1)
    target = source[i + 1:i + 1 + seq_len]
    return data, target


def generate_square_subsequent_mask(size):
    # 生成上三角的mask矩阵，对角线上为0
    return torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)


ntoken = len(vocab)
d_model = 200
d_hid = 200
nlayers = 2
nhead = 2
dropout = 0.2
model = TransformerModel(ntoken, d_model, nhead, d_hid, nlayers, dropout)

### train
import copy
import time

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1.0, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)


def train(epoch):
    model.train()
    total_loss = 0
    log_interval = 200
    start_time = time.time()
    src_mask = generate_square_subsequent_mask(bptt)

    num_batches = len(train_data) // bptt

    for i in range(num_batches):
        data, target = get_batch(train_data, i)
        batch_size = data.size(0)
        _mask = src_mask
        if batch_size != bptt:
            _mask = src_mask[:batch_size, :batch_size]

        output = model(data, _mask)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if i % log_interval == 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp((cur_loss))
            print(f'| epoch {epoch:3d} | {i:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()


EPOCHS = 2
for epoch in range(EPOCHS):
    train(epoch)