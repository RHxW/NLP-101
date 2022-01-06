import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

test_sentence = """n-gram models are widely used in statistical natural 
language processing . In speech recognition , phonemes and sequences of 
phonemes are modeled using a n-gram distribution . For parsing , words 
are modeled such that each n-gram is composed of n words . For language 
identification , sequences of characters / graphemes ( letters of the 
alphabet ) are modeled for different languages For sequences of characters , 
the 3-grams ( sometimes referred to as " trigrams " ) that can be 
generated from " good morning " are " goo " , " ood " , " od " , " dm ", 
" mo " , " mor " and so forth , counting the space character as a gram 
( sometimes the beginning and end of a text are modeled explicitly , adding 
" __g " , " _go " , " ng_ " , and " g__ " ) . For sequences of words , 
the trigrams that can be generated from " the dog smelled like a skunk " 
are " # the dog " , " the dog smelled " , " dog smelled like ", " smelled 
like a " , " like a skunk " and " a skunk # " .""".split()


# vocab = set(test_sentence)

def get_ngram(text, N):
    grams = []
    L = len(text)
    if N - 1 <= 0:  # unigram
        for i in range(L):
            grams.append((text[i],))
        return grams
    for i in range(L - N + 1):  # N-gram
        tmp = []
        for j in range(N - 1):
            tmp.append(text[i + j])
        grams.append((tmp, text[i + N - 1]))
    return grams


class NGramDataset():
    def __init__(self, text: list, N: int):
        assert isinstance(text, list)
        assert isinstance(N, int), "N must be int in [1, 10]"
        assert 0 < N < 11, "invalid N(%d)" % N
        self.grams = get_ngram(text, N)
        self.L = len(self.grams)
        self.vocab = set(text)
        self.vocab_size = len(self.vocab)
        self.word2idx = dict()
        self.idx2word = dict()
        for i, word in enumerate(self.vocab):
            self.word2idx[word] = i
            self.idx2word[i] = word

    def __getitem__(self, idx):
        gram = self.grams[idx]
        x = []
        for w in gram[0]:
            x.append(self.word2idx[w])
        y = self.word2idx[gram[-1]]
        x = torch.tensor(x)
        y = torch.tensor(y)
        return x, y

    def __len__(self):
        return self.L


class NGramNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.L1 = nn.Linear(context_size * embedding_dim, 128)
        self.L2 = nn.Linear(128, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x).reshape([x.shape[0], -1])
        out = self.L1(embeds)
        out = F.relu(out, inplace=True)
        out = self.L2(out)
        return out


if __name__ == "__main__":
    N = 3
    context_size = N - 1
    dataset = NGramDataset(test_sentence, N)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=4,
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    model = NGramNetwork(dataset.vocab_size, 16, context_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    EPOCHS = 100

    for epoch in range(EPOCHS):
        total_loss = 0
        for i, (data, target) in enumerate(dataloader):
            pred = model(data)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(total_loss)

    # eval/test
    print("-"*50)
    model.eval()
    test_examples = [
        ["widely", "used"],
        ["and", "so"],
        ["are", "modeled"]
    ]
    for te in test_examples:
        l = len(te)
        if l != N - 1:
            print("test example does not fit the ngram model.(%s)"% str(te))
            continue
        x = [dataset.word2idx[w] for w in te]
        x = torch.tensor(x).unsqueeze(0)
        pred = F.softmax(model(x), dim=1)
        pred = torch.argmax(pred).item()
        res = dataset.idx2word[pred]
        print("{} + {} = {}".format(te[0], te[1], res))
