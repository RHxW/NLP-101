import torch
from torch import nn

def prepare_sequence(seq, to_ix):
    """
    获取数据和对应词性标签
    :param seq:
    :param to_ix:
    :return:
    """
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

training_data = [
    # Tags are: DET - determiner; NN - noun; V - verb
    # For example, the word "The" is a determiner
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]

word_to_idx = dict()
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)

print("word_to_idx: ", word_to_idx)

tag_to_ix = {"DET": 0, "NN": 1, "V": 2}  # Assign each tag with a unique index

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, X):
        embeds = self.word_embeddings(X)
        lstm_out, _ = self.lstm(embeds.view(len(X), 1, -1))  # 用 .view 把shape变为(1, -1)
        tag_space = self.hidden2tag(lstm_out.view(len(X), -1))
        return tag_space

# train the LSTM model
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_idx), len(tag_to_ix))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

# inputs = prepare_sequence(training_data[0][0], word_to_idx)
# tag_score = model(inputs)
# print(tag_score)

EPOCHS = 2
for epoch in range(EPOCHS):
    for sentence, tags in training_data:
        sentence_input = prepare_sequence(sentence, word_to_idx)
        targets = prepare_sequence(tags, tag_to_ix)

        pred = model(sentence_input)
        loss = criterion(pred, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

model.eval()
inputs = prepare_sequence(training_data[0][0], word_to_idx)
label = prepare_sequence(training_data[0][1], tag_to_ix)
tagscore = model(inputs)
output = torch.nn.functional.softmax(tagscore)
print(output)
print(label)
print(torch.argmax(output, -1))