import torch
from torch import nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, x, hidden):
        embedding = self.embedding(x).view(1, 1, -1)
        output = embedding
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

class DecoderRNN(nn.Module):
    # simple decoder of RNN
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        output = self.embedding(x).view(1, 1, -1)
        output = F.relu(output, inplace=True)
        output, hidden = self.gru(output, hidden)
        output = self.linear(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)