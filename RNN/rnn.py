import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.W = nn.Linear(input_size, hidden_size)  # input to hidden
        self.U = nn.Linear(hidden_size, hidden_size)  # pre hidden to hidden
        self.V = nn.Linear(hidden_size, output_size)  # hidden to output
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X, pre_hidden):
        hidden_1 = self.U(pre_hidden)
        hidden_1 = self.relu(hidden_1)
        hidden_2 = self.W(X)
        hidden_2 = self.relu(hidden_2)
        hidden = hidden_1 + hidden_2
        output = self.V(hidden)
        output = self.softmax(output)
        return output, hidden
