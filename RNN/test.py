import os
import string
import torch

from rnn import RNN


def get_name_data():
    data_dir = 'data/names/'
    name_files = os.listdir(data_dir)

    language_names = dict()
    all_languages = []

    for nf in name_files:
        lang = nf.split('.')[0]
        all_languages.append(lang)
        nf_path = data_dir + nf
        with open(nf_path) as f:
            names = []
            for ln in f.readlines():
                _name = ln.split("\n")[0]
                if _name:
                    names.append(_name)
            language_names[lang] = names

    return all_languages, language_names

class LetterTool():
    # letter to one-hot encoding tensor
    def __init__(self):
        self.all_letters = string.ascii_letters + ".,;'"
        self.n_letters = len(self.all_letters)

    def letter2index(self, letter):
        return self.all_letters.find(letter)

    def letter2tensor(self, letter):
        tensor = torch.zeros(1, self.n_letters)
        _idx = self.letter2index(letter)
        tensor[0][_idx] = 1
        return tensor

    def line2tensor(self, line):
        tensor = torch.zeros(len(line), 1, self.n_letters)
        for i, letter in enumerate(line):
            _idx = self.letter2index(letter)
            tensor[i][0][_idx] = 1

        return tensor

def test():
    all_languages, language_names = get_name_data()
    N = len(all_languages)
    hidden_size = 128
    lt = LetterTool()
    X = lt.letter2tensor('A')
    rnn = RNN(lt.n_letters, hidden_size, N)
    hidden = torch.zeros(1, hidden_size)
    output, next_hidden = rnn(X, hidden)
    print(output.shape)
    print(next_hidden.shape)

if __name__ =='__main__':
    test()