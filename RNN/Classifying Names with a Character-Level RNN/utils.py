import os
import string
import torch
import random
import copy

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


def train_test_dataset_partition(trainset_ratio: float, dataset_all: list, remain_origin: bool = True):
    """
    训练集和测试级按比例划分
    :param trainset_ratio:
    :param dataset_all:
    :param remain_origin:
    :return:
    """
    assert isinstance(trainset_ratio, float) and 0 < trainset_ratio < 1
    assert isinstance(dataset_all, list)

    n = len(dataset_all)
    train_n = int(n * trainset_ratio)

    if remain_origin:
        dataset = copy.deepcopy(dataset_all)
    else:
        dataset = dataset_all

    random.shuffle(dataset)
    train_set = dataset[:train_n]
    test_set = dataset[train_n:]
    return train_set, test_set