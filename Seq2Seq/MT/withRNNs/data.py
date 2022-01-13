import os
import re
import random
import copy

import torch

SOS_token = 0
EOS_token = 1


class Lang():
    def __init__(self, name):
        self.SOS_token = SOS_token
        self.EOS_token = EOS_token

        self.name = name
        self.word2index = dict()
        self.word2count = dict()
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def addWord(self, word):
        if word in self.word2index:
            self.word2count[word] += 1
        else:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word2count[word] = 1
            self.n_words += 1

    def addSentence(self, sentence):
        for word in sentence.split(" "):
            self.addWord(word)


def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r"\1", s)
    s = re.sub(r"[^a-zA-Z.!?]", r" ", s)
    return s


def readLangs(file_path, lang1, lang2, reverse=False):
    if not os.path.exists(file_path):
        print("file path not exists")
        print(file_path)
        exit(0)

    pairs = []
    with open(file_path) as f:
        for l in f.readlines():
            p = []
            for s in l.split("\t"):
                p.append(normalizeString(s))
            pairs.append(p)
    print("file: '%s' Loaded.")
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


class DataConverter():
    # convert data from text to tensor
    def __init__(self, input_lang, output_lang):
        self.SOS_token = SOS_token
        self.EOS_token = EOS_token

        self.input_lang = input_lang
        self.output_lang = output_lang

    def indexFromSentence(self, lang, sentence):
        idx = []
        for w in sentence.split(" "):
            idx.append(lang.word2index[w])
        return idx

    def tensorFromSentence(self, lang, sentence):
        indexes = self.indexFromSentence(lang, sentence)
        indexes.append(self.EOS_token)
        return torch.tensor(indexes, dtype=torch.long).view(-1, 1)

    def tensorsFromPair(self, pair):
        input_tensor = self.tensorFromSentence(self.input_lang, pair[0])
        target_tensor = self.tensorFromSentence(self.output_lang, pair[1])
        return (input_tensor, target_tensor)


class MTDataset():
    # machine translation dataset
    def __init__(self, file_dir, stage, lang1, lang2, shuffle=True):
        if file_dir[-1] != '/':
            file_dir += '/'
        file_dir += '%s-%s/' % (lang1, lang2)
        file_path = file_dir + '%s.txt' % stage
        all_data_file_path = file_dir + '%s-%s.txt' % (lang1, lang2)  # 包含全部数据

        self.input_lang, self.output_lang, _ = readLangs(all_data_file_path, lang1, lang2)
        # 不管什么阶段都读取全部数据类别
        _1, _2, self.pairs = readLangs(file_path, lang1, lang2)
        for pair in self.pairs:
            self.input_lang.addSentence(pair[0])
            self.output_lang.addSentence(pair[1])
        if shuffle:
            random.shuffle(self.pairs)
        self.N = len(self.pairs)
        self.data_converter = DataConverter(self.input_lang, self.output_lang)

    def __getitem__(self, idx):
        assert -1 < idx < self.N
        pair = self.pairs[idx]
        return self.data_converter.tensorsFromPair(pair)

    def __len__(self):
        return self.N


def dataset_partition(ratios: list, dataset_origin: list, inplace: bool = False):
    """
    数据集按任意比例划分
    :param ratios:
    :param dataset_origin:
    :param inplace:
    :return:
    """
    total_ratio = sum(ratios)
    if total_ratio < 1:
        ratios.append((1 - total_ratio))
    elif total_ratio > 1:
        raise RuntimeError('total ratio greater than 1 !!!')

    assert isinstance(dataset_origin, list)

    n = len(dataset_origin)
    counts = []
    for r in ratios[:-1]:
        counts.append(int(n * r))
    counts.append(n - sum(counts))

    if inplace:
        dataset = dataset_origin
    else:
        dataset = copy.deepcopy(dataset_origin)
    random.shuffle(dataset)

    partitions = []

    start_idx = 0
    for c in counts[:-1]:
        partitions.append(dataset[start_idx:start_idx + c])
        start_idx += c

    partitions.append(dataset[start_idx:])

    return partitions
