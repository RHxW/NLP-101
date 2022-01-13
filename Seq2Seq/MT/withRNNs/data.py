import os
import re
import random

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


def readLangs(file_dir, lang1, lang2, reverse=False):
    file_name = '%s-%s.txt' % (lang1, lang2)
    if not file_dir[-1] == '/':
        file_dir += '/'
    file_path = file_dir + file_name
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

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


class DataConverter():
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
    def __init__(self, file_dir, lang1, lang2, shuffle=True, trainset_ratio=0.8):
        self.input_lang, self.output_lang, self.pairs = readLangs(file_dir, lang1, lang2)
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