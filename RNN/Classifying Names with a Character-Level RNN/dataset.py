import os
import torch
import random

from utils import get_name_data, LetterTool, train_test_dataset_partition

class NamesClassifyDataset():
    def __init__(self):
        self.languages, self.name_data_all = get_name_data()
        self.trainset_ratio = 0.8
        assert 0 < self.trainset_ratio < 1
        self.train_test_dataset_partition()
        self.LetterTool = LetterTool()
        self.N = len(self.trainset)  # 总训练样本数
        self.C = len(self.languages)  # 总类别数
        self.input_size = self.LetterTool.n_letters

    def train_test_dataset_partition(self):
        langs = self.name_data_all.keys()
        self.trainset_origin = dict()  # language: names
        self.testset_origin = dict()

        self.trainset = []  # [[name, language],...]
        self.testset = []

        for lang in langs:
            data_all = self.name_data_all[lang]
            train_set, test_set = train_test_dataset_partition(self.trainset_ratio, data_all)
            self.trainset_origin[lang] = train_set
            self.testset_origin[lang] = test_set

            for name in train_set:
                self.trainset.append([name, lang])
            for name in test_set:
                self.testset.append([name, lang])


    def __getitem__(self, idx):
        name, language = self.trainset[idx]
        class_idx = self.languages.index(language)
        name_encoding = self.LetterTool.line2tensor(name)

        return name_encoding, class_idx


    def __len__(self):
        return self.N

