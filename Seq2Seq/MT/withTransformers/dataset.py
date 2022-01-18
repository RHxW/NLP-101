import torch

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k


class TrainDataset():
    def __init__(self, src_language, tgt_language):
        self.src_language = src_language
        self.tgt_language = tgt_language

        self.token_transform = dict()
        self.vocab_transform = dict()

        self.token_transform[self.src_language] = get_tokenizer('spacy', language='de_core_news_sm')
        self.token_transform[self.tgt_language] = get_tokenizer('spacy', language='en_core_web_sm')

        self.language_index = {self.src_language: 0, self.tgt_language: 1}

        # Define special symbols and indices
        self.UNK_IDX = 0  # unknow
        self.PAD_IDX = 1  # pad
        self.BOS_IDX = 2  # begin
        self.EOS_IDX = 3  # end

        self.special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

        for ln in [self.src_language, self.tgt_language]:
            train_iter = Multi30k(split='train', language_pair=(self.src_language, self.tgt_language))
            # create torchtext vocab object
            self.vocab_transform[ln] = build_vocab_from_iterator(
                self.yield_token(train_iter, ln),
                min_freq=1,
                specials=self.special_symbols,
                special_first=True
            )

        train_iter = Multi30k(split='train', language_pair=(self.src_language, self.tgt_language))
        self.raw_data = []
        for data in train_iter:
            # e.g. ('Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.\n', 'Two young, White males are outside near many bushes.\n')
            self.raw_data.append(data)
        self.N = len(self.raw_data)

        # set UNK_IDX as the default index. This index is returned when the token is not found.
        for ln in [self.src_language, self.tgt_language]:
            self.vocab_transform[ln].set_default_index(self.UNK_IDX)

        #
        self.text_transform = dict()
        for ln in [self.src_language, self.tgt_language]:
            self.text_transform[ln] = self.sequential_transforms(
                self.token_transform[ln],  # Tokenization
                self.vocab_transform[ln],  # Numericalization
                self.tensor_transform  # add BOS/EOS and create
            )

    def yield_token(self, data_iter, language):
        lang_idx = self.language_index[language]

        for data_sample in data_iter:
            yield self.token_transform[language](data_sample[lang_idx])

    def tensor_transform(self, token_ids: list):
        """
        加上开始结束标志符拼成完整张量
        :param token_ids:
        :return:
        """
        return torch.cat((
            torch.tensor([self.BOS_IDX]),
            torch.tensor(token_ids),
            torch.tensor([self.EOS_IDX])
        ))

    def __getitem__(self, idx):  # TODO
        raw_data = self.raw_data[idx]
        src_sample, tgt_sample = raw_data
        src_data = src_sample.rstrip('\n')
        tgt_data = tgt_sample.rstrip('\n')
        return src_data, tgt_data

    def sequential_transforms(self, *transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input

        return func

    def __len__(self):
        return self.N


class MaskGenerator():
    def __init__(self):
        pass
