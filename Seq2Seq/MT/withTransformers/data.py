from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k


SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

token_transform = dict()
vocab_transform = dict()

token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')


def yield_token(data_iter, language):
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])


# Define special symbols and indices
UNK_IDX = 0  # unknow
PAD_IDX = 1  # pad
BOS_IDX = 2  # begin
EOS_IDX = 3  # end

special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    # create torchtext vocab object
    vocab_transform[ln] = build_vocab_from_iterator(
        yield_token(train_iter, ln),
        min_freq=1,
        specials=special_symbols,
        special_first=True
    )

# set UNK_IDX as the default index. This index is returned when the token is not found.
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[ln].set_default_index(UNK_IDX)


