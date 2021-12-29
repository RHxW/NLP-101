import re
import math
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split

# dataset: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection#

def SMSSpamDetect():
    text_raw = './smsspamcollection/SMSSpamCollection'
    with open(text_raw) as f:
        lines = f.readlines()

    labels = []
    data = []
    for line in lines:
        l, d = line.split('\t')
        labels.append(l)
        data.append(d)

    # 一个区分词性的方法
    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    rule = re.compile("[^a-zA-Z\d ]")
    wnl = WordNetLemmatizer()
    for i in range(len(data)):
        # 用正则表达式去除data中的标点符号，并全部转为小写
        data[i] = re.sub("[^a-zA-Z\d ]", "", data[i].lower())

        tokens = data[i].split(' ')  # 分词

        tks = []
        for w in tokens:
            if w in stopwords.words('english'):
                continue
            tks.append(w)
        tagged_sent = nltk.pos_tag(tks)  # 获取单词词性

        lemmas_sent = []
        for tag in tagged_sent:
            wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
            lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
            data[i] = lemmas_sent

    # 计算标签的基础概率
    pos, neg = 0, 0
    for l in labels:
        if l == 'ham':
            neg += 1  # 正常短信算负类
        else:
            pos += 1
    base_prob = (pos / (pos + neg), neg / (pos + neg))

    # 计算每个单词属于各个类别的概率
    n = len(data)
    word_dict = dict()
    for i in range(n):
        lab = labels[i]
        d = list(set(data[i]))

        for w in d:
            if w not in word_dict:
                word_dict[w] = {'ham': 1, 'spam': 1}
            word_dict[w][lab] += 1

    for w in word_dict:
        dt = word_dict[w]
        ham = dt['ham']
        spam = dt['spam']
        word_dict[w]['ham'] = ham / (ham + spam)
        word_dict[w]['spam'] = spam / (ham + spam)

    # predict
    # TODO
    res = []


if __name__ == "__main__":
    SMSSpamDetect()
