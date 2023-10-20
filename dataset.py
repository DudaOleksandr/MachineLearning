# import numpy as np
#
# from prepro import readfile, createBatches, createMatrices, iterate_minibatches, addCharInformatioin, padding
#
# trainSentences = readfile("data/train.txt")
# devSentences = readfile("data/valid.txt")
# testSentences = readfile("data/test.txt")
#
# trainSentences = addCharInformatioin(trainSentences)
# devSentences = addCharInformatioin(devSentences)
# testSentences = addCharInformatioin(testSentences)
#
# labelSet = set()
# words = {}
#
# for dataset in [trainSentences, devSentences, testSentences]:
#     for sentence in dataset:
#         for token, char, label in sentence:
#             labelSet.add(label)
#             words[token.lower()] = True
#
# # :: Create a mapping for the labels ::
# label2Idx = {}
# for label in labelSet:
#     label2Idx[label] = len(label2Idx)
#
# # :: Hard coded case lookup ::
# case2Idx = {'numeric': 0, 'allLower': 1, 'allUpper': 2, 'initialUpper': 3, 'other': 4, 'mainly_numeric': 5,
#             'contains_digit': 6, 'PADDING_TOKEN': 7}
# caseEmbeddings = np.identity(len(case2Idx), dtype='float32')
#
# # :: Read in word embeddings ::
# word2Idx = {}
# wordEmbeddings = []
#
# fEmbeddings = open("embeddings/glove.6B.100d.txt", encoding="utf-8")
#
# for line in fEmbeddings:
#     split = line.strip().split(" ")
#     word = split[0]
#
#     if len(word2Idx) == 0:  # Add padding+unknown
#         word2Idx["PADDING_TOKEN"] = len(word2Idx)
#         vector = np.zeros(len(split) - 1)  # Zero vector vor 'PADDING' word
#         wordEmbeddings.append(vector)
#
#         word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
#         vector = np.random.uniform(-0.25, 0.25, len(split) - 1)
#         wordEmbeddings.append(vector)
#
#     if split[0].lower() in words:
#         vector = np.array([float(num) for num in split[1:]])
#         wordEmbeddings.append(vector)
#         word2Idx[split[0]] = len(word2Idx)
#
# wordEmbeddings = np.array(wordEmbeddings)
#
# char2Idx = {"PADDING": 0, "UNKNOWN": 1}
# for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
#     char2Idx[c] = len(char2Idx)
#
# train_set = padding(createMatrices(trainSentences, word2Idx, label2Idx, case2Idx, char2Idx))
# dev_set = padding(createMatrices(devSentences, word2Idx, label2Idx, case2Idx, char2Idx))
# test_set = padding(createMatrices(testSentences, word2Idx, label2Idx, case2Idx, char2Idx))
#
# idx2Label = {v: k for k, v in label2Idx.items()}
# np.save("models/idx2Label.npy", idx2Label)
# np.save("models/word2Idx.npy", word2Idx)
#
# train_batch, train_batch_len = createBatches(train_set)
# dev_batch, dev_batch_len = createBatches(dev_set)
# test_batch, test_batch_len = createBatches(test_set)
#
#
# def get_train_batch():
#     return train_batch, train_batch_len
#
#
# def get_dev_batch():
#     return dev_batch, dev_batch_len
#
#
# def get_test_batch():
#     return test_batch, test_batch_len
#
#
# def get_word_embeddings():
#     return wordEmbeddings
#
#
# def get_case_embeddings():
#     return caseEmbeddings
#
#
# def get_idx():
#     return char2Idx, label2Idx, idx2Label

import numpy as np
from prepro import readfile, createBatches, createMatrices, addCharInformatioin, padding


trainSentences = readfile("data/train.txt")
devSentences = readfile("data/valid.txt")
testSentences = readfile("data/test.txt")

trainSentences = addCharInformatioin(trainSentences)
devSentences = addCharInformatioin(devSentences)
testSentences = addCharInformatioin(testSentences)

labelSet = set()
words = set()

for dataset in [trainSentences, devSentences, testSentences]:
    for sentence in dataset:
        for token, char, label in sentence:
            labelSet.add(label)
            words.add(token.lower())

label2Idx = {label: idx for idx, label in enumerate(labelSet)}

case2Idx = {'numeric': 0, 'allLower': 1, 'allUpper': 2, 'initialUpper': 3, 'other': 4, 'mainly_numeric': 5,
            'contains_digit': 6, 'PADDING_TOKEN': 7}
caseEmbeddings = np.identity(len(case2Idx), dtype='float32')

word2Idx = {"PADDING_TOKEN": 0, "UNKNOWN_TOKEN": 1}
wordEmbeddings = [np.zeros(100), np.random.uniform(-0.25, 0.25, 100)]

for i, sentence in enumerate(trainSentences + devSentences + testSentences):
    for token, _, _ in sentence:
        token_lower = token.lower()
        if token_lower not in word2Idx:
            if token_lower in words:
                word2Idx[token_lower] = len(word2Idx)
                wordEmbeddings.append(np.random.uniform(-0.25, 0.25, 100))  # Replace the previous line
            else:
                word2Idx[token_lower] = len(word2Idx)
                wordEmbeddings.append(np.random.uniform(-0.25, 0.25, 100))

wordEmbeddings = np.array(wordEmbeddings)

char2Idx = {"PADDING": 0, "UNKNOWN": 1}
char2Idx.update({c: i + 2 for i, c in enumerate(" 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|")})

train_set = padding(createMatrices(trainSentences, word2Idx, label2Idx, case2Idx, char2Idx))
dev_set = padding(createMatrices(devSentences, word2Idx, label2Idx, case2Idx, char2Idx))
test_set = padding(createMatrices(testSentences, word2Idx, label2Idx, case2Idx, char2Idx))

idx2Label = {v: k for k, v in label2Idx.items()}
np.save("models/idx2Label.npy", idx2Label)
np.save("models/word2Idx.npy", word2Idx)

train_batch, train_batch_len = createBatches(train_set)
dev_batch, dev_batch_len = createBatches(dev_set)
test_batch, test_batch_len = createBatches(test_set)


def get_train_batch():
    return train_batch, train_batch_len


def get_dev_batch():
    return dev_batch, dev_batch_len


def get_test_batch():
    return test_batch, test_batch_len


def get_word_embeddings():
    return wordEmbeddings


def get_case_embeddings():
    return caseEmbeddings


def get_idx():
    return char2Idx, label2Idx, idx2Label


def get_train_batch():
    return train_batch, train_batch_len


def get_dev_batch():
    return dev_batch, dev_batch_len


def get_test_batch():
    return test_batch, test_batch_len


def get_word_embeddings():
    return wordEmbeddings


def get_case_embeddings():
    return caseEmbeddings


def get_idx():
    return char2Idx, label2Idx, idx2Label
