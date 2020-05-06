import numpy as np
import string
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from math import sqrt


# Reverse One hot encoding, pulls list of ingredients
# from one-hot encoded and list of ingredients
def un_onehot(one_hot, ref):
    res = []
    for i in range(len(ref)):
        if one_hot[i] == 1:
            res.append(ref[i])

    return res


# Converts a word unique value pairs
def word2vec(word):
    cw = Counter(word)
    sw = set(cw)
    lw = sqrt(sum(c*c for c in cw.values()))

    return cw, sw, lw


# calculates cosine distance between two strings
def cosdis(v1, v2):
    common = v1[1].intersection(v2[1])
    return sum(v1[0][ch]*v2[0][ch] for ch in common)/v1[2]/v2[2]


def train_correlation(X_tr, ings, cuisines):
    pred = []

    for ind in range(X_tr.shape[0]):
        print("Index {}/{}".format(ind, X_tr.shape[0]))
        element = X_tr[ind, :]
        ing_list = un_onehot(element, ings)

        sim_mat = np.empty(len(cuisines))
        for el_ing in ing_list:
            sim_scores = []
            for cuisine in cuisines:
                sim_scores.append(cosdis(word2vec(el_ing), word2vec(cuisine)))

            sim_mat = np.vstack((sim_mat, sim_scores))

        sim_mat = sim_mat[1:, :].flatten()
        best_match = np.argmax(sim_mat) % 20
        pred.append(cuisines[best_match])

    return pred


def accuracy_corrolation(yhat, y):
    acc = np.char.equal(yhat, y)
    return np.count_nonzero(acc) / len(y)
