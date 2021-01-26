# Author
#
# Gabriel G. Ziegler

import os
import sys
import pickle
import string
import pandas as pd
import numpy as np
import statistics
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as offline
import plotly.graph_objs as go
import plotly.figure_factory as ff
offline.init_notebook_mode()

from operator import itemgetter
from tqdm import tnrange, tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, hamming_loss, zero_one_loss
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from xgboost import XGBClassifier
from helpers import data_stats, hamming_loss_non_zero, groupby_process, show_metrics, get_labels

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

chi2 = pd.read_csv("../data/Chi2-30-08-2018.csv", index_col=0)
chi2_words = chi2.iloc[:, :27]
print(len(chi2_words.columns), chi2_words.columns)
chi2_scores = chi2.iloc[:, 27:]
print(len(chi2_scores.columns), chi2_scores.columns)

n_words = 100
vocab = []
for i in chi2_words.columns:
    vocab.extend(chi2[i].iloc[:n_words].astype(str).tolist())

vocab = set(vocab)
print("Vocabulary size: {}".format(len(vocab)))

train = pd.read_csv("../data/train.csv")
validation = pd.read_csv("../data/validation.csv")
test = pd.read_csv("../data/test.csv")

print("{}\n {}\n {}\n".format(train.head(), validation.head(), test.head()))

corpus = list(map(str, train.body))
corpus.extend(list(map(str, validation.body)))
print(data_stats(corpus))

labels = list(train.themes)
labels.extend(validation.themes)
labels = get_labels(labels)
# labels[:5]

test_labels = list(test.themes)
test_labels = get_labels(test_labels)
# test_labels[:5]

X_train, X_val, y_train, y_val = train_test_split(
        corpus, labels, test_size=0.33, random_state=42)

mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(y_train)
y_val = mlb.fit_transform(y_val)
test_labels = mlb.transform(test_labels)
#  print(len(set(mlb.classes_)), set(mlb.classes_))

xgb = Pipeline([
    ('tfidf', TfidfVectorizer(vocabulary=vocab,
                              min_df=0.05,
                              max_features=None,
                              analyzer='word',
                              ngram_range=(1, 5),
                              use_idf=1,
                              smooth_idf=1,
                              sublinear_tf=1,)),
    ('clf', OneVsRestClassifier(
        XGBClassifier(n_jobs=-1, max_depth=6), n_jobs=-1))
])

print("fitting on training set")
xgb.fit(X_train, y_train)


print("fitting on validation set")
xgb_val = xgb.predict(X_val)

#  print('Validation dataset metrics:\n{}'.format(
#      show_metrics(mlb, y_val, xgb_val, target_names=[str(x) for x in mlb.classes_])))

xgb_test = xgb.predict(test.body)

print('Test dataset metrics:\n{}\n\n\n\n'.format(
    show_metrics(mlb, test_labels, xgb_test, target_names=[str(x) for x in mlb.classes_])))
