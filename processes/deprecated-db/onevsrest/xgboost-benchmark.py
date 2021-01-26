# Author
# 
# Gabriel G. Ziegler

from operator import itemgetter
from tqdm import tnrange
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, hamming_loss, zero_one_loss
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from xgboost import XGBClassifier

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import os
import sys
import pickle
import string
import pandas as pd
import numpy as np
import statistics

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as offline
import plotly.graph_objs as go
import plotly.figure_factory as ff
offline.init_notebook_mode()


def bar_plot(x, y, title=""):
    trace = go.Bar(
        x = x,
        y = y,
        text = y,
        textposition = 'auto',
        textfont=dict(
            family='sans serif',
            size=16,
            color='#ffffff'
        ),
        marker=dict(
                color='rgb(66, 69, 244)'
            )
        )

    data = [trace]
    layout = dict(title = title)
    fig = dict(data=data, layout=layout)
    offline.iplot(fig)
    
def plot_confusion_matrix(cm, classes, normalized=True, cmap='bone'):
    plt.figure(figsize=[24, 16])
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(norm_cm, annot=cm, fmt='g', xticklabels=classes, yticklabels=classes, cmap=cmap)
    
def hamming_loss_non_zero(Y_test, np_pred, debug=False):
    max_hamming = 0
    for i in range(0,len(Y_test)):
        temp_hamming = 0
        max_pred = 0
        error_pred = 0
        for j in range(len(Y_test[i])):
            if Y_test[i][j] == np_pred[i][j] == 0:
                pass

            if Y_test[i][j] != 0:
                max_pred+=1

            if Y_test[i][j] != np_pred[i][j]:
                error_pred+=1
        if max_pred != 0:
            temp_hamming = float(error_pred)/float(max_pred)
            if temp_hamming > 1.0:
                temp_hamming = 1.0
            if debug:
                print("MAX: {}  ERROR: {} HAMMING: {}".format(max_pred, error_pred, temp_hamming))
            max_hamming+=temp_hamming
    return float(max_hamming)/float(len(Y_test))
    
def tokenize(corpus, maxlen=1000, vocab_size=10000):
    print('Using vocab size {} and maxlen {}'.format(vocab_size, maxlen))
    tokenizer = Tokenizer(num_words=vocab_size, lower=False)
    tokenizer.fit_on_texts(corpus)
    print('Text 2 Sequence step initialized')
    sequences = tokenizer.texts_to_sequences(corpus)
    print('Finished text2sequence')
    data = pad_sequences(sequences, maxlen=maxlen)
    return pd.DataFrame(data)

def binarize_label(labels):
    mlb = MultiLabelBinarizer()
    labels_ohe = mlb.fit_transform(labels)
    return mlb, labels_ohe

def data_stats(data):
    len_docs = [len(str(x).split()) for x in data]
    print('max: %d' % max(len_docs))
    print('min: %d' % min(len_docs))
    print('mean: %d' % (sum(len_docs) / float(len(len_docs))))
    print('std dev: %d' % statistics.stdev(len_docs))
    
def groupby_process(df):
    new_df = df.sort_values(['process_id', 'page'])
    new_df = new_df.groupby(['process_id', 'themes'], group_keys=False).apply(lambda x: x.body_pre_processed.str.cat(sep=' ')).reset_index()
    new_df = new_df.rename(index=str, columns={0:"body"})
    return new_df

def show_metrics(mlb, y_true, y_pred, target_names):
    """
    Both y_true and y_pred must be already transformed
    """
    print("Hamming Loss: {}".format(hamming_loss(y_true, y_pred)))
    print("Zero One Loss: {}".format(zero_one_loss(y_true, y_pred)))
    print("Hamming Loss Non Zero: {}\n".format(hamming_loss_non_zero(y_true, np.array(y_pred))))
    print(classification_report(y_true, y_pred, target_names=target_names))
    
def nestrepl(lst, what, repl):
    for index, item in enumerate(lst):
        if isinstance(item, list):
            nestrepl(item, what, repl)
        else:
            if item == what:
                lst[index] = repl
    return lst

def get_labels(labels):
    new_labels = list(labels)
    if not isinstance(labels[0], list):
        new_labels = list(map(lambda x: x.split(','), new_labels))
    new_labels = list(map(lambda x: [int(i) for i in x], new_labels))
    return new_labels

def head(df):
    table = ff.create_table(df)
    offline.iplot(table)

def get_vocab(n_words):
    vocab = []
    for i in chi2_words.columns:
        vocab.extend(chi2[i].iloc[:n_words].astype(str).tolist())
    
    vocab[:10]
    vocab = set(vocab)
    print("Vocabulary size: {}".format(len(vocab)))
    return vocab

print('Reading chi squared data ...')

chi2 = pd.read_csv("../data/Chi2-30-08-2018.csv", index_col=0)

chi2_words = chi2.iloc[:, :27]
print(len(chi2_words.columns), chi2_words.columns)

chi2_scores = chi2.iloc[:, 27:]
print(len(chi2_scores.columns), chi2_scores.columns)

train = pd.read_csv("../data/train.csv")
validation = pd.read_csv("../data/validation.csv")
test = pd.read_csv("../data/test.csv")

# train = groupby_process(train)
# print("{}\n {}\n {}\n".format(train.head(), validation.head(), test.head()))

# validation = groupby_process(validation)

# test = groupby_process(test)

mlb = MultiLabelBinarizer()
corpus = list(map(str, train.body))
corpus.extend(list(map(str, validation.body)))

labels = list(train.themes)
labels.extend(validation.themes)
labels = get_labels(labels)

test_labels = list(test.themes)
test_labels = get_labels(test_labels)

X_train, X_val, y_train, y_val = train_test_split(corpus, labels, test_size=0.33, random_state=42) 

mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(y_train)
y_val = mlb.fit_transform(y_val)
test_labels = mlb.transform(test_labels)

# OneVsRest - eXtreme Gradient Boosting

vocab_sizes = [2000, 3000, 4000, 5000, 10000]

y_pred = []

print('Starting benchmark training')

for i in vocab_sizes:
    vocab = get_vocab(i)

    print("Training model with {} vocab size\n".format(len(vocab)))
    
    xgb = Pipeline([
        ('vectorizer', CountVectorizer(vocabulary=vocab,
                                       lowercase=False)),
#        ('tfidf', TfidfVectorizer(min_df=0.05,
#                                  max_features=None,
#                                  analyzer='word',
#                                  ngram_range=(1, 2),
#                                  use_idf=1,
#                                  smooth_idf=1,
#                                  sublinear_tf=1,)),
        ('clf', OneVsRestClassifier(XGBClassifier(n_jobs=-1, max_depth=4), n_jobs=-1))
    ])
 
    xgb.fit(X_train, y_train)
    
    # Validation dataset metrics

    # xgb_val = xgb.predict(X_val)
    
    # print('Validation dataset metrics:\n{}'.format(show_metrics(mlb, y_val, xgb_val, target_names=[str(x) for x in mlb.classes_])))
    
    # Test dataset metrics
    
    xgb_test = xgb.predict(test.body)
    y_pred.append(xgb_test) 
    # print('Test dataset metrics:\n{}'.format(show_metrics(mlb, test_labels, xgb_test, target_names=[str(x) for x in mlb.classes_])))
    print('\n\n\n\n')

with open("predictions.txt", "w") as f:
    f.write(y_pred)

