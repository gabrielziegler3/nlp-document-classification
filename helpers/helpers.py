import pandas as pd
import numpy as np
import statistics

#  import seaborn as sns
#  import matplotlib.pyplot as plt
#  import plotly.offline as offline
#  import plotly.graph_objs as go
#  import plotly.figure_factory as ff
#  offline.init_notebook_mode()

from operator import itemgetter
from tqdm import tnrange, tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, hamming_loss, zero_one_loss
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer


def bar_plot(x, y, title=""):
    trace = go.Bar(
        x=x,
        y=y,
        text=y,
        textposition='auto',
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
    layout = dict(title=title)
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


def data_stats(data):
    len_docs = [len(str(x).split()) for x in data]
    print('max: %d' % max(len_docs))
    print('min: %d' % min(len_docs))
    print('mean: %d' % (sum(len_docs) / float(len(len_docs))))
    print('std dev: %d' % statistics.stdev(len_docs))


def groupby_process(df):
    new_df = df.sort_values(['process_id', 'page'])
    new_df = new_df.groupby(
                ['process_id', 'themes'],
                group_keys=False
            ).apply(lambda x: x.body_pre_processed.str.cat(sep=' ')).reset_index()
    new_df = new_df.rename(index=str, columns={0: "body"})
    return new_df


def show_metrics(y_true, y_pred, target_names):
    """
    Both y_true and y_pred must be already transformed to MultiLabelBinarizer
    """
    print("Hamming Loss: {}".format(hamming_loss(y_true, y_pred)))
    print("Zero One Loss: {}".format(zero_one_loss(y_true, y_pred)))
    print("Hamming Loss Non Zero: {}\n".format(hamming_loss_non_zero(y_true, np.array(y_pred))))
    print(classification_report(y_true, y_pred, target_names=target_names))


def get_labels(labels):
    new_labels = list(labels)
    if not isinstance(labels[0], list):
        new_labels = list(map(lambda x: x.split(','), new_labels))
    new_labels = list(map(lambda x: [int(i) for i in x], new_labels))
    return new_labels


def change_themes(themes, keepers):
    f = lambda x: [i if i in keepers else 0 for i in x]
    unique = lambda x: list(sorted(set(x)))

    themes = [eval(theme) for theme in themes]
    themes = list(map(f, tqdm(themes)))
    themes = list(map(unique, themes))
    return themes

