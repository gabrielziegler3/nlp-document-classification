import os
import sys
import pickle
import pandas as pd
import dask.dataframe as dd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

from joblib import dump, load
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, classification_report, hamming_loss, zero_one_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


DATA_PATH = 'singlelabel-data.csv'
TFIDF_PATH = './tfidf.pkl'
THEMES = [5, 6, 33, 163, 181, 188, 232, 313, 339, 350, 406, 409, 424, 555, 589,
          597, 634, 660, 695, 766, 773, 793, 800, 810, 852, 895, 951]


def groupby_process(df):
    new_df = df.sort_values(['process_id', 'page'])
    new_df = new_df.groupby(
                ['process_id', 'themes'],
                group_keys=False
            ).apply(lambda x: x.body.str.cat(sep=' ')).reset_index()
    new_df = new_df.rename(index=str, columns={0: "body"})
    return new_df


def get_data(path):
    data = dd.read_csv(path)
    data = data.set_index('Unnamed: 0', sorted=True)
    data = data.rename(columns={
        'Unnamed: 0': 'index',
        'document_id': 'process_id',
        'pages': 'page'
        })
    return data


def split_data(data):
    train_body, test_body, train_labels, test_labels = train_test_split(
        data.body.compute(),
        data.themes.compute(),
        test_size=0.33,
        random_state=42,
        #stratify=data.themes.compute()
    )

    return train_body, test_body, train_labels, test_labels


def tfidf(body):
    tfidf = TfidfVectorizer(
        min_df=0.2,
        max_df=0.9,
        lowercase=False,
        use_idf=1,
        smooth_idf=1,
        #  sublinear_tf=1,
        max_features=10000,
        # ngram_range=(1, 2)
    )

    transformed = tfidf.fit_transform(body)

    print('\t\tShape', transformed.shape)

    return transformed


def tfidf_from_file(path, body):
    with open(path, 'rb') as f:
        tfidf = pickle.load(f)

    transformed = tfidf.transform(body)
    return transformed


def transform_y(train_labels, test_labels):
    le = LabelEncoder()
    le.fit(train_labels)

    le_train = le.transform(train_labels)
    le_test = le.transform(test_labels)

    print(le.classes_)

    return le_train, le_test, le


def train_pipeline(X_train, y_train):
    xgb = Pipeline([
        ('tfidf', TfidfVectorizer(
            #  vocabulary=vocab,
            min_df=0.05,
            ngram_range=(1, 1),
            use_idf=1,
            smooth_idf=1,
            sublinear_tf=1)),
        ('clf', XGBClassifier(
                    n_jobs=-1,
                    max_depth=2,
                    learning_rate=0.1,
                    n_estimators=150,
                    #  objective='binary:logistic',
                    #  tree_method='gpu_exact',
                    #  gpu_id=0,
                    #  num_class=y_train.shape[1],
                    )),
    ])

    xgb.fit(X_train, y_train)
    print("Finished training")

    return xgb


def save_model(model, path):
    print("Saving model at {} ...".format(path))
    dump(xgb, path)


def plot_confusion_matrix(cm, classes, normalized=True, cmap='bone'):
    plt.figure(figsize=[24, 16])
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    graph = sns.heatmap(norm_cm, annot=cm, fmt='g', xticklabels=classes, yticklabels=classes, cmap=cmap)
    graph.savefig("confusion_matrix.png")


def model_report(y_true, y_pred, target_names=None, name='model', label_encoder=None):
    report = classification_report(y_true, y_pred, target_names=target_names)
    print(report)
    with open('classification_report_{}.txt'.format(name), 'w') as f:
        f.write(report)

    plt.figure(figsize=(14, 10))
    try:
        cm = confusion_matrix(label_encoder.inverse_transform(y_true), label_encoder.inverse_transform(y_pred), labels=target_names)
        plot_confusion_matrix(cm, target_names)
    except Exception as e:
        cm = confusion_matrix(y_true, y_pred, labels=target_names)
        plot_confusion_matrix(cm, target_names)
        print(e)


if __name__ == '__main__':
    data = get_data(DATA_PATH)
    print(data.head())
    data.themes = data.themes.apply(lambda x: x if x in THEMES else 0)

    print('Values counts: \n', data.themes.value_counts().compute())

    train_body, test_body, train_labels, test_labels = split_data(data)
    #  CHUNKS = 3000
    CHUNKS = train_body.shape[0]

    print('Chunk size: {}'.format(CHUNKS))

    y_train, y_test, le = transform_y(train_labels[:CHUNKS], test_labels[:CHUNKS])
    X_train = train_body
    X_test = test_body

    print('Classes: ', le.classes_)

    print('We\'re classifying {} themes!'.format(y_train.shape[0]))

    print('Started Training!')
    xgb = train_pipeline(X_train, y_train)

    save_model(xgb, 'xgboost.joblib')

    print('Starting predictions on dtest!')

    train_pred = xgb.predict(X_train)
    print(train_pred[:5])
    print(y_train[:5])
    print('Train score')
    model_report(y_train, train_pred, target_names=[str(x) for x in le.classes_], name='XGBoost', label_encoder=le)

    test_pred = xgb.predict(X_test)
    print('Test score')
    model_report(y_test, test_pred, target_names=[str(x) for x in le.classes_], name='XGBoost', label_encoder=le)

