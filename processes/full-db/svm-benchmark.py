import os
import sys
import pickle
import pandas as pd
import dask.dataframe as dd
import numpy as np
import xgboost as xgb

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


DATA_PATH = '../../data/balanced-data.csv'
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
    data = pd.read_csv(path)
    data = data.rename(columns={'document_id': 'process_id', 'pages': 'page'})
    #print(data.head())
    return data


def predict(dtest, clf):
    pred = clf.predict(dtest)
    return pred


def split_data(data):
    data.themes = data.themes.apply(lambda x: 'Theme ' + str(x))
    train_body, test_body, train_labels, test_labels = train_test_split(
        data.body,
        data.themes,
        test_size=0.33,
        random_state=42,
        stratify=data.themes
    )

    return train_body, test_body, train_labels, test_labels


def tfidf(train_body, test_body):
    tfidf = TfidfVectorizer(
        min_df=0.2,
        max_df=0.9,
        lowercase=False,
        use_idf=1,
        smooth_idf=1,
        #  sublinear_tf=1,
        # max_features=20000
    )

    X_train = tfidf.fit_transform(train_body)
    X_test = tfidf.transform(test_body)

    print('\tX_train', X_train.shape)
    print('\t\tX_test', X_test.shape)

    return X_train, X_test


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


def report(y_true, pred):
    try:
        print(classification_report(y_true, pred))
        # print(confusion_matrix(y_true, pred))
    except Exception as e:
        print(e)


if __name__ == '__main__':
    #  dtrain, dtest = get_data('./dtrain', './dtest')

    CHUNKS = 30000

    data = get_data(DATA_PATH)
    #print(data.head())

    print('Values counts: \n', data.themes.value_counts())

    train_body, test_body, train_labels, test_labels = split_data(data)
    #print(train_labels.head(100))

    print('Chunk size: {}'.format(CHUNKS))

    try:
        X_train, X_test = tfidf(train_body, test_body)
    except Exception as e:
        print(e)

    y_train, y_test, le = transform_y(train_labels, test_labels)

    print('We\'re classifying {} themes!'.format(len(le.classes_)))

    print('Started Training!')
    svc = OneVsRestClassifier(
            LinearSVC(),
            n_jobs=-1
            )

    svc.fit(X_train, y_train)
    print("Finished training")

    print('deleting obsolette vars')
    del X_train, y_train

    print('Starting predictions on dtest!')
    pred = svc.predict(X_test)

    print(y_test[:5], pred[:5])
    print(pred.max(), pred.min(), pred.mean(), pred.sum())
    #print(le.inverse_transform(y_test))
    #print(le.inverse_transform(pred.astype(int)))
    report(le.inverse_transform(y_test), le.inverse_transform(pred.astype(int)))

