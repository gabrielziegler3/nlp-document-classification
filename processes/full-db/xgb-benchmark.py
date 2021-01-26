import os
import sys
import pickle
import pandas as pd
import dask.dataframe as dd
import numpy as np
import xgboost as xgb

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


DATA_PATH = '../../data/balanced-data.csv'
TFIDF_PATH = './tfidf.pkl'
THEMES = [5, 6, 33, 163, 181, 188, 232, 313, 339, 350, 406, 409, 424, 555, 589,
          597, 634, 660, 695, 766, 773, 793, 800, 810, 852, 895, 951]
WEIGHTS = [
        0.002393223379933684, 0.004455950355909849, 0.004555198513198374,
        0.009302355224611798, 0.009454127165733949, 0.010012254183437467,
        0.017559370026255856, 0.019754291279537838, 0.019810571311673277,
        0.02066422148706484, 0.023139802097828014, 0.02410228953343958,
        0.027925745101997263, 0.027981933723932872, 0.02826630296909479,
        0.02915518042095312, 0.031823846821040364, 0.03593545493745385,
        0.036986758140411274, 0.037688403958793054, 0.04604973861190278,
        0.04746423570237078, 0.0549684626908879, 0.06180898249242062,
        0.06236332314257686, 0.07641220363073978, 0.09089556248885386,
        0.1390702106079464
    ]


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
    data = data.rename(columns={'document_id': 'process_id', 'pages': 'page'})
    print(data.head())
    return data


def get_dmatrix(X, y=None, weight=None):
    dmatrix = xgb.DMatrix(X, label=y, weight=weight)
    return dmatrix


def train_model(dtrain, num_class, num_round):
    param = {
        'max_depth': 6,  # the maximum depth of each tree
        'eta': 0.1,  # the training step for each iteration
        'objective': 'multi:softmax',  # error evaluation for multiclass
        'num_class': num_class,
        # 'gpu_id': 0,
        # 'max_bin': 16,
        # 'max_delta_step': 5,
        # 'tree_method': 'gpu_hist',
    }
    num_round = num_round
    print('Booster params:', param)

    bst = xgb.train(param, dtrain, num_round)
    return bst


def predict(dtest, clf):
    pred = clf.predict(dtest)
    return pred


def split_data(data):
    data.themes = data.themes.apply(lambda x: 'Theme ' + str(x))
    train_body, test_body, train_labels, test_labels = train_test_split(
        data.body.compute(),
        data.themes.compute(),
        test_size=0.33,
        random_state=42,
        stratify=data.themes.compute()
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
        max_features=20000
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
    CHUNKS = 30000

    data = get_data(DATA_PATH)
    print(data.head())

    print('Values counts: \n', data.themes.value_counts().compute())

    train_body, test_body, train_labels, test_labels = split_data(data)
    print(train_labels.head(100))

    print('Chunk size: {}'.format(CHUNKS))

    try:
        X_train, X_test = tfidf(train_body[:CHUNKS], test_body[:CHUNKS])
    except Exception as e:
        print(e)

    y_train, y_test, le = transform_y(train_labels[:CHUNKS], test_labels[:CHUNKS])

    dtrain = get_dmatrix(X_train, y_train, WEIGHTS)
    print('We\'re classifying {} themes!'.format(len(le.classes_)))

    print('Started Training!')
    bst = train_model(dtrain, len(le.classes_), 10)
    print("Finished training")

    bst.save_model('bst.model')

    print('deleting obsolette vars')
    del X_train, y_train, dtrain, data
    dtest = get_dmatrix(X_test, y_test)

    print('Starting predictions on dtest!')
    pred = predict(dtest, bst)

    print(y_test[:5], pred[:5])
    print(pred.max(), pred.min(), pred.mean(), pred.sum())

    report(le.inverse_transform(y_test), le.inverse_transform(pred.astype(int)))

