import os
import sys
import pickle
import pandas as pd
import dask.dataframe as dd
import numpy as np
import gc

from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
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
    data = dd.read_csv(path)
    data = data.rename(columns={'document_id': 'process_id', 'pages': 'page'})
    print(data.head())
    return data


def split_data(data):
    data.themes = data.themes.apply(lambda x: 'Theme ' + str(x))
    train_body, test_body, train_labels, test_labels = train_test_split(
        data.body.compute(),
        data.themes.compute(),
        test_size=0.3,
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


def __transform_y(train_labels, test_labels):
    le = LabelEncoder()
    le.fit(train_labels)

    le_train = le.transform(train_labels)
    le_test = le.transform(test_labels)

    print(le.classes_)

    return le_train, le_test, le


def transform_y(train_labels, test_labels):
    le = LabelBinarizer()
    le.fit(train_labels)

    le_train = le.transform(train_labels)
    le_test = le.transform(test_labels)

    print(le.classes_)

    return le_train, le_test, le


def final_prediction(xgb_pred, log_pred, clf1_weight, clf2_weight, threshold=0.5):
    final_pred = []
    # assert(len(xgb_pred) == len(log_pred))
    # assert(type(xgb_pred[0]))

    for i in range(len(xgb_pred)):
        final_pred.append(np.zeros(xgb_pred.shape[0]))
        weighted_pred = xgb_pred[i] * clf1_weight + log_pred[i] * clf2_weight
        final_pred[i][np.argmax(weighted_pred)] = 1

    print('Final pred', final_pred[:5], final_pred.shape)
    #  final_pred = np.vstack(final_pred).astype(int)

    # assert(len(xgb_pred) == len(final_pred))
    return final_pred


def report(y_true, pred):
    try:
        print(classification_report(y_true, pred))
        # print(confusion_matrix(y_true, pred))
    except Exception as e:
        print(e)


if __name__ == '__main__':
    data = get_data(DATA_PATH)
    print(data.head())

    print('Values counts: \n', data.themes.value_counts().compute())

    train_body, test_body, train_labels, test_labels = split_data(data)
    print(train_labels.head(100))

    CHUNKS = 10000
    print('Chunk size: {}'.format(CHUNKS))

    try:
        X_train, X_test = tfidf(train_body[:CHUNKS], test_body[:CHUNKS])
    except Exception as e:
        print(e)

    y_train, y_test, le = transform_y(train_labels[:CHUNKS], test_labels[:CHUNKS])

    print('We\'re classifying {} themes!'.format(len(le.classes_)))

    print('Started Training!')
    xgb = OneVsRestClassifier(
            XGBClassifier(
                n_jobs=-1,
                max_depth=3,
                gpu_id=0,
                tree_method='gpu_hist',
                num_round=100
                ),
            n_jobs=-1
            )

    logres = OneVsRestClassifier(LogisticRegression(n_jobs=-1), n_jobs=-1)

    xgb.fit(X_train, y_train)

    logres.fit(X_train, y_train)

    print("Finished training")

    print('deleting obsolette vars')
    del X_train, y_train
    gc.collect()

    print('Starting predictions on test set!')

    #  train_pred = xgb.predict(X_train)

    #  print('Train score')
    #  report(le.inverse_transform(y_train), le.inverse_transform(train_pred.astype(int)))

    xgb_pred_proba_test = xgb.predict_proba(X_test)
    log_pred_proba_test = logres.predict_proba(X_test)

    print(xgb_pred_proba_test[:5], log_pred_proba_test[:5])

    test_pred = final_prediction(xgb_pred_proba_test, log_pred_proba_test, 0.6, 0.4, threshold=0.5)

    print(test_pred[:5])

    try:
        print(classification_report(y_test, test_pred, target_names=le.classes_))
    except Exception as e:
        print(classification_report(le.inverse_transform(y_test), le.inverse_transform(test_pred), target_names=le.classes_))
        print(e)

    print('\nAnalysing weights')

    threshold_intervals = np.arange(0.0, 1, 0.1)

    threshold_preds = []

    for i in tqdm(threshold_intervals):
        threshold_preds.append(np.round(final_prediction(xgb_pred_proba_test, log_pred_proba_test, i, 1-i, threshold=i), 3))

    y_f1 = np.array([np.round(f1_score(le.inverse_transform(y_test), le.inverse_transform(pred), average='micro'), 3) for pred in threshold_preds])
    print('Best F1 score: {} with XGBoost weight: {}'.format(y_f1.max(), threshold_intervals[y_f1.argmax()]))

