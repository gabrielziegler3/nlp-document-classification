import pickle
import dask.dataframe as dd
import pandas as pd
import numpy as np

from joblib import dump, load
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, classification_report, hamming_loss, zero_one_loss
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

TRAIN_PATH = '/home/isis/Davi_Alves/data/parts/train_pecas_06-02-2019_clean.csv'
VAL_PATH = '/home/isis/Davi_Alves/data/parts/validation_pecas_06-02-2019_clean.csv'
TEST_PATH = '/home/isis/Davi_Alves/data/parts/test_pecas_06-02-2019_clean.csv'
TFIDF_PATH = './tfidf.pkl'


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
    return data


def split_data(data):
    train_body, test_body, train_labels, test_labels = train_test_split(
        data.page_corpus_processed,
        data.Piece,
        test_size=0.33,
        random_state=42,
        # stratify=data.document_type
    )

    return train_body, test_body, train_labels, test_labels


def tfidf(train_body, test_body):
    tfidf = TfidfVectorizer(
        min_df=0.2,
        max_df=0.9,
        lowercase=False,
        use_idf=1,
        smooth_idf=1,
        # sublinear_tf=1,
        max_features=10000,
        # ngram_range=(1, 2)
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


def train_pipeline(X_train, y_train):
    xgb = Pipeline([
        ('tfidf', TfidfVectorizer(
            # vocabulary=vocab,
            min_df=0.1,
            ngram_range=(1, 2),
            max_features=10000,
            use_idf=1,
            smooth_idf=1,
            sublinear_tf=1)),
        ('clf', OneVsRestClassifier(
            XGBClassifier(
                n_jobs=-1,
                max_depth=4,
                learning_rate=0.1,
                n_estimators=1000,
                # eval_metric='mlogloss',
                # tree_method='gpu_hist',
                # gpu_id=0,
                # num_class=y_train.shape[1],
                ),
            n_jobs=-1))
    ])

    xgb.fit(X_train, y_train)
    print("Finished training")

    print('TFIDF: ', xgb.named_steps['tfidf'])
    print('Classifier: ', xgb.named_steps['clf'])

    return xgb


def save_model(model, path):
    print("Saving model at {} ...".format(path))
    dump(xgb, path)


def model_report(y_true, y_pred, target_names=None):
    """
    Both y_true and y_pred must be already transformed to MultiLabelBinarizer
    """
    print("Hamming Loss: {}".format(hamming_loss(y_true, y_pred)))
    print("Zero One Loss: {}".format(zero_one_loss(y_true, y_pred)))
    print(classification_report(y_true, y_pred, target_names=target_names))
    try:
        print(confusion_matrix(y_true, y_pred, labels=target_names))
    except Exception as e:
        print(e)


if __name__ == '__main__':
    train = get_data(TRAIN_PATH)
    val = get_data(VAL_PATH)
    test = get_data(TEST_PATH)

    # train = pd.concat([train, val])
    print(train.head())
    print(train.columns)

    print('Values counts: \n', train.Piece.astype(str).value_counts())

    CHUNKS = train.shape[0]
    # CHUNKS = 1000

    print('Chunk size: {}'.format(CHUNKS))

    y_train, y_test, le = transform_y(train.Piece[:CHUNKS], test.Piece[:CHUNKS])

    X_train = train.page_corpus_processed
    X_test = test.page_corpus_processed

    print('X_train: {}, \n\ty_train: {}'.format(X_train.shape, y_train.shape))
    print('X_test: {}, \n\ty_test: {}'.format(X_test.shape, y_test.shape))

    print('We\'re classifying {} documents!'.format(y_train.shape[0]))
    print('Classes: ', le.classes_)

    print('Started Training!')
    xgb = train_pipeline(X_train, y_train)

    save_model(xgb, 'xgboost.joblib')

    print('Starting predictions on dtest!')

    train_pred = xgb.predict(X_train)
    print('train pred:', train_pred[:10])
    print('Train score')

    y_train = le.inverse_transform(y_train)
    train_pred = le.inverse_transform(train_pred)
    model_report(y_train, train_pred, target_names=[str(x) for x in le.classes_])

    test_pred = xgb.predict(X_test)
    print('Test score')
    y_test = le.inverse_transform(y_test)
    test_pred = le.inverse_transform(test_pred)
    model_report(y_test, test_pred, target_names=[str(x) for x in le.classes_])

