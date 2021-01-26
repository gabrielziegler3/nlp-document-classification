import pickle
import dask.dataframe as dd
import numpy as np

from joblib import dump, load
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, classification_report, hamming_loss, zero_one_loss
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


DATA_PATH = '/home/isis/Davi_Alves/data/multilabel/dataset_multilabel_10-03-2019.csv'
TFIDF_PATH = './tfidf.pkl'


def get_data(path):
    data = dd.read_csv(path)
    data = data.set_index('Unnamed: 0', sorted=True)
    data = data.rename(columns={'document_id': 'process_id', 'pages': 'page'})
    return data


def split_data(data):
    train_body, test_body, train_labels, test_labels = train_test_split(
        data.body.compute(),
        data.document_type.compute(),
        test_size=0.30,
        random_state=42,
        stratify=data.document_type.compute()
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
            ngram_range=(1, 1),
            max_features=10000,
            use_idf=1,
            smooth_idf=1,
            sublinear_tf=1)),
        ('clf', OneVsRestClassifier(
            XGBClassifier(
                n_jobs=-1,
                max_depth=4,
                learning_rate=0.1,
                n_estimators=100,
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
    data = get_data(DATA_PATH)
    print(data.head())

    data.document_type = data[data.document_type.compute() != 'outros']
    print('Values counts: \n', data.document_type.astype(str).value_counts().compute())
    train_body, test_body, train_labels, test_labels = split_data(data)
    print(train_labels.head())

    CHUNKS = train_body.shape[0]
    #  CHUNKS = 10000

    print('Chunk size: {}'.format(CHUNKS))

    y_train, y_test, le = transform_y(train_labels[:CHUNKS], test_labels[:CHUNKS])
    X_train = train_body
    X_test = test_body
    print('X_train: {}, \n\ty_train: {}'.format(X_train.shape, y_train.shape))
    print('X_test: {}, \n\ty_test: {}'.format(X_test.shape, y_test.shape))

    print('We\'re classifying {} documents!'.format(y_train.shape[0]))
    print('Classes: ', le.classes_)

    print('Started Training!')
    xgb = train_pipeline(X_train, y_train)

    save_model(xgb, 'xgboost.joblib')

    print('Starting predictions on dtest!')

    train_pred = xgb.predict(X_train)
    print('Train score')
    model_report(y_train, train_pred, target_names=[str(x) for x in le.classes_])

    test_pred = xgb.predict(X_test)
    print('Test score')
    model_report(y_test, test_pred, target_names=[str(x) for x in le.classes_])
