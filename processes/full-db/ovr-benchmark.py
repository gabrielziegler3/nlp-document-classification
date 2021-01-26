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


DATA_PATH = '../../data/grouped-02-01-2019.csv'
TFIDF_PATH = './tfidf.pkl'
THEMES = [5, 6, 33, 163, 232, 313, 339, 350, 406, 409, 555, 589,
          597, 634, 660, 695, 766, 773, 793, 800, 810, 852, 895, 951]
OTHERS_INCLUDED = [139, 975, 729, 26]
THEMES = THEMES + OTHERS_INCLUDED
print('THEMES: ', THEMES)


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
    data = data.rename(columns={'document_id': 'process_id', 'pages': 'page'})
    return data


def split_data(data):
    train_body, test_body, train_labels, test_labels = train_test_split(
        data.body.compute(),
        data.themes.compute(),
        test_size=0.33,
        random_state=42,
        # stratify=data.themes.compute()
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
    mlb = MultiLabelBinarizer()
    mlb.fit(train_labels)

    mlb_train = mlb.transform(train_labels)
    mlb_test = mlb.transform(test_labels)

    print(mlb.classes_)

    return mlb_train, mlb_test, mlb


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


def hamming_loss_non_zero(Y_test, np_pred, debug=False):
    max_hamming = 0
    for i in range(0, len(Y_test)):
        temp_hamming = 0
        max_pred = 0
        error_pred = 0
        for j in range(len(Y_test[i])):
            if Y_test[i][j] == np_pred[i][j] == 0:
                pass

            if Y_test[i][j] != 0:
                max_pred += 1

            if Y_test[i][j] != np_pred[i][j]:
                error_pred += 1
        if max_pred != 0:
            temp_hamming = float(error_pred)/float(max_pred)
            if temp_hamming > 1.0:
                temp_hamming = 1.0
            if debug:
                print("MAX: {}  ERROR: {} HAMMING: {}".format(max_pred, error_pred, temp_hamming))
            max_hamming += temp_hamming
    return float(max_hamming)/float(len(Y_test))


def model_report(y_true, y_pred, target_names=None):
    """
    Both y_true and y_pred must be already transformed to MultiLabelBinarizer
    """
    print("Hamming Loss: {}".format(hamming_loss(y_true, y_pred)))
    print("Zero One Loss: {}".format(zero_one_loss(y_true, y_pred)))
    print("Hamming Loss Non Zero: {}\n".format(hamming_loss_non_zero(y_true, np.array(y_pred))))
    print(classification_report(y_true, y_pred, target_names=target_names))


if __name__ == '__main__':
    data = get_data(DATA_PATH)
    print(data.head())

    # data.themes = data.themes.compute()
    data.themes = data.themes.compute().map(eval).apply(
            lambda x: list(set(sorted([i if i in THEMES else 0 for i in x])))
            )

    print('Values counts: \n', data.themes.astype(str).value_counts().compute())
    train_body, test_body, train_labels, test_labels = split_data(data)
    print(train_labels.head())

    CHUNKS = train_body.shape[0]
    #  CHUNKS = 10000

    print('Chunk size: {}'.format(CHUNKS))

    y_train, y_test, mlb = transform_y(train_labels[:CHUNKS], test_labels[:CHUNKS])
    X_train = train_body
    X_test = test_body
    print('X_train: {}, \n\ty_train: {}'.format(X_train.shape, y_train.shape))
    print('X_test: {}, \n\ty_test: {}'.format(X_test.shape, y_test.shape))

    print('We\'re classifying {} themes!'.format(y_train.shape[1]))
    print('Classes: ', mlb.classes_)

    print('Started Training!')
    xgb = train_pipeline(X_train, y_train)

    save_model(xgb, 'xgboost.joblib')

    print('Starting predictions on dtest!')

    train_pred = xgb.predict(X_train)
    print('Train score')
    model_report(y_train, train_pred, target_names=[str(x) for x in mlb.classes_])

    test_pred = xgb.predict(X_test)
    print('Test score')
    model_report(y_test, test_pred, target_names=[str(x) for x in mlb.classes_])

