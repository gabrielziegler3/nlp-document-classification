import os
import sys
import pickle
import pandas as pd
import dask.dataframe as dd
import numpy as np

from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


DATA_PATH = '../../data/02-01-2019.csv'
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
    return data


if __name__ == '__main__':
    data = get_data(DATA_PATH)
    data = groupby_process(data)
    print(data.head())

    print('Values counts: \n', data.themes.value_counts())
    f = '../../data/grouped-02-01-2019.csv'
    print('Saving data at ', f)
    data.to_csv(f)
