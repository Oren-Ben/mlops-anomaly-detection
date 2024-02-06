import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from utils import DropColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer


DATA_PATH = '../data/'

def load_lstm_dfs(path_to_data=DATA_PATH):
# benchmark files checking
    all_files=[]
    for root, dirs, files in os.walk(path_to_data):
        for file in files:
            if file.endswith(".csv") and 'lgbm_baseline_predictions' not in file:
                all_files.append(os.path.join(root, file))
    
    list_of_df = [pd.read_csv(file, 
                          sep=';', 
                          index_col='datetime', 
                          parse_dates=True) for file in all_files if 'anomaly-free' not in file]
    return list_of_df

def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


class TakeFirstNRowsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_rows=400):
        self.n_rows = n_rows
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.iloc[:self.n_rows, :]

class LstmDataPrep:
    def __init__(self, target_column : str = 'anomaly') -> None:
        self.target_column = target_column
        
    def get_features_pipeline(self):
        return Pipeline([
            ('keep_rows', TakeFirstNRowsTransformer(n_rows=400)),
            ('drop_targets',DropColumnTransformer(['changepoint',self.target_column])),
            ('scaler',StandardScaler()),
            ('split_sequences',FunctionTransformer(func=split_sequences, kw_args={'n_steps': 5}))
        ])

    def get_targets_pipeline(self):
        raise NotImplementedError()
    
    def run(self):
       features_pipeline = self.get_features_pipeline()
       return features_pipeline