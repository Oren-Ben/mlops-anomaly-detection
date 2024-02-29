import os
import numpy as np
import pandas as pd
from typing import List, Optional
from utils_data import get_absolute_path

DATA_PATH = "../data/"


def load_lstm_dfs(data_dir=DATA_PATH):
    # benchmark files checking
    data_absolute_path = get_absolute_path(data_dir)
    
    all_files = []
    for root, dirs, files in os.walk(data_absolute_path):
        for file in files:
            if file.endswith(".csv") and "baseline_predictions" not in file:
                all_files.append(os.path.join(root, file))

    list_of_df = [
        pd.read_csv(file, sep=";", index_col="datetime", parse_dates=True)
        for file in all_files
        if "anomaly-free" not in file
    ]
    return list_of_df


def load_train_test_lstm(list_of_dfs: Optional[List] = None):
    if not list_of_dfs:
        list_of_dfs = load_lstm_dfs()
    train_set = pd.concat([df[:400] for df in list_of_dfs])
    test_set = pd.concat(list_of_dfs)
    return train_set, test_set


def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)
