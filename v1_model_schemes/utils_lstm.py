import os
import pandas as pd
from typing import List, Optional


DATA_PATH = "../data/"


def load_lstm_dfs(path_to_data=DATA_PATH):
    # benchmark files checking
    all_files = []
    for root, dirs, files in os.walk(path_to_data):
        for file in files:
            if file.endswith(".csv") and "lgbm_baseline_predictions" not in file:
                all_files.append(os.path.join(root, file))

    list_of_df = [
        pd.read_csv(file, sep=";", index_col="datetime", parse_dates=True)
        for file in all_files
        if "anomaly-free" not in file
    ]
    return list_of_df

def load_train_test_lstm(list_of_dfs: Optional[List]= None):
    if not list_of_dfs:
        list_of_dfs = load_lstm_dfs()
    train_set = pd.concat([df[:400] for df in list_of_dfs])
    test_set = pd.concat(list_of_dfs)
    return train_set, test_set