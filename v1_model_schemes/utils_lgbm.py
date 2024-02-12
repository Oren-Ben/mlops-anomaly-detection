import pandas as pd 
from typing import Tuple
from utils_data import load_dfs_by_source


DATA_PATH = '../data/'

def lgbm_train_val_test_split_original() -> (
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
):
    """
    Load original lgbm data sets

    Returns:
         x_train, x_valid, x_test, y_train, y_valid, y_test
    """
    # train_pre(valve1_data is dataframe)
    valve1_data_sets = load_dfs_by_source(data_dir=DATA_PATH, data_source="valve1")
    train_set = valve1_data_sets.iloc[:12712]
    valid_set = valve1_data_sets.iloc[12712:-1853]
    test_set = valve1_data_sets.iloc[-1853:]
    return train_set, valid_set, test_set