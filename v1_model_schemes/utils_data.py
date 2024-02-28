import os
import numpy as np
import pandas as pd
from typing import List, Union, Tuple
from abc import ABC, abstractmethod


def load_dfs_by_source(data_dir: str, data_source: str) -> pd.DataFrame:
    """
    Load all dfs by source ('valve1','valve2','other')

    Args:
        data_dir (str): _description_
        data_source (str): _description_

    Returns:
        pd.DataFrame: _description_
    """
    all_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".csv"):
                all_files.append(os.path.join(root, file))
    all_files.sort()

    dfs_list = [
        pd.read_csv(file, sep=";", index_col="datetime", parse_dates=True)
        for file in all_files
        if data_source in file
    ]

    return (
        pd.concat(dfs_list)
        # .drop(columns=['changepoint'])
        .sort_index()
    )


def x_y_split(df: pd.DataFrame, label_column: str) -> Tuple[pd.DataFrame, np.array]:
    """Splits DataFrame to features df  and label column as np.array

    Args:
        df : dataset
        label_column: the name of the label column in the dataset.

    Returns:
        Tuple: (X, y)
    """
    X = df.drop(label_column, axis=1)
    y = df[label_column].values
    return X, y


def smooth_curve(x):
    # x=1 dimension array
    window_len = 11
    s = np.r_[x[window_len - 1 : 0 : -1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w / w.sum(), s, mode="valid")
    return y[5 : len(y) - 5]


def create_dataset(dataset, look_back=10):

    data_X = np.zeros((len(dataset) - look_back + 1, 3))
    j = 0

    for i in range(look_back - 1, len(dataset)):

        data_pre = dataset[i - look_back + 1 : i + 1, 0]

        data_pre_mean = np.mean(data_pre, axis=0)
        data_pre_min = np.min(data_pre, axis=0)
        data_pre_max = np.max(data_pre, axis=0)

        data_X[j, :] = np.array([data_pre_mean, data_pre_min, data_pre_max])
        j += 1

    return np.array(data_X).reshape(-1, 3)


def data_splitter(
    X: Union[pd.DataFrame, np.ndarray], num_splits: int
) -> List[pd.DataFrame]:
    """
    Splits a dataframe or ndarray to multiple splits determined by num_splits argument

    Args:
        X (Union[pd.DataFrame, np.ndarray]): The data set to split by the index (not shuffled)
        num_splits (int): the number splits to the data.

    Returns:
        List[pd.DataFrame]: list that contains the dataframe splits
    """
    if isinstance(X, pd.DataFrame):
        num_rows = len(X)
    elif isinstance(X, np.ndarray):
        num_rows = X.shape[0]
    else:
        raise ValueError("Input data must be a DataFrame or ndarray.")

    if num_splits == 0 or num_splits == 1:
        return [X]

    num_rows_per_part = num_rows // num_splits

    parts = []
    for i in range(num_splits):
        start_idx = i * num_rows_per_part
        end_idx = (i + 1) * num_rows_per_part if i < num_splits - 1 else None
        part = X[start_idx:end_idx]
        parts.append(part)

    return parts


class AbstractFullModelPipeline(ABC):
    """
    The parent of FullLstmPipeline, FullLgbmPipeline
    """

    @abstractmethod
    def run(self) -> pd.Series:
        pass

    @abstractmethod
    def get_y_test(self):
        pass
