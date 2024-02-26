from typing import List
import os
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def get_file_paths(directory=".", extension=".csv"):
    """
    Get all the file paths in the given directory.

    Parameters
    ----------
    directory : str
        The directory to search for files.

    Returns
    -------
    list
        A list of file paths in the given directory.
    """
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file_paths.append(os.path.join(root, file))
    return file_paths


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


class DropColumnTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, columns_to_drop: List):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Check if the DataFrame has the specified columns to drop
        if not set(self.columns_to_drop).issubset(X.columns):
            missing_cols = list(set(self.columns_to_drop) - set(X.columns))
            raise ValueError(f"Columns to drop not found: {missing_cols}")

        # Drop the specified columns
        X_transformed = X.drop(columns=self.columns_to_drop, axis=1)

        return X_transformed


def data_spliter(df, num_splits):
    num_rows_per_part = len(df) // num_splits

    parts = []
    for i in range(num_splits):
        if i < num_splits - 1:
            part = df.iloc[i * num_rows_per_part : (i + 1) * num_rows_per_part]
        else:
            # For the last part, include the remaining rows
            part = df.iloc[i * num_rows_per_part :]
        parts.append(part)
    return parts
