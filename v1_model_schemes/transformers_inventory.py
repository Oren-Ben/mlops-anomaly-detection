import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.base import BaseEstimator, TransformerMixin
from utils_data import smooth_curve, create_dataset


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


class LgbmTargetsCreator(BaseEstimator, TransformerMixin):
    def __init__(self, look_back: int = 10) -> None:
        self.look_back = look_back

    def fit(self, X, y=None):
        # The fit method is required by the TransformerMixin
        return self

    def transform(self, X):

        return X[self.look_back - 1 :]


class SmoothCurve(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Perform arbitary transformation
        X_win = np.zeros_like(X.values)
        data_dim = X.shape[1]

        for i in range(0, data_dim):
            X_win[:, i] = smooth_curve(X.values[:, i].flatten())

        return X_win


class CreateStatsDataframe(BaseEstimator, TransformerMixin):

    stats_cols = [
        "A1_mean",
        "A1_min",
        "A1_max",
        "A2_mean",
        "A2_min",
        "A2_max",
        "Cur_mean",
        "Cur_min",
        "Cur_max",
        "Pre_mean",
        "Pre_min",
        "Pre_max",
        "Temp_mean",
        "Temp_min",
        "Temp_max",
        "Ther_mean",
        "Ther_min",
        "Ther_max",
        "Vol_mean",
        "Vol_min",
        "Vol_max",
        "Flow_mean",
        "Flow_min",
        "Flow_max",
    ]

    def __init__(self, look_back: int = 10) -> None:
        self.look_back = look_back

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        data_dim = X.shape[1]
        for i in range(data_dim):
            if i == 0:
                X_win = create_dataset(X[:, i].reshape(-1, 1), look_back=self.look_back)
            else:
                X_win = np.concatenate(
                    [
                        X_win,
                        create_dataset(
                            X[:, i].reshape(-1, 1), look_back=self.look_back
                        ),
                    ],
                    axis=-1,
                )

        X_win = X_win.reshape(-1, 3 * data_dim)

        df = pd.DataFrame(data=X_win, columns=CreateStatsDataframe.stats_cols)

        return df


class DataSplitter(BaseEstimator, TransformerMixin):
    def __init__(self, num_splits: int, part_num):
        self.num_splits = num_splits
        self.part_num = part_num

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Splits a dataframe to multiple splits determined by num_splits argument

        Args:
            X (pd.DataFrame): The data set to split by the index (not shuffled)

        Returns:
            List[pd.DataFrame]: list that contains the dataframe splits
        """
        num_rows_per_part = len(X) // self.num_splits
        parts = []
        for i in range(self.num_splits):
            if i < self.num_splits - 1:
                part = X.iloc[i * num_rows_per_part : (i + 1) * num_rows_per_part]
            else:
                # For the last part, include the remaining rows
                part = X.iloc[i * num_rows_per_part :]
            parts.append(part)
        if self.part_num is not None:
            print(self.part_num)
            parts = parts[self.part_num]
        return parts


class AggregateModelScores(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # No fitting needed
        return self

    def transform(self, X):
        # Get the number of columns in the matrix
        print("X.shape: ", X.shape)
        num_columns = X.shape[1]

        # Create column names as running numbers (e.g., 0, 1, 2, ...)
        column_names = [f"model_{i}" for i in range(num_columns)]

        # Create a DataFrame from the matrix with running number column names
        df = pd.DataFrame(X, columns=column_names).assign(
            avg_prediction=lambda df_: df_.mean(axis=1),
            median_prediction=lambda df_: df_.median(axis=1),
            max_prediction=lambda df_: df_.max(axis=1),
        )

        return df


class SequenceSplitterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_steps):
        self.n_steps = n_steps

    def fit(self, X, y=None):
        # No fitting needed
        return self

    def transform(self, sequences):

        if isinstance(sequences, pd.DataFrame):
            sequences = sequences.values

        X, y = list(), list()
        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + self.n_steps
            # check if we are beyond the dataset
            if end_ix > len(sequences) - 1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)
