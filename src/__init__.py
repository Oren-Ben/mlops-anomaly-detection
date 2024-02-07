import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from .lgbm_multi import MultiModelLGBM
from .util import get_file_paths
from .calc_metrics import calc_metrics
from .plot import plot

dirname = os.path.dirname(__file__)

valve1_paths = get_file_paths("../data/valve1")

dfs = []

for valve1_path in valve1_paths:
    df = pd.read_csv(valve1_path, sep=";", index_col="datetime", parse_dates=True)
    dfs.append(df)

# Concatenate all the DataFrames in the list into a single DataFrame
valve1_df = pd.concat(dfs)

# Split Valve data
total_rows = len(valve1_df)
index_70_percent = int(total_rows * 0.7)
index_90_percent = index_70_percent + int(total_rows * 0.2)

train_df = valve1_df.iloc[:index_70_percent]
valid_df = valve1_df.iloc[index_70_percent:index_90_percent]
test_df = valve1_df.iloc[index_90_percent:]

X_train = train_df.drop("anomaly", axis=1)
y_train = train_df["anomaly"]

# TODO: valid_df?

X_test = test_df.drop("anomaly", axis=1)
y_test = test_df["anomaly"]


# =====the searching area of hyper parameter =====
lr = 10 ** np.random.uniform(-3, 0)
min_data_in_leaf = np.random.choice(range(1, 21), 1)[0]
max_depth = np.random.choice(range(3, 31), 1)[0]
num_leaves = np.random.choice(range(20, 41), 1)[0]
# ================================================
lgb_params = {
    "objective": "binary",
    "metric": "binary_error",
    "force_row_wise": True,
    "seed": 0,
    "learning_rate": lr,
    "min_data_in_leaf": min_data_in_leaf,
    "max_depth": max_depth,
    "num_leaves": num_leaves,
}

# Train the Model
lgbm_model = lgb.LGBMClassifier()
lgbm_model.fit(X_train, y_train)
prob_not_anomaly = lgbm_model.predict_proba(X_test)
lgbm_metrics = calc_metrics(y_test, prob_not_anomaly[:, 1])
# Train the Model list
lgbm_multi_model = MultiModelLGBM()
lgbm_multi_model.fit(X_train, y_train)
mean, max, min = lgbm_multi_model.predict_proba(X_test)

lgbm_multi_model_mean = calc_metrics(y_test, mean[:, 1])
lgbm_multi_model_max = calc_metrics(y_test, max[:, 1])
lgbm_multi_model_min = calc_metrics(y_test, min[:, 1])
# print('lgbm_metrics', lgbm_metrics)
# print('lgbm_multi_model_mean', lgbm_multi_model_mean)
# print('lgbm_multi_model_max', lgbm_multi_model_max)
# print('lgbm_multi_model_min', lgbm_multi_model_min)

# plot(lgbm_metrics, lgbm_multi_model_mean, title = 'Average Probability')
# plot(lgbm_metrics, lgbm_multi_model_max, title = 'Max Probability')
# plot(lgbm_metrics, lgbm_multi_model_min, title = 'Min Probability')
# # TODO: Save your model
# # save_path = os.path.join(dirname, '../model/')
# # lgbm_file = 'lgbm.txt'
# # lgbm_model.booster_.save_model(f'{save_path}{lgbm_file}')


# # lgbm_multi_file = 'lgbm_multi'
# # lgbm_multi_model.savel(f'{save_path}{lgbm_multi_file}')