import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from transformers_inventory import DropColumnTransformer
from sklearn.preprocessing import StandardScaler
from transformers_inventory import SequenceSplitterTransformer, AggregateModelScores
from lstm_model import LstmModel
import copy
from utils_lstm import load_train_test_lstm
from utils_data import data_splitter


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


class LstmDataPrep:
    def __init__(self, target_column: str = "anomaly", n_steps:int =5, num_splits: int = 5) -> None:
        self.target_column = target_column
        self.n_steps = n_steps
        self.num_splits = num_splits
        self.y_test = self.create_y_test()
        

    def get_preprocess_pipeline(self):
        return Pipeline(
            [   
             ( "drop_targets",DropColumnTransformer(["changepoint", self.target_column]),),
             ("scaler", StandardScaler()),
             ("split_sequences",SequenceSplitterTransformer(n_steps=self.n_steps))
             ])
        
    def get_features_pipeline(self):
        
        single_features_pipeline =Pipeline([
        ])
        transformers_list = []
        
        for i in range(self.num_splits):
            y_test_parts = data_splitter(self.y_test, num_splits=self.num_splits)
            curr_pipe = copy.deepcopy(single_features_pipeline)
            predict_step = ('classifier',
                            LstmModel(
                                partition_num = i ,
                                num_splits = self.num_splits,
                                )
                            )
            curr_pipe.steps.append(predict_step)
            transformers_list.append((f'part_{i}',curr_pipe))        
        
        union_pipeline =FeatureUnion(transformer_list=transformers_list)
        
        features_pipeline = Pipeline([
            ('transformations', union_pipeline),
            ('processing_results',AggregateModelScores())
        ])
        
        return features_pipeline

    def get_targets_pipeline(self):
        raise NotImplementedError()

    def run(self):
        preprocess_pipeline = self.get_preprocess_pipeline()
        
        features_pipeline = self.get_features_pipeline()
        return preprocess_pipeline, features_pipeline
        
    def create_y_test(self):
        train_set, test_set = load_train_test_lstm(list_of_dfs=load_lstm_dfs())
        preprocess_pipe = self.get_preprocess_pipeline()
        preprocess_pipe.fit(train_set)
        x_test, y_test = preprocess_pipe.transform(test_set)
        return y_test



    def model_pipeline(self):
        
        single_model_pipeline = Pipeline(
                [   
                ( "drop_targets",DropColumnTransformer(["changepoint", self.target_column]),),
                ("scaler", StandardScaler()),
                ("split_sequences",SequenceSplitterTransformer(n_steps=self.n_steps))
                ])
        
        transformers_list = []
        for i in range(self.num_splits):
            curr_pipe = copy.deepcopy(single_model_pipeline)
            predict_step = ('classifier',LstmModel(partition_num=i ,num_splits=self.num_splits))
            curr_pipe.steps.append(predict_step)
            transformers_list.append((f'part_{i}',curr_pipe))        
                
            union_pipeline =FeatureUnion(transformer_list=transformers_list)
                
            features_pipeline = Pipeline([
                ('transformations', union_pipeline),   
                ('processing_results',AggregateModelScores())
                ])
                
                
        return features_pipeline