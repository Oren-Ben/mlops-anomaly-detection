import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from transformers_inventory import DropColumnTransformer
from transformers_inventory import AggregateModelScores
from lstm_model import LstmModel
import copy
from utils_lstm import load_train_test_lstm, load_lstm_dfs
from utils_data import data_splitter

DATA_PATH = "../data/"
class LstmDataPrep:
    def __init__(self, target_column: str = "anomaly", n_steps:int =5, num_splits: int = 4) -> None:
        self.target_column = target_column
        self.n_steps = n_steps
        self.num_splits = num_splits
        

    def get_preprocess_pipeline(self):
        return Pipeline([( "drop_targets",DropColumnTransformer(["changepoint", self.target_column]),),])

    def model_pipeline(self):
        
        single_model_pipeline = self.get_preprocess_pipeline()
        
        transformers_list = []
        if self.num_splits == 0:
            curr_pipe = copy.deepcopy(single_model_pipeline)
            predict_step = ('classifier',LstmModel(partition_num=0, num_splits=self.num_splits))
            curr_pipe.steps.append(predict_step)
            transformers_list.append(('part_0',curr_pipe))
                
            union_pipeline =FeatureUnion(transformer_list=transformers_list)
            features_pipeline = Pipeline([
                ('transformations', union_pipeline),   
                ('processing_results',AggregateModelScores())
                ])
            return features_pipeline
            
        for i in range(self.num_splits):
            curr_pipe = copy.deepcopy(single_model_pipeline)
            predict_step = ('classifier',LstmModel(partition_num=i,num_splits=self.num_splits))
            curr_pipe.steps.append(predict_step)
            transformers_list.append((f'part_{i}',curr_pipe)) 
                
            union_pipeline =FeatureUnion(transformer_list=transformers_list)
                
            features_pipeline = Pipeline([
                ('transformations', union_pipeline),   
                ('processing_results',AggregateModelScores())
                ])
                
        return features_pipeline