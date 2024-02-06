import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from lstm_data_prep import LstmDataPrep
from utils import data_spliter
from orig_lstm import Vanilla_LSTM

N_STEPS = 5
EPOCHS = 25
BATCH_SIZE = 32
VAL_SPLIT = 0.2
PARAMS = [N_STEPS, EPOCHS, BATCH_SIZE, VAL_SPLIT]
Q = 0.99 # quantile for upper control limit (UCL) selection

class LstmModel:
    """
    Example usage:
    model = LstmModel()
    model.train(df)
    preds = model.predict(df)
    
    """
    
    def __init__(self,params= None, num_models=5) -> None:
        self.params = params if params else PARAMS
        self.num_models = num_models
        self.partitioned_dfs = None
        self.x_train = None
        self.y_train = None
        self.x_test= None 
        self.y_test = None
        self._y_pred = None
        self.y_pred = None
        self.features_pipe = None
        self.residuals_train = None
        self.trained_model =None
        self.ucl = None
    
    def train(self,train_set):
        self.features_pipe   =LstmDataPrep().run()
        self.x_train, self.y_train = self.features_pipe.fit_transform(train_set)
        model =  Vanilla_LSTM(self.params)
        model.fit(self.x_train,self.y_train)
        self.trained_model = model
        self.residuals_train = pd.DataFrame(self.y_train - self.trained_model.predict(self.x_train)).abs().sum(axis=1)
        self.ucl = self.residuals_train.quantile(Q) * 5

    
    
    def predict(self,df:pd.DataFrame):
        self.x_test, self.y_test = self.features_pipe[1:].transform(df)
        self._y_pred = self.trained_model.predict(self.x_test)
        residuals = pd.DataFrame(self.y_test - self._y_pred).abs().sum(axis=1)
        self.y_pred = pd.Series((residuals > self.ucl).astype(int).values).fillna(0)
        return self.y_pred
    
class MultiModel:
        
    def __init__(self,params,num_models:int=5) -> None:
        self.params = params if params else PARAMS
        self.num_models = num_models
        self.train_set = None
        self.models = {}
        self.partitioned_dfs = None
        self.predictions_df = None
        self.y_test = None
        
    def train(self,train_set):
    
        self.partitioned_dfs = data_spliter(df=train_set,num_splits=self.num_models)

        for i, df in enumerate(self.partitioned_dfs):
            self.models[f'lstm_{i}']= {}
            self.models[f'lstm_{i}']['trained_model'] = Vanilla_LSTM(self.params)
            self.models[f'lstm_{i}']['trained_model'].train(df)
            
    def predict(self,df:pd.DataFrame)->pd.DataFrame:
    
        features_pipe= LstmDataPrep().run()
        _ , self.y_test = features_pipe[1:].transform(df)
    
        preds_dfs = []
        for model, _ in self.models.items():
            self.models[model]['predictions'] = pd.Series(self.models[model]['trained_model'].predict(df))
            preds_df = pd.DataFrame({model:self.models[model]['predictions']})
            preds_dfs.append(preds_df)
        
        self.predictions_df =  (
            pd.concat(preds_dfs,axis=1)
            .assign(
                avg_predicion = lambda df_ : df_.mean(axis=1),
                median_predicion = lambda df_ : df_.median(axis=1),
                max_predicion = lambda df_ : df_.max(axis=1),
                )
            )
        return self.predictions_df