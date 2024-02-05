import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import lightgbm as lgb
from lgbm_data_prep import LgbmDataPrep


tf.random.set_seed(0)
np.random.seed(0)
random.seed(0)
os.environ["PYTHONHASHSEED"] = "0"


MODEL_HP={'objective':'binary',
            'metric':'binary_error',
            'force_row_wise':True,
            'seed':0,
            'learning_rate':0.0424127,
            'min_data_in_leaf':15,
            'max_depth':24,
            'num_leaves':29
           }

class LgbmModel:
    def __init__(self,model_hp = None) -> None:
        self.model_hp = model_hp if model_hp else MODEL_HP
        self.trained_model = None
        self.x_train = None
        self.y_train = None
        self.x_valid = None
        self.y_valid = None
        self.x_test = None
        self.y_test = None
        self.y_pred = None
        self.features_pipe = None 
        self.targets_pipe = None

    def train(self,train_set, valid_set=None):
        
        self.features_pipe, self.targets_pipe  =LgbmDataPrep().run()
        self.x_train = self.features_pipe.fit_transform(train_set)
        self.y_train = self.targets_pipe.fit_transform(train_set)
        train_set=lgb.Dataset(self.x_train, self.y_train)
        
        if valid_set is not None:
            self.x_valid = self.features_pipe.transform(valid_set)
            self.y_valid = self.targets_pipe.transform(valid_set)
            valid_set=lgb.Dataset(self.x_valid, self.y_valid)
            
        self.trained_model=lgb.train(
            params=self.model_hp,
            train_set=train_set,
            )
    
    def predict(self,df:pd.DataFrame):
        if self.trained_model:
            self.x_test = self.features_pipe.transform(df)
            self.y_test = self.targets_pipe.transform(df)
            self.y_pred = self.trained_model.predict(self.x_test)
            
            return self.y_pred
        else:
            raise Exception("The model is not trained")