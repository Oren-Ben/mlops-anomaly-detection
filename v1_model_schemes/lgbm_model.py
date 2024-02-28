from typing import Dict, Any
from utils_data import data_splitter
import lightgbm as lgb
import pandas as pd


MODEL_HP = {
    "objective": "binary",
    "metric": "binary_error",
    "force_row_wise": True,
    "seed": 0,
    "learning_rate": 0.0424127,
    "min_data_in_leaf": 15,
    "max_depth": 24,
    "num_leaves": 29,
}

class LgbmModel:
    def __init__(
        self,
        partition_num : int,
        model_hp:Dict[str,Any] = None,
        num_splits: int = 5
        ) -> None:
        
        self.model_hp = model_hp if model_hp else MODEL_HP
        self.num_splits = num_splits
        self.partition_num = partition_num 
        self.x_train = None
        self.y_train = None
        self.train_set = None
        self.test_set = None
        self.trained_model = None
        self.y_pred = None
        
    def fit(self, X,y)->None:
        
        x_parts = data_splitter(X, num_splits=self.num_splits)
        y_parts = data_splitter(y, num_splits=self.num_splits)
        self.x_train = x_parts[self.partition_num]
        self.y_train = y_parts[self.partition_num]
        
        train_set = lgb.Dataset(self.x_train,self.y_train)
        self.trained_model = lgb.train(
            params=self.model_hp,
            train_set=train_set,
        )
        return self
        
    def transform(self,X):
        if self.trained_model:
            self.y_pred = self.trained_model.predict(X)
            return self.y_pred.reshape(-1,1)
        else:
            raise Exception("The model is not trained")