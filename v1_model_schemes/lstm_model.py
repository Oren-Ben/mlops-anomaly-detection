from utils_data import data_splitter
from Vanilla_LSTM import Vanilla_LSTM
import pandas as pd
from typing import List, Optional


N_STEPS = 5
EPOCHS = 25
BATCH_SIZE = 32
VAL_SPLIT = 0.2
MODEL_HP = [N_STEPS, EPOCHS, BATCH_SIZE, VAL_SPLIT]
Q = 0.99  # quantile for upper control limit (UCL) selection



class LstmModel:
    def __init__(
        self,
        partition_num : int,
        # y_test,
        model_hp: Optional[List] = None,
        num_splits: int = 5,
        ) -> None:
        self.model_hp =model_hp if model_hp else MODEL_HP
        self.num_splits = num_splits
        self.partition_num = partition_num 
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self._y_pred = None
        self.y_pred = None
        # self.y_test = y_test
        self.y = None
        self.ucl = None
        self.trained_model = None
        self.residuals_train = None
        
    def fit(self,X,y=None)-> None:
        x,y = X
        x_parts = data_splitter(x, num_splits=self.num_splits)
        y_parts = data_splitter(y, num_splits=self.num_splits)
        self.x_train = x_parts[self.partition_num]
        self.y_train = y_parts[self.partition_num]
        
        model = Vanilla_LSTM(self.model_hp)
        model.fit(self.x_train, self.y_train)
        self.trained_model = model
        self.residuals_train = (
            pd.DataFrame(self.y_train - self.trained_model.predict(self.x_train)).abs().sum(axis=1))
        self.ucl = self.residuals_train.quantile(Q) * 5
        
        print("x",  self.x_train.shape)
        print("y", self.y_train.shape)
        
        
        
        return self
    
    def transform(self,X)->pd.Series:
        if self.trained_model:
            x,y = X    
            self._y_pred = self.trained_model.predict(x)
            # print("self._y_pred: ", self._y_pred)
            residuals = pd.DataFrame(y - self._y_pred).abs().sum(axis=1)
            self.y_pred = pd.Series((residuals > self.ucl).astype(int).values).fillna(0).values.reshape(-1,1)
            return self.y_pred
        else:
            raise Exception("The model is not trained")
        
        #     # self.x_test, self.y_test  
        #     self._y_pred = self.trained_model.predict(X)
        #     print("X: ")
        #     print(X.shape)
        #     print("_y_pred:")
        #     print(self._y_pred.shape)
        #     print("self.y_test")
        #     print(self.y_test.shape)
        #     if self._y_pred.shape[0] == self.y_test.shape[0]:
        #         residuals = pd.DataFrame(self.y_test - self._y_pred).abs().sum(axis=1)
        #     else:
        #         residuals = pd.DataFrame(self.y - self._y_pred).abs().sum(axis=1)
        #     self.y_pred = pd.Series((residuals > self.ucl).astype(int).values).fillna(0).values.reshape(-1,1)
        #     return self.y_pred
        # else:
        #     raise Exception("The model is not trained")