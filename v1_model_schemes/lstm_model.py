from utils_data import data_splitter
from Vanilla_LSTM import Vanilla_LSTM
import pandas as pd
from typing import List, Optional
from utils_lstm import split_sequences
from sklearn.preprocessing import StandardScaler



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
        model_hp: Optional[List] = None,
        num_splits: int = 5,
        sequences_length : int = 5
            ) -> None:
        
        self.model_hp =model_hp if model_hp else MODEL_HP
        self.num_splits = num_splits
        self.sequences_length = sequences_length
        self.partition_num = partition_num
        self.ucl = None
        self.trained_model = None
        self.prediction = None
        self.test_residuals = None
        self.scaler = None
        
    def fit(self,X,y=None)->None:
        X= X.values
        X_parts = data_splitter(X, num_splits=self.num_splits)[self.partition_num]
        self.scaler = StandardScaler()
        X_parts = self.scaler.fit_transform(X_parts)
        
        x,y = split_sequences(X_parts,n_steps=self.sequences_length)
        orig_model = Vanilla_LSTM(self.model_hp)
        orig_model.fit(x,y)
        self.trained_model = orig_model
        residuals = pd.DataFrame(y - self.trained_model.predict(x)).abs().sum(axis=1)
        self.ucl = residuals.quantile(Q) * 5
        return self
    
    def transform(self,X)->pd.Series:
        X = X.values
        if self.trained_model:
            X =  self.scaler.transform(X)
            x,y = split_sequences(X,n_steps=self.sequences_length)
            self.test_residuals = pd.DataFrame(y - self.trained_model.predict(x)).abs().sum(axis=1)
            self.prediction = pd.Series((self.test_residuals > self.ucl).astype(int).values).fillna(0)
            return self.prediction.values.reshape(-1,1)
        else:
            raise Exception("The model is not trained")