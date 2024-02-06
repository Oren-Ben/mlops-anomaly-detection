import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
# from sklearn.compose import ColumnTransformer
from utils import load_dfs_by_source


DATA_PATH = '/Users/orenben/Documents/runi/mlops-anomaly-detection/data/'


def train_val_test_split_original() ->Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load original lgbm data sets
    
    Returns:
         x_train, x_valid, x_test, y_train, y_valid, y_test 
    """
    #train_pre(valve1_data is dataframe)
    valve1_data_sets = load_dfs_by_source(data_dir=DATA_PATH,data_source='valve1')
    train_set = valve1_data_sets.iloc[:12712]
    valid_set = valve1_data_sets.iloc[12712:-1853]
    test_set = valve1_data_sets.iloc[-1853:]
    return train_set, valid_set, test_set

def data_spliter(df,num_splits):
    num_rows_per_part = len(df) // num_splits

    parts = []
    for i in range(num_splits):
        if i < num_splits-1:
            part = df.iloc[i * num_rows_per_part: (i + 1) * num_rows_per_part]
        else:
            # For the last part, include the remaining rows
            part = df.iloc[i * num_rows_per_part:]
        parts.append(part)
    return parts

def smooth_curve(x):
    #x=1 dimension array
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5] 

def create_dataset(dataset,look_back=10):
    
    data_X=np.zeros((len(dataset)-look_back+1,3))
    j=0
    
    for i in range(look_back-1,len(dataset)):
        
        data_pre=dataset[i-look_back+1:i+1,0]
    
        data_pre_mean=np.mean(data_pre,axis=0)
        data_pre_min=np.min(data_pre,axis=0)
        data_pre_max=np.max(data_pre,axis=0)
        
        data_X[j,:]=np.array([data_pre_mean,data_pre_min,data_pre_max])
        j+=1
    
    return np.array(data_X).reshape(-1,3) 


class TargetsCreator(BaseEstimator, TransformerMixin):
    
    def __init__(self,target_column='anomaly',look_back:int=10) -> None:
        self.look_back = look_back
        self.target_column = target_column

    def fit(self, X ,y=None):
        # The fit method is required by the TransformerMixin
        return self

    def transform(self, X ,y=None):
        # Apply the specified transformation function to the target column
        y = X[self.target_column]
        
        return pd.Series(y[self.look_back-1:]).rename('anomaly').values



class SmoothCurve(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Perform arbitary transformation
        X_win=np.zeros_like(X.values)
        data_dim = X.shape[1]
        
        for i in range(0,data_dim):
            X_win[:,i]=smooth_curve(X.values[:,i].flatten())
    
        return X_win

class CreateStatsDataframe(BaseEstimator, TransformerMixin):
    
    stats_cols=['A1_mean','A1_min','A1_max', \
          'A2_mean','A2_min','A2_max', \
          'Cur_mean','Cur_min','Cur_max', \
          'Pre_mean','Pre_min','Pre_max', \
          'Temp_mean','Temp_min','Temp_max', \
          'Ther_mean','Ther_min','Ther_max', \
          'Vol_mean','Vol_min','Vol_max', \
          'Flow_mean','Flow_min','Flow_max']
    
    
    def __init__(self,look_back:int=10) -> None:
        self.look_back = look_back
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        
        data_dim = X.shape[1]
        for i in range(data_dim):
            if i==0:
                X_win=create_dataset(X[:,i].reshape(-1,1),look_back=self.look_back)
            else:
                X_win =  np.concatenate([X_win,create_dataset(X[:,i].reshape(-1,1),look_back=self.look_back)],axis=-1)
                
        X_win=X_win.reshape(-1,3*data_dim)
        
        df = pd.DataFrame(data=X_win,columns=CreateStatsDataframe.stats_cols)
        
        return df
    
class DropColumnTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns_to_drop:List):
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


class LgbmDataPrep:
    def __init__(self, target_column : str = 'anomaly') -> None:
        self.target_column = target_column
    
    def get_features_pipeline(self):
        return Pipeline(
            [
                ('drop_changepoint',DropColumnTransformer(['changepoint',self.target_column])),
                ('smooth',SmoothCurve()),
                ('scaler',StandardScaler()),
                ('df_creator',CreateStatsDataframe(look_back=10))
                ]
            )
    
    def get_targets_pipeline(self):
        return Pipeline(
            [
                ('targets_creation',TargetsCreator())
                ]
            )

    def run(self):
        features_pipeline = self.get_features_pipeline()
        targets_pipeline = self.get_targets_pipeline()
            
        return features_pipeline, targets_pipeline