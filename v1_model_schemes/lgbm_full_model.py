from utils_data import x_y_split , data_splitter
from utils_lgbm import lgbm_train_val_test_split_original
from lgbm_data_prep import LgbmDataPrep
from sklearn.pipeline import Pipeline, FeatureUnion
import numpy as np
import pandas as pd
from typing import List


from typing import Dict, Any

class FullLgbmPipeline:
    
    """
    model_config = {
        'target_column': 'anomaly',
        'look_back': 10,
        'num_splits': 5,
    }
    """
    
    def __init__(self,config:Dict[str,Any]) -> None:
        self.config = config
        self.y_test = None
        
    def run(self)-> pd.Series:
        
        # load data + data prep
        train_set, validation_set, test_set = lgbm_train_val_test_split_original()
        X, y = x_y_split(train_set, self.config['target_column'])
        X_test, y_test = x_y_split(test_set, self.config['target_column'])
        model_pipeline, targets_pipe =  LgbmDataPrep(** self.config).run()
        
        self.y_test = targets_pipe.transform(y_test)
        y = targets_pipe.fit_transform(y)
        
        model_pipeline.fit(X,y)
        
        preds = model_pipeline.transform(X_test)
        y_pred = preds['avg_prediction'].copy()
        
        return y_pred
    
    def get_y_test(self):
        return self.y_test
        
