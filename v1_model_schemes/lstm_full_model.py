import pandas as pd
from utils_lstm import load_lstm_dfs, load_train_test_lstm
from lstm_data_prep import LstmDataPrep
from typing import Dict, Any

class FullLstmPipeline:
    """
    model_config = {
        'target_column': 'anomaly',
        'n_steps' : 5,
        'num_splits' : 34,
    }
    
    """
    
    def __init__(self,config:Dict[str,Any]) -> None:
        self.config = config
        self.y_test = None
        
    def run(self)-> pd.Series:
        
        # load data + data prep
        list_of_dfs=load_lstm_dfs()
        self.y_test = pd.concat([df[self.config['target_column']][5:] for df in list_of_dfs],ignore_index=True)
        train_set, test_set = load_train_test_lstm(list_of_dfs=list_of_dfs)
        
        # fit the model
        model_pipe = LstmDataPrep(**self.config).run()
        model_pipe.fit(train_set)
        #get the predictions
        raw_preds = [model_pipe.transform(df) for df in  list_of_dfs]
        preds = pd.concat(raw_preds, ignore_index=True)
        y_pred = preds['avg_prediction'].copy()
        
        return y_pred

    def get_y_test(self):
        return self.y_test