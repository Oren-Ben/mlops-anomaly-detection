import pandas as pd
from lgbm_full_model import FullLgbmPipeline
from evaluation import ModelSelector
from evaluation import ModelEvaluation

model_config = {
    'target_column': 'anomaly',
    'look_back': 10,
    'num_splits': 2,
    'th' : 0.35
}

selector = ModelSelector(model_config,FullLgbmPipeline,min_splits=1,max_splits=10)
res = selector.select_best(f1_th=0.9)

base_preds = pd.read_csv('../data/lgbm_baseline_predictions.csv')['y_pred']

eval_config = {
    'baseline' : {
        'y_test' :res['model'].y_test,
        'y_pred': base_preds.values,
        'th': 0.5
    },
    
    res['model_name'] : {
        'y_test' :res['model'].y_test,
        'y_pred':res['predictions'].values,
        'th' : 0.35
    }
}

eval = ModelEvaluation(eval_config=eval_config)
eval.plot_metrics()