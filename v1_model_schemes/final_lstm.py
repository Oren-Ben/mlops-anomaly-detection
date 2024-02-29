import pandas as pd
from lstm_full_model import FullLstmPipeline
from evaluation import ModelSelector
from evaluation import ModelEvaluation
from utils_data import get_absolute_path

data_absolute_path = get_absolute_path('../data/lstm_baseline_predictions.csv')

model_config = {
    'target_column': 'anomaly',
    'n_steps' : 5,
    'num_splits' : 34,
    'th' : 0.91
    }

selector = ModelSelector(model_config,FullLstmPipeline,min_splits=32,max_splits=35)
res = selector.select_best(f1_th=0.5)
base_preds = pd.read_csv(data_absolute_path)['y_pred']

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


def main():
    eval = ModelEvaluation(eval_config=eval_config)
    eval.plot_metrics()

if __name__ == "__main__":
    main()