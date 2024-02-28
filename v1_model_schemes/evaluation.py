import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from typing import Any, Dict
from utils_data import AbstractFullModelPipeline
import copy
from IPython.display import display

def calc_model_metrics(y_true,y_pred,th):
    accuracy = accuracy_score(y_true, np.where(y_pred>=th,1,0))
    cm = confusion_matrix(y_true, np.where(y_pred>=th,1,0))
    tn = cm[0,0]
    tp = cm[1,1]
    fn = cm[1,0]
    fp = cm[0,1]
    mar = round(fn / (fn + tp), 4) if (fn + tp) != 0 else 0
    far = round(fp/(fp+tp),4) if (fp+tp) != 0 else 0 
    f1 = round(f1_score(y_true, np.where(y_pred>=th,1,0)),4)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    return accuracy,cm,mar,far,f1,fpr, tpr, thresholds, roc_auc    

class ModelEvaluation:
    
    """
    eval_config = {
        model_name: {
            'y_test': [],
            'y_pred': [],
            'th': ,
            'mar_score: ,
            'f1 score: ,
        },
        model_name: {
            'y_test': [],
            'y_pred': [],
            'th': ,
            'mar_score: ,
            'f1 score: ,
            
        }
    }
    """

    def __init__(self,eval_config) -> None:
        self.eval_config = eval_config
    
    def calc_metrics(self):
        for model_name in self.eval_config.keys():
            accuracy,cm,mar,far,f1,fpr, tpr, thresholds, roc_auc = calc_model_metrics(
                self.eval_config[model_name]['y_test'],
                self.eval_config[model_name]['y_pred'],
                self.eval_config[model_name]['th']
                )
            
            self.eval_config[model_name]['accuracy'] = accuracy
            self.eval_config[model_name]['cm'] = cm
            self.eval_config[model_name]['mar_score'] = mar
            self.eval_config[model_name]['far_score'] = far
            self.eval_config[model_name]['f1'] = f1
            
            self.eval_config[model_name]['fpr'] = fpr
            self.eval_config[model_name]['tpr'] = tpr
            self.eval_config[model_name]['thresholds'] = thresholds
            self.eval_config[model_name]['roc_auc'] = roc_auc

    def plot_metrics(self,title = 'Model Evaluation Metrics'):
        self.calc_metrics()

        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14, 5))
        fig.suptitle(title, fontsize=16, y=1.02)
        roc_curve_ax = ax[2]
        colors = ['blue','darkorange']
        
        for i,model_name in enumerate(self.eval_config.keys()):
            cm_ax = ax[i]
            ConfusionMatrixDisplay(confusion_matrix=self.eval_config[model_name]['cm']).plot(ax=cm_ax,cmap='Blues',values_format='d')
            cm_ax.set_title(f'{model_name} Confusion Matrix')
            cm_ax.set_xlabel('Predicted labels')
            cm_ax.set_ylabel('True labels')
            
            label=f'AUC {model_name} = {self.eval_config[model_name]["roc_auc"]:.2f}'
            roc_curve_ax.plot(
                self.eval_config[model_name]['fpr'], 
                self.eval_config[model_name]['tpr'],
                color=colors[i], lw=2, label=label
            )
        roc_curve_ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        roc_curve_ax.set_xlabel('False Positive Rate')
        roc_curve_ax.set_ylabel('True Positive Rate')
        roc_curve_ax.set_title('ROC Curve')
        roc_curve_ax.legend(loc='lower right')
            
        plt.tight_layout()
        plt.show()

class ModelSelector:
    """
    model_config = {
        'target_column': 'anomaly',
        'look_back': 10,
        'num_splits': 5,
        'th': 0.5
}
    """
    def __init__(self,model_config:Dict[str,Any],model:AbstractFullModelPipeline,min_splits=1,max_splits=20) -> None:
        
        self.model_config = model_config
        self.model =model
        self.best_model = None
        self.min_splits = min_splits
        self.max_splits = max_splits
        self.summary_table = None
    
    def run_model_permutations(self):
        results = {}
        config = copy.deepcopy(self.model_config)
        model = copy.deepcopy(self.model)
        summary_table = pd.DataFrame(index=['f1', 'mar','far'])
        
        for i in range(self.min_splits,self.max_splits+1):
            print(f"Run model number {i}")
            config['num_splits'] = i
            full_model = model(config)
            y_pred = full_model.run()
            y_test = full_model.y_test
            accuracy,cm,mar,far,f1,fpr, tpr, thresholds, roc_auc = calc_model_metrics(y_true=y_test,y_pred=y_pred,th=config['th'])
            print('mar', mar,'far', far, 'f1', f1)
            
            results[f'model_{i}'] =  {
                'mar': mar,
                'far': far,
                'f1': f1,
                'th': self.model_config['th'],
                'predictions' : y_pred,
                'model_name' : f'model_{i}',
                'model': full_model,
            }
            
            summary_table = summary_table.assign(
                **{
                    f'model_{i}' : [f1, mar, far],
                }
            )
            
        return summary_table, results
    
    def select_best(self,f1_th):
        summary_table,results = self.run_model_permutations()
        table = summary_table.T.sort_values(by=['mar','f1'],ascending=[False,False])
        
        display(table.style.background_gradient(axis=0))
        best_model_name = table[table['f1'] >= f1_th].head(1)
        
        if best_model_name.empty:
            best_model_name = table.head(1)
        
        self.best_model = results[str(best_model_name.index[0])]
        display(best_model_name.style.set_caption("Best Model Scores"))
        
        return self.best_model
    