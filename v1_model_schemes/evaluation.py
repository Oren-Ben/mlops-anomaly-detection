import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc

"""
    {
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


def _calc_metrics(y_true,y_pred,th):
    accuracy = accuracy_score(y_true, np.where(y_pred>=th,1,0))
    cm = confusion_matrix(y_true, np.where(y_pred>=th,1,0))
    tn = cm[0,0]
    tp = cm[1,1]
    fn = cm[1,0]
    fp = cm[0,1]
    mar = fn/(fn+tp)
    f1 = f1_score(y_true, np.where(y_pred>=th,1,0))
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    return accuracy,cm,mar,f1,fpr, tpr, thresholds, roc_auc


class ModelEvaluation:
    def __init__(self,models_config) -> None:
        self.models_config = models_config
    
    def calc_metrics(self):
        for model_name in self.models_config.keys():
            accuracy,cm,mar,f1,fpr, tpr, thresholds, roc_auc = _calc_metrics(
                self.models_config[model_name]['y_test'],
                self.models_config[model_name]['y_pred'],
                self.models_config[model_name]['th']
                )
            
            self.models_config[model_name]['accuracy'] = accuracy
            self.models_config[model_name]['cm'] = cm
            self.models_config[model_name]['mar_score'] = mar
            self.models_config[model_name]['f1'] = f1
            
            self.models_config[model_name]['fpr'] = fpr
            self.models_config[model_name]['tpr'] = tpr
            self.models_config[model_name]['thresholds'] = thresholds
            self.models_config[model_name]['roc_auc'] = roc_auc
        

    def plot_metrics(self,title = 'Model Evaluation Metrics'):
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14, 5))
        fig.suptitle(title, fontsize=16, y=1.02)
        roc_curve_ax = ax[2]
        colors = ['blue','darkorange']
        
        for i,model_name in enumerate(self.models_config.keys()):
            cm_ax = ax[i]
            ConfusionMatrixDisplay(confusion_matrix=self.models_config[model_name]['cm']).plot(ax=cm_ax,cmap='Blues')
            cm_ax.set_title(f'{model_name} Confusion Matrix')
            cm_ax.set_xlabel('Predicted labels')
            cm_ax.set_ylabel('True labels')
            
            label=f'AUC = {self.models_config[model_name]["roc_auc"]:.2f}'
            roc_curve_ax.plot(
                self.models_config[model_name]['fpr'], 
                self.models_config[model_name]['tpr'],
                color=colors[i], lw=2, label=label
            )
        roc_curve_ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        roc_curve_ax.set_xlabel('False Positive Rate')
        roc_curve_ax.set_ylabel('True Positive Rate')
        roc_curve_ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        roc_curve_ax.legend(loc='lower right')
            
        plt.tight_layout()
        plt.show()

class ModelSelector:
    def __init__(self,model,params,train_set,test_set) -> None:
        self.train_set =train_set
        self.test_set = test_set
        self.best_model = None
        
    def choose_best(self):
        pass
    