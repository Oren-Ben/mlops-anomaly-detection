import numpy as np
from sklearn.metrics import f1_score, accuracy_score,confusion_matrix,roc_curve, auc

def calc_metrics(y_true,y_pred,threshold=0.5):
  y_labels = np.where(y_pred>=threshold,1,0)
  accuracy = accuracy_score(y_true, y_labels)
  cm = confusion_matrix(y_true, y_labels)
  f1 = f1_score(y_true, y_labels)
  fpr, tpr, thresholds = roc_curve(y_true, y_pred)
  roc_auc = auc(fpr, tpr)

  return { 
    'accuracy': accuracy,
    'cm': cm,
    'f1': f1,
    'fpr': fpr,
    'tpr': tpr, 
    'thresholds': thresholds, 
    'roc_auc': roc_auc 
  }
 