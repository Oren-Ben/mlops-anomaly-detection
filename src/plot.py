import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def plot(baseline, test, title = 'Model Evaluation Metrics'):
  fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))
  fig.suptitle(title, fontsize=16, y=1.02)
  
  baseline_confusion_matrix_ax = ax[0]
  test_confusion_matrix_ax = ax[1]
  roc_curve_ax = ax[2]
  
  # Baseline Confusion Matrix in the first subplot
  ConfusionMatrixDisplay(confusion_matrix=baseline['cm']).plot(ax=baseline_confusion_matrix_ax)
  baseline_confusion_matrix_ax.set_title('Baseline Confusion Matrix')
  baseline_confusion_matrix_ax.set_xlabel('Predicted labels')
  baseline_confusion_matrix_ax.set_ylabel('True labels')

  # Test Confusion Matrix in the first subplot
  ConfusionMatrixDisplay(confusion_matrix=test['cm']).plot(ax=test_confusion_matrix_ax)
  test_confusion_matrix_ax.set_title('Test Confusion Matrix')
  test_confusion_matrix_ax.set_xlabel('Predicted labels')
  test_confusion_matrix_ax.set_ylabel('True labels')

  # ROC Curve in the second subplot
  roc_curve_ax.plot(baseline['fpr'], baseline['tpr'], color='green', lw=2, label=f"Baseline AUC = {baseline['roc_auc']:.2f}")
  roc_curve_ax.plot(test['fpr'], test['tpr'], color='darkorange', lw=2, label=f"Test AUC = {test['roc_auc']:.2f}")

  roc_curve_ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  roc_curve_ax.set_xlabel('False Positive Rate')
  roc_curve_ax.set_ylabel('True Positive Rate')
  roc_curve_ax.set_title('Receiver Operating Characteristic (ROC) Curve')
  roc_curve_ax.legend(loc='lower right')

  # Display the plot
  plt.tight_layout()
  plt.show()