from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt



class ModelEvaluation:
    

    def calc_metrics(self, y_true,y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        return accuracy,cm,f1,fpr, tpr, thresholds, roc_auc


    def plot_metrics(self ,cm, fpr, tpr, roc_auc, thresholds, title = 'Model Evaluation Metrics'):
        
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
        fig.suptitle(title, fontsize=16, y=1.02)
        confusion_matrix_ax = ax[0]
        roc_curve_ax = ax[1]

        # Confusion Matrix in the first subplot
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=confusion_matrix_ax)
        confusion_matrix_ax.set_title('Confusion Matrix')
        confusion_matrix_ax.set_xlabel('Predicted labels')
        confusion_matrix_ax.set_ylabel('True labels')

        # ROC Curve in the second subplot
        roc_curve_ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
        roc_curve_ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        roc_curve_ax.set_xlabel('False Positive Rate')
        roc_curve_ax.set_ylabel('True Positive Rate')
        roc_curve_ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        roc_curve_ax.legend(loc='lower right')

        # Annotate some key thresholds on the curve
        # Select thresholds to annotate on the ROC curve
        indices_to_annotate = [0, len(thresholds) // 3, 2 * len(thresholds) // 3, -1]
        for i in indices_to_annotate:
            roc_curve_ax.annotate(f'{thresholds[i]:.2f}', (fpr[i], tpr[i]), textcoords="offset points", xytext=(10,-5), ha='center')

        # Display the plot
        plt.tight_layout()
        plt.show()
