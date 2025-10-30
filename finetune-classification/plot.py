import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_confusion_matrix(class_names, confusion_matrix, result_path):
    cfmt =pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    plt.rcParams['font.size'] = 12
    plt.figure(figsize=(13, 11), dpi=100)
    plt.title('Confusion Matrix')
    ax= sns.heatmap(cfmt, annot=True, cmap='BuGn', fmt="d")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(result_path + '/confusion_matrix.png')

def plot_roc_auc_curve(y_true, y_pred_prob, model_name, result_path):
    y_test_true = np.array(y_true)
    y_test_predprob = np.array(y_pred_prob)
    fpr, tpr, thresholds = roc_curve(y_test_true, y_test_predprob, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(fpr, tpr, color="blue", lw=3, label= model_name + "(area = %.4f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="grey", lw=3, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(result_path + '/roc_auc_curve.png')
    plt.show()