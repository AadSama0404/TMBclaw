# -*- coding: utf-8 -*-
"""
@Time    : 2025/4/28 21:55
@Author  : AadSama
@Software: Pycharm
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, f1_score, confusion_matrix, accuracy_score


def Metrics_Calculation(file_path):
    result_df = pd.read_csv(file_path)

    auc_list, acc_list = [], []
    f1_class0_list, f1_class1_list = [], []
    tp_list, tn_list = [], []
    ppv_list, npv_list, recall_list, specificity_list = [], [], [], []
    dop_list, best_threshold_list = [], []

    folds = result_df['fold'].unique()

    new_group = result_df['group'].copy() if 'group' in result_df.columns else pd.Series(index=result_df.index, dtype=int)

    for fold in folds:
        fold_df = result_df[result_df['fold'] == fold]
        true_labels = fold_df['true_label'].values
        scores = fold_df['score'].values

        thresholds = np.linspace(min(scores), max(scores), num=100)
        dop_per_threshold = []

        for thresh in thresholds:
            predicted_labels = (scores >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels, labels=[0,1]).ravel()

            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            dop = np.sqrt((recall - 1) ** 2 + (specificity - 1) ** 2 + (ppv - 1) ** 2 + (npv - 1) ** 2)
            dop_per_threshold.append(dop)

        best_idx = np.argmin(dop_per_threshold)
        best_thresh = thresholds[best_idx]
        best_threshold_list.append(best_thresh)

        new_labels = (scores >= best_thresh).astype(int)
        new_group.loc[fold_df.index] = new_labels

        auc_list.append(roc_auc_score(true_labels, scores))
        acc_list.append(accuracy_score(true_labels, new_labels))
        f1_scores = f1_score(true_labels, new_labels, average=None, zero_division=0)
        f1_class0_list.append(f1_scores[0])
        f1_class1_list.append(f1_scores[1])
        tn, fp, fn, tp = confusion_matrix(true_labels, new_labels, labels=[0,1]).ravel()
        tp_list.append(tp)
        tn_list.append(tn)
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv_list.append(ppv)
        npv_list.append(npv)
        recall_list.append(recall)
        specificity_list.append(specificity)
        dop = np.sqrt((recall - 1) ** 2 + (specificity - 1) ** 2 + (ppv - 1) ** 2 + (npv - 1) ** 2)
        dop_list.append(dop)
    result_df['group'] = new_group.values
    def avg(lst): return sum(lst) / len(lst)
    metrics = {
        "Accuracy": avg(acc_list),
        "AUC": avg(auc_list),
        "F1-Class0": avg(f1_class0_list),
        "F1-Class1": avg(f1_class1_list),
        "TP": int(sum(tp_list)),
        "TN": int(sum(tn_list)),
        "DOP": avg(dop_list),
    }

    print("\n=== Cross-Validation Metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float) and "p-value" not in k:
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    result_df.to_csv(file_path, index=False)

    return metrics, result_df


if __name__ == "__main__":
    file_path = '../results/TMBclaw.csv'
    Metrics_Calculation(file_path)