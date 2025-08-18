# -*- coding: UTF-8 -*-
'''
@Time    : 2025/4/29 15:42
@Author  : AadSama
@Software: Pycharm
'''
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import random
from sklearn import svm

from tmb_dataset import pad_dataset
from evaluation.metrics_calculation import Metrics_Calculation
from evaluation.KM_plot import KM_Plot

import warnings
warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


## Load dataset features
cohort_num = torch.load("../data/cohort_num.pt")
clone_num = torch.load("../data/clone_num.pt")
max_clone = clone_num

## Training results
study_cv = []
true_label_cv = []
PFS_cv = []
Status_cv = []
score_cv = []
group_cv = []
fold_cv = []


def Separate_SVM_Train_Test(train_loader, test_loader, subgroup_num, fold):
    subgroup_train_features = [[] for _ in range(subgroup_num)]
    subgroup_train_labels = [[] for _ in range(subgroup_num)]

    for batch_idx, (features, response) in enumerate(train_loader):
        features, response = features.squeeze(0), response.squeeze(0)
        study_id = response[0]
        study_index = int(study_id - 1) % subgroup_num  # 确保索引在0到subgroup_num-1之间

        subgroup_train_features[study_index].append(features.numpy())
        subgroup_train_labels[study_index].append(response[1].item())

    subgroup_models = []
    for i in range(subgroup_num):
        if len(subgroup_train_features[i]) == 0:
            subgroup_models.append(None)
            continue

        X_train = np.array(subgroup_train_features[i])
        y_train = np.array(subgroup_train_labels[i])

        model = svm.SVC(probability=True)
        model.fit(X_train, y_train)
        subgroup_models.append(model)

    for batch_idx, (features, response) in enumerate(test_loader):
        features, response = features.squeeze(0), response.squeeze(0)

        study_id = response[0]
        study_index = int(study_id - 1) % subgroup_num
        label = response[1].item()
        PFS = response[2].item()
        Status = response[3].item()

        model = subgroup_models[study_index]
        if model is None:
            continue

        features_np = features.numpy().reshape(1, -1)
        score = model.predict_proba(features_np)[0, 1]

        study_cv.append(study_id.item())
        true_label_cv.append(label)
        PFS_cv.append(PFS)
        Status_cv.append(Status)
        score_cv.append(score)
        group_cv.append(model.predict(features_np)[0])
        fold_cv.append(fold)


def Oversampling(oversample_rate, train_subset_raw):
    if(oversample_rate == 0):
        return train_subset_raw
    minority_samples = []
    majority_samples = []
    for i in range(len(train_subset_raw)):
        features, response = train_subset_raw[i]
        label = response[1].item()
        if label == 1:
            minority_samples.append((features, response))
        else:
            majority_samples.append((features, response))
    num_minority = len(minority_samples)
    num_majority = len(majority_samples)
    num_to_add = int((num_majority - num_minority) * oversample_rate)
    oversampled_minority = minority_samples.copy()
    for _ in range(num_to_add):
        chosen_sample = random.choice(minority_samples)
        oversampled_minority.append(chosen_sample)
    oversampled_data = majority_samples + oversampled_minority
    return oversampled_data


def Cross_Validation(raw_data):
    '''
    raw_data: [['Study ID', 'ORR', 'PFS', 'Status', ['TMB_sum', 'AF_avg', 'CCF_clone']]]
    '''
    dataset = pad_dataset(raw_data, clone_num)
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    labels = [patient[1] for patient in raw_data]

    for fold, (train_idx, val_idx) in enumerate(kfold.split(raw_data, labels)):
        print(f"Fold {fold + 1} ############################################################")

        train_subset_raw = Subset(dataset, train_idx)
        train_subset = Oversampling(1, train_subset_raw)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)

        Separate_SVM_Train_Test(train_loader, val_loader, cohort_num, fold)


    result_cv = pd.DataFrame({
        "study": study_cv,
        "true_label": true_label_cv,
        "PFS": PFS_cv,
        "Status": pd.Series(Status_cv).astype(int),
        "score": score_cv,
        "group": pd.Series(group_cv).astype(int),
        "fold": fold_cv
    })
    result_cv.to_csv('../results/Separate_SVM.csv', index=False)
    Metrics_Calculation('../results/Separate_SVM.csv')
    KM_Plot('../results/Separate_SVM.csv')


if __name__ == "__main__":
    raw_data = torch.load("../data/raw_data.pt")
    Cross_Validation(raw_data)