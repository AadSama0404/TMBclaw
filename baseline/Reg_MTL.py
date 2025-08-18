# -*- coding: utf-8 -*-
"""
@Time    : 2025/8/18 11:24
@Author  : AadSama
@Software: Pycharm
"""
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pandas as pd
import random

from tmb_dataset import pad_dataset
from model.MLP import MLP
from evaluation.metrics_calculation import Metrics_Calculation
from evaluation.KM_plot import KM_Plot

import warnings
warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


## Load dataset features
L = torch.load("../data/L.pt")
S = torch.load("../data/S.pt")
cohort_num = torch.load("../data/cohort_num.pt")
clone_num = torch.load("../data/clone_num.pt")
max_clone = clone_num

## load hyperparameters
gamma = torch.load('../hyperparameter/gamma.pt')
pos_weights = torch.load('../hyperparameter/pos_weights.pt')
epochs = torch.load('../hyperparameter/epochs.pt')
lrs = torch.load('../hyperparameter/lrs.pt')

## Training results
study_cv = []
true_label_cv = []
PFS_cv = []
Status_cv = []
score_cv = []
group_cv = []
fold_cv = []
A_matrix_cv = []


def Reg_MLP_Train(train_loader, models, optimizers, fold):
    '''
    sorted_data: [['TMB_sum', 'AF_avg', 'CCF_clone']]
    response: ['Study ID', 'ORR', 'PFS', 'Status']
    '''
    for i in range(cohort_num):
        models[i].train()

    NLL_loss = 0.

    for batch_idx, (features, response) in enumerate(train_loader):
        features, response = features.squeeze(0), response.squeeze(0)
        study_id = response[0]
        study_index = int(study_id - 1)
        label = response[1]

        loss = []
        error = []
        theta_X = []
        for i in range(cohort_num):
            loss_i, predicted_prob_i, error_i, _ = models[i].calculate(features, label, pos_weights[fold][i])
            loss.append(loss_i)
            error.append(error_i)
            theta_X.append(predicted_prob_i)

        # Calculate the first part of the loss related to S and loss
        loss_subgroup = sum(loss[i] * S[study_index][i] for i in range(cohort_num))
        NLL_loss = NLL_loss + loss[study_index].item()

        # Stack theta_X into a vector
        theta_X_vec = torch.stack(theta_X).squeeze(1)  # shape: [subgroup_num, 1]
        # Convert L to a PyTorch tensor
        L_tensor = torch.tensor(L, dtype=torch.float32)  # shape: [subgroup_num, subgroup_num]
        # Compute regularization using matrix multiplication
        loss_regularization = torch.mm(theta_X_vec.T, torch.mm(L_tensor, theta_X_vec))  # theta_X^T L theta_X
        # Final loss
        loss_all = loss_subgroup + gamma * loss_regularization

        # Backpropagate
        for i in range(cohort_num):
            optimizers[i].zero_grad()
        loss_all.backward()
        for i in range(cohort_num):
            optimizers[i].step()

    avg_NLL_loss = NLL_loss / len(train_loader)
    print(f"Train Loss: {avg_NLL_loss:.4f}")


def Reg_MLP_Val(val_loader, models, save_flag, fold):
    for i in range(cohort_num):
        models[i].eval()

    test_loss_all = 0.
    test_error_all = 0.
    prediction_list = []

    with torch.no_grad():
        for batch_idx, (features, response) in enumerate(val_loader):
            features, response = features.squeeze(0), response.squeeze(0)
            study_id = response[0]
            study_index = int(study_id - 1)
            label = response[1]
            PFS = response[2]
            Status = response[3]

            loss, predicted_prob, error, predicted_label = models[study_index].calculate(features, label)

            test_loss_all = test_loss_all + loss.data[0]
            test_error_all = test_error_all + error

            prediction_list.append([label.item(), predicted_prob.item(), predicted_label.item(), study_id.item(), PFS.item(), Status.item()])

    if (save_flag == 1):
        '''
        [label, predicted_prob, predicted_label, study_id, PFS, Status, fold]
        '''
        for i in range(len(prediction_list)):
            study_cv.append(prediction_list[i][3])
            true_label_cv.append(prediction_list[i][0])
            PFS_cv.append(prediction_list[i][4])
            Status_cv.append(prediction_list[i][5])
            score_cv.append(prediction_list[i][1])
            group_cv.append(prediction_list[i][2])
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
    dataset = pad_dataset(raw_data, max_clone)
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    labels = [patient[1] for patient in raw_data]

    for fold, (train_idx, val_idx) in enumerate(kfold.split(raw_data, labels)):
        print(f"Fold {fold + 1} #####################################################################")

        train_subset_raw = Subset(dataset, train_idx)
        train_subset = Oversampling(1, train_subset_raw)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)

        ## Initialize the models
        models = {}
        optimizers = {}
        for i in range(cohort_num):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch.manual_seed(42)
            model = MLP(max_clone).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lrs[fold][i], betas=(0.9, 0.999), weight_decay=10e-5)
            models[i] = model
            optimizers[i] = optimizer

        for epoch in range(epochs[fold]):
            print(f"Epoch {epoch + 1} --------------------------------------------------------------------")
            Reg_MLP_Train(train_loader, models, optimizers, fold)
            Reg_MLP_Val(val_loader, models, epoch == epochs[fold] - 1, fold)

    print("\n")
    print("*****************************************************************************")
    print("Cross validation results:\n")
    result_cv = pd.DataFrame({
        "study": study_cv,
        "true_label": true_label_cv,
        "PFS": PFS_cv,
        "Status": pd.Series(Status_cv).astype(int),
        "score": score_cv,
        "group": pd.Series(group_cv).astype(int),
        "fold": fold_cv
    })
    result_cv.to_csv('../results/Reg_MTL.csv', index=False)
    Metrics_Calculation('../results/Reg_MTL.csv')
    KM_Plot('../results/Reg_MTL.csv')


if __name__ == "__main__":
    raw_data = torch.load("../data/raw_data.pt")
    Cross_Validation(raw_data)