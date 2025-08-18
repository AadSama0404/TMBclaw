# -*- coding: utf-8 -*-
"""
@Time    : 2025/4/28 19:42
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

from tmb_dataset import TMBclaw_dataset
from model.TMBclaw import TMBclaw
from evaluation.metrics_calculation import Metrics_Calculation
from evaluation.KM_plot import KM_Plot
from evaluation.weight_analysis import Violin_Plot
from evaluation.weight_analysis import Heatmap_Plot

import warnings
warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


## Load dataset features
L = torch.load("data/L.pt")
S = torch.load("data/S.pt")
cohort_num = torch.load("data/cohort_num.pt")
clone_num = torch.load("data/clone_num.pt")
max_clone = clone_num

## load hyperparameters
gamma = torch.load('hyperparameter/gamma.pt')
pos_weights = torch.load('hyperparameter/pos_weights.pt')
epochs = torch.load('hyperparameter/epochs.pt')
oversample_rates = torch.load('hyperparameter/oversample_rates.pt')
lrs = torch.load('hyperparameter/lrs.pt')

## Training results
study_cv = []
true_label_cv = []
PFS_cv = []
Status_cv = []
score_cv = []
group_cv = []
fold_cv = []
A_matrix_cv = []


def TMBclaw_Train(train_loader, models, optimizers, fold):
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
            loss_i, predicted_prob_i, error_i, _, _ = models[i].calculate(features, label, pos_weights[fold][i])
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


def TMBclaw_Val(val_loader, models, save_flag, fold):
    for i in range(cohort_num):
        models[i].eval()

    test_loss_all = 0.
    test_error_all = 0.
    prediction_list = []
    A_matrix = np.full((len(val_loader), clone_num + 1), -1.0, dtype=float)

    with torch.no_grad():
        for batch_idx, (features, response) in enumerate(val_loader):
            features, response = features.squeeze(0), response.squeeze(0)
            study_id = response[0]
            study_index = int(study_id - 1)
            label = response[1]
            PFS = response[2]
            Status = response[3]

            loss, predicted_prob, error, predicted_label, A = models[study_index].calculate(features, label)

            test_loss_all = test_loss_all + loss.data[0]
            test_error_all = test_error_all + error

            prediction_list.append([label.item(), predicted_prob.item(), predicted_label.item(), study_id.item(), PFS.item(), Status.item()])
            for i in range(A.shape[1]):
                A_matrix[batch_idx][i] = np.round(A[0][i].detach().numpy(), 4)
            A_matrix[batch_idx][clone_num] = study_id

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
            A_matrix_cv.append(A_matrix[i])


def Oversampling(oversample_rate, train_subset_raw, max_clone):
    '''
    sorted_data: [['TMB_sum', 'AF_avg', 'CCF_clone']]
    response: ['Study ID', 'ORR', 'PFS', 'Status']
    '''
    if (oversample_rate != 0):
        features_list = []
        labels_list = []
        response_dict = {}
        for i in range(len(train_subset_raw)):
            features, response = train_subset_raw[i]
            k, feature_dim = features.shape
            if k < max_clone:
                pad_size = (max_clone - k, feature_dim)
                padded_features = torch.cat([features, torch.zeros(pad_size)], dim=0)
            else:
                padded_features = features[:max_clone]
            flat_features = padded_features.numpy().flatten()
            features_list.append(flat_features)
            labels_list.append(response[1].item())
            response_dict[tuple(flat_features)] = response
        features_array = np.array(features_list)
        labels_array = np.array(labels_list)
        ros = RandomOverSampler(sampling_strategy=oversample_rate, random_state=42)
        features_resampled, labels_resampled = ros.fit_resample(features_array, labels_array)
        resampled_dataset = []
        for i in range(len(features_resampled)):
            resampled_features = torch.tensor(features_resampled[i].reshape(max_clone, 3), dtype=torch.float)
            non_zero_rows = (resampled_features.sum(dim=1) != 0)
            resampled_features = resampled_features[non_zero_rows]
            original_response = response_dict.get(tuple(features_resampled[i]),
                                                  torch.tensor([0, labels_resampled[i], 0, 0], dtype=torch.float))
            resampled_dataset.append((resampled_features, original_response))
    else:
        resampled_dataset = train_subset_raw
    return resampled_dataset


def Cross_Validation(raw_data):
    '''
    raw_data: [['Study ID', 'ORR', 'PFS', 'Status', ['TMB_sum', 'AF_avg', 'CCF_clone']]]
    '''
    dataset = TMBclaw_dataset(raw_data)
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    labels = [patient[1] for patient in raw_data]

    for fold, (train_idx, val_idx) in enumerate(kfold.split(raw_data, labels)):
        print(f"Fold {fold + 1} #####################################################################")

        train_subset_raw = Subset(dataset, train_idx)
        train_subset = Oversampling(oversample_rates[fold], train_subset_raw, max_clone)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)

        ## Initialize the models
        models = {}
        optimizers = {}
        for i in range(cohort_num):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch.manual_seed(42)
            model = TMBclaw().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lrs[fold][i], betas=(0.9, 0.999), weight_decay=10e-5)
            models[i] = model
            optimizers[i] = optimizer

        for epoch in range(epochs[fold]):
            print(f"Epoch {epoch + 1} --------------------------------------------------------------------")
            TMBclaw_Train(train_loader, models, optimizers, fold)
            TMBclaw_Val(val_loader, models, epoch == epochs[fold] - 1, fold)

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
    result_cv.to_csv('results/TMBclaw.csv', index=False)
    Metrics_Calculation('results/TMBclaw.csv')
    KM_Plot('results/TMBclaw.csv')

    df = pd.DataFrame(A_matrix_cv)
    df.to_csv('results/Attention_Weights.csv', index=False, header=False)
    Violin_Plot('results/Attention_Weights.csv')
    Heatmap_Plot('results/Attention_Weights.csv', cohort_num)


if __name__ == "__main__":
    raw_data = torch.load("data/raw_data.pt")
    Cross_Validation(raw_data)