# -*- coding: UTF-8 -*-
'''
@Time    : 2025/4/29 11:20
@Author  : AadSama
@Software: Pycharm
'''
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import random
import statistics

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
clone_num = torch.load("../data/clone_num.pt")
max_clone = clone_num

## load hyperparameters
separate_pos_weights = torch.load('../hyperparameter/pos_weights.pt')
pos_weights = [statistics.median(row) for row in separate_pos_weights]
epochs = torch.load('../hyperparameter/epochs.pt')
separate_lrs = torch.load('../hyperparameter/lrs.pt')
lrs = [statistics.median(row) for row in separate_lrs]

## Training results
study_cv = []
true_label_cv = []
PFS_cv = []
Status_cv = []
score_cv = []
group_cv = []
fold_cv = []


def Pooled_MLP_Train(train_loader, model, optimizer, fold):
    '''
    sorted_data: [['TMB_sum', 'AF_avg', 'CCF_clone']]
    response: ['Study ID', 'ORR', 'PFS', 'Status']
    '''
    train_loss = 0.
    train_error = 0.

    model.train()

    for batch_idx, (features, response) in enumerate(train_loader):
        features, response = features.squeeze(0), response.squeeze(0)
        label = response[1]

        loss, predicted_prob, error, _ = model.calculate(features, label, pos_weights[fold])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss = train_loss + loss.item()
        train_error = train_error + error

    avg_NLL_loss = train_loss / len(train_loader)
    print(f"Train Loss: {avg_NLL_loss:.4f}")


def Pooled_MLP_Val(val_loader, model, save_flag, fold):
    model.eval()

    test_loss_all = 0.
    test_error_all = 0.
    prediction_list = []

    with torch.no_grad():
        for batch_idx, (features, response) in enumerate(val_loader):
            features, response = features.squeeze(0), response.squeeze(0)
            study_id = response[0]
            label = response[1]
            PFS = response[2]
            Status = response[3]

            loss, predicted_prob, error, predicted_label = model.calculate(features, label)

            test_loss_all = test_loss_all + loss.data[0]
            test_error_all = test_error_all + error

            prediction_list.append([label.item(), predicted_prob.item(), predicted_label.item(), study_id.item(), PFS.item(), Status.item()])

    test_loss_all = test_loss_all / len(val_loader)
    test_error_all = test_error_all / len(val_loader)
    print(f'Test Loss: {test_loss_all.item():.4f}, Test error: {test_error_all:.4f}')

    if(save_flag ==1):
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)
        model = MLP(max_clone).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lrs[fold], betas=(0.9, 0.999), weight_decay=10e-5)

        for epoch in range(epochs[fold]):
            print(f"Epoch {epoch + 1} --------------------------------------------------------------------")
            Pooled_MLP_Train(train_loader, model, optimizer, fold)
            Pooled_MLP_Val(val_loader, model, epoch==epochs[fold]-1, fold)

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
    result_cv.to_csv('../results/Pooled_MLP.csv', index=False)
    Metrics_Calculation('../results/Pooled_MLP.csv')
    KM_Plot('../results/Pooled_MLP.csv')


if __name__ == "__main__":
    raw_data = torch.load("../data/raw_data.pt")
    Cross_Validation(raw_data)