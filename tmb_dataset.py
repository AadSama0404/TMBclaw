# -*- coding: utf-8 -*-
"""
@Time    : 2025/4/28 21:46
@Author  : AadSama
@Software: Pycharm
"""
from torch.utils.data import Dataset
import torch


class TMBclaw_dataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        '''
            raw_data: raw_data: [['Study ID', 'ORR', 'PFS', 'Status', ['TMB_sum', 'AF_avg', 'CCF_clone']]]

            sorted_data: [['TMB_sum', 'AF_avg', 'CCF_clone']]
            response: ['Study ID', 'ORR', 'PFS', 'Status']
        '''
        sample = self.data[idx]

        study_id = int(sample[0])
        lable = float(sample[1])
        PFS = float(sample[2])
        Status = int(sample[3])
        response = torch.tensor([study_id, lable, PFS, Status], dtype=torch.float)

        features = torch.tensor(sample[4], dtype=torch.float)

        _, sorted_indices = torch.sort(features[:, -1], descending=True)
        sorted_features = features[sorted_indices]

        return (sorted_features, response)


class total_tmb_dataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        study_id = int(sample[0])
        lable = float(sample[1])
        PFS = float(sample[2])
        Status = int(sample[3])
        response = torch.tensor([study_id, lable, PFS, Status], dtype=torch.float)
        features = torch.tensor(sample[4], dtype=torch.float)
        _, sorted_indices = torch.sort(features[:, -1], descending=True)
        sorted_features = features[sorted_indices]
        weights = sorted_features[:, -1]
        sum_weights = torch.sum(weights)
        result = torch.zeros_like(sorted_features[0:1, :])
        result[0, 0] = torch.sum(sorted_features[:, 0] * weights / sum_weights)
        result[0, 1] = torch.sum(sorted_features[:, 1] * weights / sum_weights)
        result[0, -1] = torch.sum(sorted_features[:, -1])

        return (result, response)


class clonal_dataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        study_id = int(sample[0])
        lable = float(sample[1])
        PFS = float(sample[2])
        Status = int(sample[3])
        response = torch.tensor([study_id, lable, PFS, Status], dtype=torch.float)
        features = torch.tensor(sample[4], dtype=torch.float)
        _, sorted_indices = torch.sort(features[:, -1], descending=True)
        sorted_features = features[sorted_indices]
        first_row = sorted_features[0].unsqueeze(0)
        return (first_row, response)


class subclonal_dataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        study_id = int(sample[0])
        lable = float(sample[1])
        PFS = float(sample[2])
        Status = int(sample[3])
        response = torch.tensor([study_id, lable, PFS, Status], dtype=torch.float)
        features = torch.tensor(sample[4], dtype=torch.float)
        _, sorted_indices = torch.sort(features[:, -1], descending=True)
        sorted_features = features[sorted_indices]
        remaining_rows = sorted_features[1:]  # remaining (k-1) rows
        return (remaining_rows, response)


class pad_dataset(Dataset):
    def __init__(self, data_list, max_clone):
        self.data = data_list
        self.max_clone = max_clone
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        study_id = int(sample[0])
        lable = float(sample[1])
        PFS = float(sample[2])
        Status = int(sample[3])
        response = torch.tensor([study_id, lable, PFS, Status], dtype=torch.float)
        features = torch.tensor(sample[4], dtype=torch.float)
        _, sorted_indices = torch.sort(features[:, -1], descending=True)
        sorted_features = features[sorted_indices]

        def flatten_and_pad(tensor, max_clone):
            k = tensor.size(0)
            # If k < max_clone, pad with zeros; if k > max_clone, truncate
            if k < max_clone:
                # Pad with zeros along the rows (dim=0)
                padded = torch.cat([tensor, torch.zeros(max_clone - k, 3)], dim=0)
            else:
                # Truncate to max_clone rows
                padded = tensor[:max_clone, :]
            flattened = padded.view(-1)
            return flattened

        ## Sorted features after flattening and zero padding
        padded_features = flatten_and_pad(sorted_features, self.max_clone)

        return (padded_features, response)