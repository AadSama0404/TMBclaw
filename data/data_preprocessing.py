# -*- coding: utf-8 -*-
"""
@Time    : 2025/4/28 21:09
@Author  : AadSama
@Software: Pycharm
"""
import pandas as pd
import torch
import numpy as np
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')


def Genomic_Data_Preprocess(genomic_data):
    '''
    genomic_data: [['Study ID', 'PATIENT_ID', 'AF', 'CCF', 'clone']]

    key: ['Study ID', 'PATIENT_ID', 'clone']
    values: ['AF', 'CCF']

    instance_list: [['Study ID', 'PATIENT_ID', 'clone', 'TMB_sum', 'AF_avg', 'CCF_clone']]
    '''
    instances = {}
    for row in genomic_data:
        # [PATIENT_ID, Clone ID]
        key = (row[0], row[1], row[4])
        if key not in instances:
            instances[key] = []
        instances[key].append(row[2:4])

    instance_list = []
    for (key1, key2, key3), values in instances.items():
        TMB_num = len(values)
        AF_avg = sum(row[0] for row in values) / TMB_num if values else 0
        CCF_avg = sum(row[1] for row in values) / TMB_num if values else 0
        instance_list.append([key1, key2, key3, TMB_num, AF_avg, CCF_avg])

    ## normalization
    df = pd.DataFrame(instance_list, columns=['Study ID', 'PATIENT_ID', 'clone', 'TMB_num', 'AF_avg', 'CCF_clone'])
    groups = df.groupby('Study ID')
    df_normalized = groups.apply(Quantile_Normalize).reset_index(drop=True)
    instance_list_normalized = np.array(df_normalized).tolist()

    return instance_list_normalized


def Quantile_Normalize(group, q_low=0.1, q_high=0.9):
    upper = group['TMB_num'].quantile(q_high)
    scale_factor = upper
    group['TMB_num'] = group['TMB_num'] / scale_factor
    return group


def Match(bag_list, instance_list):
    '''
    bag_list: [['Study ID', 'PATIENT_ID', 'ORR', 'PFS', 'Status']]
    instance_list: [['Study ID', 'PATIENT_ID', 'clone', 'TMB_sum', 'AF_avg', 'CCF_clone']]

    raw_data: [['Study ID', 'ORR', 'PFS', 'Status', ['TMB_sum', 'AF_avg', 'CCF_clone']]]
    '''
    raw_data = []
    for row in bag_list:
        subgroup_id = row[0]
        patient_id = row[1]
        ORR = row[2]
        PFS = row[3]
        Status = row[4]
        instances = []
        for instance in instance_list:
            if instance[1] == patient_id:
                instances.append([instance[3], instance[4], instance[5]])
        if len(instances) >= 1:
            raw_data.append([subgroup_id, ORR, PFS, Status, instances])
        else:
            print(f"No instance found in this bag. PatientID: {patient_id}")
    return raw_data


def Laplacian_matrix_calculation(raw_data, cohort_num, clone_num):
    '''
    raw_data: [['Study ID', 'ORR', 'PFS', 'Status', ['TMB_sum', 'AF_avg', 'CCF_clone']]]
    '''
    group_features = {i: [] for i in range(cohort_num)}
    max_samples = 0  # Record the maximum number of samples in all groups

    ## Extract features and calculate the maximum number of samples
    for item in raw_data:
        group = int(item[0]) - 1  # Assuming that Study ID starts at 1, adjust to 0 index
        matrix = item[4]
        flat_matrix = [val for sublist in matrix for val in sublist]  # flatten
        padded_matrix = flat_matrix[:clone_num * 3] + [0] * (clone_num * 3 - len(flat_matrix))  # padding
        group_features[group].append(padded_matrix)
        max_samples = max(max_samples, len(group_features[group]))  # Update the maximum number of samples

    feature_dim = clone_num * 3   # Feature dimension of each sample

    group_vectors = []
    for group in range(cohort_num):
        group_matrix = np.array(group_features[group])

        num_samples = group_matrix.shape[0] if len(group_matrix) > 0 else 0

        if num_samples < max_samples:
            padding = np.zeros((max_samples - num_samples, feature_dim))
            group_matrix = np.vstack((group_matrix, padding))

        group_vector = group_matrix.flatten()
        group_vectors.append(group_vector)

    group_vectors = np.array(group_vectors)

    similarity_matrix = np.zeros((cohort_num, cohort_num))
    for i in range(cohort_num):
        for j in range(cohort_num):
            similarity_matrix[i, j] = 1 - cosine(group_vectors[i], group_vectors[j])

    degree_matrix = np.diag(np.sum(similarity_matrix, axis=1))
    laplacian_matrix = degree_matrix - similarity_matrix

    return similarity_matrix, laplacian_matrix


if __name__ == '__main__':
    data_path = pd.ExcelFile("Experimental.xlsx")

    ## Clinical data——bag
    clinical_data = pd.read_excel(data_path, 0, dtype={'Study ID': str, 'PATIENT_ID': str})
    bag_list = clinical_data[['Study ID', 'PATIENT_ID', 'ORR', 'PFS', 'Status']].values.tolist()

    ## Get the number of subgroups
    df = data_path.parse(sheet_name=0)
    cohort_num = df.iloc[:, 0].max()
    torch.save(cohort_num, 'cohort_num.pt')

    ## Genomic data——instance
    genomic_data = pd.read_excel(data_path, 1, dtype={'Study ID': str, "PATIENT_ID": str})
    genomic_data = genomic_data[['Study ID', 'PATIENT_ID', 'AF', 'CCF', 'clone']].values.tolist()
    instance_list = Genomic_Data_Preprocess(genomic_data)

    ## Get the number of clones
    df = data_path.parse(sheet_name=1)
    clone_num = df.iloc[:, -1].max()
    torch.save(clone_num, 'clone_num.pt')

    raw_data = Match(bag_list, instance_list)
    torch.save(raw_data, 'raw_data.pt')

    S, L = Laplacian_matrix_calculation(raw_data, cohort_num, clone_num)
    torch.save(S, 'S.pt')
    torch.save(L, 'L.pt')