# -*- coding: utf-8 -*-
"""
@Time    : 2025/4/28 21:38
@Author  : AadSama
@Software: Pycharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, max_clone):
        super(MLP, self).__init__()
        self.input = 3 * max_clone
        self.hidden1 = 64
        self.hidden2 = 32
        self.hidden3 = 16
        self.output = 1

        self.layer1 = nn.Linear(self.input, self.hidden1)
        self.layer2 = nn.Linear(self.hidden1, self.hidden2)
        self.layer3 = nn.Linear(self.hidden2, self.hidden3)
        self.layer4 = nn.Linear(self.hidden3, self.output)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        Y_prob = self.sigmoid(self.layer4(x)).unsqueeze(-1)
        Y_hat = torch.ge(Y_prob, 0.5).float()
        return Y_prob, Y_hat

    def calculate(self, X, Y, pos_weight = 1):
        Y = Y.float()
        Y_prob, Y_hat= self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (pos_weight * Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()
        return neg_log_likelihood, Y_prob, error, Y_hat