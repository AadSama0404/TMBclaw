# -*- coding: utf-8 -*-
"""
@Time    : 2025/4/28 21:53
@Author  : AadSama
@Software: Pycharm
"""
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter

def KM_Plot(file_path):
    data = pd.read_csv(file_path)

    kmf = KaplanMeierFitter()

    plt.figure(figsize=(5.5, 4))

    ## Draw the survival curves of Score-H (group == 1) and Score-L (group == 0) respectively
    for group in [1, 0]:
        group_data = data[data['group'] == group]

        ## Survival analysis using Kaplan-Meier Fitter
        kmf.fit(group_data['PFS'], event_observed=group_data['Status'], label=f'Score-{"H" if group == 1 else "L"}')
        color = 'steelblue' if group == 1 else 'salmon'
        kmf.plot(ci_show=True, color=color, linewidth=1.6)

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_linewidth(1.1)
    plt.gca().spines['left'].set_linewidth(1.1)
    plt.title('Kaplan-Meier Survival Curve', fontsize=12)
    plt.xlabel('PFS(month)')
    plt.ylabel('Probability of Survival')
    plt.legend(loc='upper right', frameon=False, fontsize=10)

    group_1_data = data[data['group'] == 1]
    group_0_data = data[data['group'] == 0]
    ## Calculate the p-value for the Log-rank test
    results = logrank_test(group_1_data['PFS'], group_0_data['PFS'], event_observed_A=group_1_data['Status'], event_observed_B=group_0_data['Status'])
    logrank_p_value = results.p_value
    logrank_p_value_formatted = f"{logrank_p_value:.4f}" if logrank_p_value >= 0.0001 else "0.0000"
    plt.text(25, 0.65, f"p-value = {logrank_p_value_formatted}", ha='right', fontsize=12, color='black')

    # Calculate hazard ratio (HR)
    cph = CoxPHFitter()
    data['group'] = data['group'].astype('category')
    cph.fit(data[['PFS', 'Status', 'group']], duration_col='PFS', event_col='Status')
    hr = cph.hazard_ratios_['group']
    hr_formatted = f"{1/hr:.1f}"
    plt.text(25, 0.55, f"HR = {hr_formatted}", ha='right', fontsize=12, color='black')

    print(f"HR = {hr_formatted}")
    print(f"p = {logrank_p_value:.4f}" if logrank_p_value >= 0.0001 else "0.0000")

    plt.show()


if __name__ == "__main__":
    file_path = '../results/TMBclaw.csv'
    KM_Plot(file_path)