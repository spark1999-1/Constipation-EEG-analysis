"""
Perform inter-group statistical analysis on EEG features (relative power), Fig 6
"""

import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import defaultdict
from itertools import combinations
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests

import matplotlib
matplotlib.use("Qt5Agg")
matplotlib.rcParams['svg.fonttype'] = 'none'
plt.ion()  # if not showing figure: plt.ioff()
# adjust figure format
plt.rc('font', size=8.5, family='Arial', weight='normal')
matplotlib.rcParams['axes.labelsize'] = 8
matplotlib.rcParams['axes.labelweight'] = 'normal'
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['axes.titlesize'] = 9
matplotlib.rcParams['axes.titleweight'] = 'normal'
matplotlib.rcParams['axes.linewidth'] = 1.0
matplotlib.rcParams['svg.fonttype'] = 'none'


def feature_barplot(task_data, task_name, feature, p_all):
    """
    Plot bar comparison charts of relative power by group and frequency band
    """
    bands = df['Band'].unique()
    groups = task_data['Group'].unique()

    # figure size
    width_cm = 7
    height_cm = width_cm * 0.75  # 宽高比（可调整）
    width_inch = width_cm / 2.54
    height_inch = height_cm / 2.54

    fig, ax = plt.subplots(figsize=(width_inch, height_inch), layout='constrained')

    # Set different colors for each brain region
    if feature == 'parietal':
        colors = {
            "Patients": '#C0392B',
            "Controls": "#F1948A",
        }
    if feature == 'temporal':
        colors = {
            "Patients": '#2E8B57',
            "Controls": "#7DCEA0",
        }
    if feature == 'occipital':
        colors = {
            "Patients": "#884EA0",
            "Controls": "#D2B4DE",
        }
    if feature == 'frontal':
        colors = {
            "Patients": '#4C72B0',
            "Controls": "#AED6F1",
        }
    if feature == 'central':
        colors = {
            "Patients": '#E67E22',
            "Controls": "#FF9F43",
        }

    sns.barplot(
        x="Band",
        y=feature,
        hue="Group",
        data=task_data,
        palette=colors,
        errwidth=0.8,
        ax=ax,
        legend=True,
        hue_order=['Controls', 'Patients']
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels,
              title='',
              frameon=False,
              loc='upper right',
              fontsize=8.5)

    ax.set_xlabel('')
    ax.set_ylabel('Relative power (' + feature + ')', fontsize=8.5)
    ax.set_xticklabels(bands, fontsize=8.5)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_title(f"{task_name}", fontsize=9)

    # All group pairs
    group_pairs = list(combinations(groups, 2))

    # Define the group pairs to be compared
    pairs = []
    for band in bands:
        for group1, group2 in group_pairs:
            pairs.append(((band, group1), (band, group2)))

    # Merge p-values & Cohen's d into annotation strings
    valid_pairs = []
    valid_annotations = []
    for pair_key in pairs:
        band, group1, group2 = pair_key[0][0], pair_key[0][1], pair_key[1][1]
        condition = (p_all['Feature'] == feature) & \
                    (p_all['Band'] == band) & \
                    (p_all['Task'] == task_name)

        p_val = p_all.loc[condition, 'P_Adjusted (q-value)']
        p_val = p_val.iloc[0]
        d_val = p_all.loc[condition, 'Cohen_d']
        d_val = d_val.iloc[0]

        # p '*'
        asterisk = convert_pvalue_to_asterisks(p_val)
        # Cohen's d '#'（|d|>0.5）
        hash_mark = "#" if (not np.isnan(d_val)) and (abs(d_val) > 0.5) else ""

        if asterisk and hash_mark:
            annotation = f"{asterisk}, {hash_mark}"
        elif asterisk:
            annotation = asterisk
        elif hash_mark:
            annotation = hash_mark
        else:
            annotation = ""

        # Keep only annotated pairs
        if annotation != "":
            valid_pairs.append(pair_key)
            valid_annotations.append(annotation)

    # Add significance annotations
    if valid_pairs != []:
        annotator = Annotator(
            ax,
            pairs=valid_pairs,
            x="Band",
            y=feature,
            hue="Group",
            data=task_data,
            order=bands,
        )

        annotator.set_custom_annotations(valid_annotations)
        annotator.annotate()

    plt.tight_layout()
    return fig


def cohens_d(group1, group2):
    """
    Calculate the Cohen's d effect size for two groups
    """
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
    cohen_d = (mean1 - mean2) / pooled_std

    return cohen_d

def convert_pvalue_to_asterisks(pvalue):
    """
    Map p-values to * symbols
    """
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return ""  # n.s.

def apply_fdr(group):
    pvals = group['P_Raw'].values
    reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
    group = group.copy()
    group['P_Adjusted (q-value)'] = pvals_corrected
    group['Is_Significant_FDR'] = reject
    return group

def run_fdr_corrected_ancova(data, feature_names):
    """
    Run ANCOVA and apply FDR correction
    """
    results_list = []
    group_var = 'Group'
    age_covariate = 'Age'

    unique_bands = data['Band'].unique()
    unique_tasks = data['Task'].unique()

    ## stage 1：ANCOVA
    print("--- Stage 1: Run ANCOVA ---")
    for feature in feature_names:
        for task in unique_tasks:
            for band in unique_bands:

                subset_data = data[
                    (data['Task'] == task) &
                    (data['Band'] == band)
                    ].copy()

                try:
                    # use age as a covariate
                    formula = f'Q("{feature}") ~ C({group_var}) + {age_covariate}'
                    model = ols(formula, data=subset_data).fit()
                    anova_table = sm.stats.anova_lm(model, typ=2)

                    # raw p-value
                    p_raw = anova_table.loc[f'C({group_var})', 'PR(>F)']

                    # Cohen's d
                    groups = subset_data[group_var].unique()
                    g1_data = subset_data[subset_data[group_var] == groups[0]][feature]
                    g2_data = subset_data[subset_data[group_var] == groups[1]][feature]
                    cohens_d_val = cohens_d(g1_data, g2_data)

                    results_list.append({
                        'Feature': feature,
                        'Task': task,
                        'Band': band,
                        'P_Raw': p_raw,
                        'TestType': 'ANCOVA',
                        'Cohen_d': cohens_d_val
                    })

                except Exception as e:
                    print(f"Error processing {feature} {task} {band}: {e}")
                    results_list.append({
                        'Feature': feature,
                        'Task': task,
                        'Band': band,
                        'P_Raw': np.nan,
                        'TestType': 'Error',
                        'Cohen_d': np.nan
                    })

    # DataFrame
    df_results = pd.DataFrame(results_list).dropna(subset=['P_Raw'])

    ## stage 2：FDR (Benjamini-Hochberg)
    print("--- Stage 2: Benjamini-Hochberg FDR ---")
    df_corrected = df_results.groupby(['Task', 'Band'], group_keys=False).apply(apply_fdr)

    return df_corrected.sort_values(by='P_Raw')


if __name__ == "__main__":
    type = ['Patients', 'Controls']

    # Age
    patients_ages = [80, 23, 25, 73, 67, 38, 56, 44, 57, 42, 45, 74, 66, 51, 62, 73, 38, 29, 52, 69, 62]
    healthy_ages = [57, 54, 58, 50, 51, 50, 54, 52, 55, 55, 52, 46, 49, 47, 54, 51, 54, 54, 31, 22, 30, 29, 29, 26, 26, 30, 22, 40, 25, 34, 21, 26, 37, 22, 35, 38, 25]

    age_dict = {
        'Patients': patients_ages,
        'Controls': healthy_ages,
    }

    # dictionary of relative power
    all_features = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(list)
            )
        )
    )

    # load relative power
    for type_i in type:
        dir_path = '../data/relative_power_region/' + type_i + '/'
        folder = Path(dir_path)
        all_data = {f.stem: json.load(open(f)) for f in folder.glob("*.json")}

        # Store all relative power across all populations, tasks, and frequency bands in a dictionary
        for key_all, value_all in all_data.items():
            for task_name, value_task in value_all.items():
                for band_name, band_value in value_task.items():
                    for feature_name, feature_value in band_value.items():
                        all_features[type_i][task_name][band_name][feature_name].append(feature_value)

    # names of brain regions
    group_names = list(all_features.keys())
    task_names = list(all_features[group_names[0]].keys())
    band_names = list(all_features[group_names[0]][task_names[0]].keys())
    feature_names = list(all_features[group_names[0]][task_names[0]][band_names[0]].keys())

    # from dict to DataFrame
    rows = []
    for group in all_features.keys():
        for task in all_features[group].keys():
            for band in all_features[group][task].keys():
                band_data = all_features[group][task][band]
                n_samples = len(band_data[feature_names[0]])

                for i in range(n_samples):
                    row = {
                        "Group": group,
                        "Task": task,
                        "Band": band,
                        "Age": age_dict[group][i]
                    }

                    for feature in feature_names:
                        row[feature] = band_data[feature][i]

                    rows.append(row)

    df = pd.DataFrame(rows)

    # Convert the brain region names to lowercase
    rename_dict = {col: col[0].lower() + col[1:] for col in feature_names}
    df.rename(columns=rename_dict, inplace=True)
    feature_names_new = rename_dict.values()

    # statistic results of ANCOVA after FDR
    df_results = run_fdr_corrected_ancova(data=df.copy(), feature_names=feature_names_new)

    # Plot the relative power differences between groups
    tasks = df['Task'].unique()
    feature_names = rename_dict.values()

    for task in tasks:
        task_data = df[df['Task'] == task]
        for feature in feature_names:
            fig1 = feature_barplot(task_data, task, feature=feature, p_all=df_results)

    print('a')