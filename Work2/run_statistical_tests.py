import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, norm
from statsmodels.stats.multitest import multipletests
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import os

# -------------------------- #
#     Useful Functions       #
# -------------------------- #

def calculate_global_min_max(dfs, column):
    """
    Calculate global min and max for a specific column across multiple DataFrames.
    
    Parameters:
    - dfs (list of DataFrame): List of DataFrames to consider for min/max calculation.
    - column (str): Column name for which min and max are computed.
    
    Returns:
    - tuple: (global_min, global_max) for the specified column.
    """
    global_min = min(df[column].min() for df in dfs)
    global_max = max(df[column].max() for df in dfs)
    return global_min, global_max

def calculate_m_config_storage(data, min_time, max_time, min_accu, max_accu, min_stor, max_stor, alpha, beta, omega):
    """
    Calculate M_config based on normalized 'Time (seconds)' and 'Accuracy' columns 
    without adding the intermediate normalized columns to the dataset.

    Parameters:
    - data (DataFrame): The input dataset with 'Time (seconds)' and 'Accuracy' columns.
    - alpha (float): Weight for Accuracy in M_config calculation.
    - beta (float): Weight for Time in M_config calculation.

    Returns:
    - DataFrame: Updated dataset with only the 'M_config' column added.
    """
    
    # Normalize 'Time (seconds)', 'Accuracy', and 'Storage percentage' with global min/max
    normalized_time = (data['Time (seconds)'] - min_time) / (max_time - min_time)
    normalized_accuracy = (data['Accuracy'] - min_accu) / (max_accu - min_accu)
    
    if max_stor == min_stor:
        normalized_storage = 1
    else:
        normalized_storage = (data['Storage percentage'] - min_stor) / (max_stor - min_stor)

    # Calculate M_config
    data['M_config'] = (alpha * normalized_accuracy) + (beta * (1 - normalized_time)) + (omega * (1 - normalized_storage))

    return data

def normalize_column(data, column_name):
    """Normalize a column in the dataset."""
    min_value = data[column_name].min()
    max_value = data[column_name].max()
    return (data[column_name] - min_value) / (max_value - min_value)

def calculate_m_config(data, alpha=0.7, beta=0.3):
    """Calculate M_config score based on normalized accuracy and time."""
    data['Normalized_Time'] = normalize_column(data, 'Time (seconds)')
    data['Normalized_Accuracy'] = normalize_column(data, 'Accuracy')
    data['M_config'] = (alpha * data['Normalized_Accuracy']) + (beta * (1 - data['Normalized_Time']))
    return data

def get_top_configurations(data, config_columns, top_n=8):
    """Group by configurations, calculate mean M_config, and return top configurations."""
    config_group = data.groupby(config_columns).agg({
        'M_config': 'mean',
        'Accuracy': 'mean',
        'Time (seconds)': 'mean'
    }).reset_index()
    return config_group.sort_values('M_config', ascending=False).head(top_n).reset_index(drop=True)

def perform_friedman_nemenyi_tests(data, top_configs, config_columns, title):
    """Perform Friedman test and, if applicable, Nemenyi post-hoc test with Holm correction, and save the CD diagram."""
    labels = [f"Config {i+1}" for i in range(len(top_configs))]
    friedman_data = []

    for _, config in top_configs.iterrows():
        config_data = data.copy()
        for col in config_columns:
            config_data = config_data[config_data[col] == config[col]]
        friedman_data.append(config_data['M_config'].values)

    friedman_data = np.array(friedman_data).T
    statistic, p_value = friedmanchisquare(*friedman_data)
    print(f"Friedman test statistic: {statistic:.4f}, p-value: {p_value:.4f}")

    if p_value < 0.05:
        nemenyi_result = sp.posthoc_nemenyi_friedman(friedman_data)
        p_values = nemenyi_result.values[np.triu_indices_from(nemenyi_result, k=1)]
        
        reject, pvals_corrected, _, _ = multipletests(p_values, method='holm')
        
        corrected_p_values_matrix = np.zeros_like(nemenyi_result)
        corrected_p_values_matrix[np.triu_indices_from(corrected_p_values_matrix, k=1)] = pvals_corrected
        corrected_p_values_matrix += corrected_p_values_matrix.T
        nemenyi_result_corrected = pd.DataFrame(corrected_p_values_matrix, index=labels, columns=labels)
                
        ranks_per_observation = np.apply_along_axis(lambda x: rankdata(-x), 1, friedman_data)
        average_ranks = ranks_per_observation.mean(axis=0)
        ranks = pd.Series(average_ranks, index=labels)
        
        # Plot and save Critical Difference Diagram
        sp.critical_difference_diagram(ranks, nemenyi_result_corrected)
        plt.title(title)
        filename = os.path.join("results", title.lower().replace(" ", "_") + ".png")
        plt.savefig(filename)
        plt.clf()
    else:
        print("No significant differences found among configurations")

def run_statistical_tests_best_knn(alpha=0.7, beta=0.3):
    data = pd.read_csv('results/knn_results.csv')
    data['r (if Minkowski)'] = data['r (if Minkowski)'].fillna(0)
    
    dataset_names = ['sick', 'balance']
    results = {}

    for dataset_name in dataset_names:
        print(f"Running tests for dataset: {dataset_name}")
        filtered_data = data[data['Dataset'].str.contains(dataset_name)].copy()
        filtered_data = calculate_m_config(filtered_data, alpha, beta)
        
        config_columns = ['k', 'Feature Weighting Method', 'Selection Method', 'Distance Metric', 'r (if Minkowski)']
        top_configs = get_top_configurations(filtered_data, config_columns)
        
        print(top_configs)
        title = f"BEST KNN {dataset_name.capitalize()} Dataset - Critical Difference Diagram"
        perform_friedman_nemenyi_tests(filtered_data, top_configs, config_columns, title)
        
        results[dataset_name] = top_configs
    
    return results

def run_statistical_tests_best_svm(alpha=0.7, beta=0.3):
    data = pd.read_csv('results/svm_results.csv')
    data['gamma'] = data['gamma'].fillna(0)
    
    dataset_names = ['sick', 'balance']
    results = {}

    for dataset_name in dataset_names:
        print(f"Running tests for dataset: {dataset_name}")
        filtered_data = data[data['Dataset'].str.contains(dataset_name)].copy()
        filtered_data = calculate_m_config(filtered_data, alpha, beta)
        
        config_columns = ['kernel', 'C', 'class_weight', 'shrinking', 'gamma']
        top_configs = get_top_configurations(filtered_data, config_columns)
        
        print(top_configs)
        title = f"BEST SVM {dataset_name.capitalize()} Dataset - Critical Difference Diagram"
        perform_friedman_nemenyi_tests(filtered_data, top_configs, config_columns, title)
        
        results[dataset_name] = top_configs
    
    return results

def perform_wilcoxon_test_knn_vs_svm(best_knn_df, best_svm_df, dataset_name):
    # Extract the 'M_config' values
    knn_m_config = best_knn_df['Accuracy'].values
    svm_m_config = best_svm_df['Accuracy'].values

    # Ensure that both arrays have the same length
    assert len(knn_m_config) == len(svm_m_config), "The two arrays must have the same length."

    # Compute differences
    differences = svm_m_config - knn_m_config

    # Remove zero differences (ties) for ranking purposes
    non_zero_indices = differences != 0
    differences_no_zero = differences[non_zero_indices]
    N = len(differences_no_zero)

    # If all differences are zero
    if N == 0:
        print(f"All differences are zero for {dataset_name} dataset. Cannot perform Wilcoxon test.")
        return

    # Get the ranks of the absolute differences
    ranks = rankdata(np.abs(differences_no_zero), method='average')

    # Initialize sums of ranks
    R_plus = 0.0  # Sum of ranks for positive differences (SVM better)
    R_minus = 0.0  # Sum of ranks for negative differences (KNN better)

    # Calculate R+ and R-
    for i in range(N):
        if differences_no_zero[i] > 0:
            R_plus += ranks[i]
        elif differences_no_zero[i] < 0:
            R_minus += ranks[i]
        # Zero differences are already removed

    # T is the smaller of R_plus and R_minus
    T = min(R_plus, R_minus)

    # Compute z statistic
    z = (T - N*(N+1)/4) / np.sqrt(N*(N+1)*(2*N+1)/24)

    # Get p-value from z
    p_value = 2 * norm.cdf(z)  # Two-tailed test

    # Print results
    print(f"\nWilcoxon signed-rank test results for {dataset_name} dataset:")
    print(f"Number of non-zero differences (N): {N}")
    print(f"Sum of ranks for positive differences (R+): {R_plus}")
    print(f"Sum of ranks for negative differences (R-): {R_minus}")
    print(f"Test statistic (T): {T}")
    print(f"z-value: {z:.4f}")
    print(f"p-value: {p_value:.4f}")

    # Determine which classifier is better
    if R_plus > R_minus:
        better_classifier = "SVM"
    else:
        better_classifier = "KNN"

    print(f"{better_classifier} performs better on the {dataset_name} dataset based on the sums of ranks.")

    # Interpretation
    alpha = 0.05
    if p_value < alpha:
        print(f"Reject the null hypothesis: There is a significant difference between KNN and SVM on the {dataset_name} dataset.")
    else:
        print(f"Fail to reject the null hypothesis: No significant difference between KNN and SVM on the {dataset_name} dataset.")

def run_statistical_tests_svm_vs_knn():
    alpha, beta = 0.7, 0.3
    
    # load knn
    
    knn_data = pd.read_csv('results/knn_results.csv')
    knn_data['r (if Minkowski)'] = knn_data['r (if Minkowski)'].fillna(0)

    knn_sick    = knn_data[knn_data['Dataset'].str.contains('sick')].copy()
    knn_balance = knn_data[knn_data['Dataset'].str.contains('balance')].copy()

    # compute m_configs:
    knn_sick    = calculate_m_config(knn_sick, alpha, beta)
    knn_balance = calculate_m_config(knn_balance, alpha, beta)

    best_knn_sick = knn_sick[
        (knn_sick['k'] == 7) & 
        (knn_sick['Feature Weighting Method'] == 'eq_weight') & 
        (knn_sick['Selection Method'] == 'sheppard') &
        (knn_sick['Distance Metric'] == 'hamming')
    ]

    best_knn_balance = knn_balance[
        (knn_balance['k'] == 7) & 
        (knn_balance['Feature Weighting Method'] == 'eq_weight') & 
        (knn_balance['Selection Method'] == 'majority') &
        (knn_balance['Distance Metric'] == 'hamming')
    ]
    
    # load svm
    
    svm_data = pd.read_csv('results/svm_results.csv')

    svm_data['gamma'] = svm_data['gamma'].fillna(0)

    svm_sick    = svm_data[svm_data['Dataset'].str.contains('sick')].copy()
    svm_balance = svm_data[svm_data['Dataset'].str.contains('balance')].copy()

    # compute m_configs:
    svm_sick    = calculate_m_config(svm_sick, alpha, beta)
    svm_balance = calculate_m_config(svm_balance, alpha, beta)

    best_svm_sick = svm_sick[
        (svm_sick['kernel'] == 'rbf') & 
        (svm_sick['C'] == 100.0) & 
        (svm_sick['class_weight'] == 'balanced') &
        (svm_sick['shrinking'] == True) &
        (svm_sick['gamma'] == '1')
    ]

    best_svm_balance = svm_balance[
        (svm_balance['kernel'] == 'linear') & 
        (svm_balance['C'] == 10.0) & 
        (svm_balance['class_weight'] == 'balanced') &
        (svm_balance['shrinking'] == True)
    ]
    
    perform_wilcoxon_test_knn_vs_svm(best_knn_sick, best_svm_sick, "Sick")
    perform_wilcoxon_test_knn_vs_svm(best_knn_balance, best_svm_balance, "Balance Scale")
    


def perform_statistical_analysis_instance_reduction(classifier_dfs, classifier_names, title='Statistical Analysis'):
    """
    Perform statistical analysis (Friedman test and Nemenyi post-hoc with Holm correction)
    on the provided classifier dataframes and plot the Critical Difference Diagram.

    Parameters:
    - classifier_dfs (list of pd.DataFrame): List of classifier dataframes with 'Dataset' and 'M_config' columns.
    - classifier_names (list of str): List of classifier names corresponding to the dataframes.
    - title (str): Title for the CD diagram plot.

    Returns:
    - None
    """
    # Check that there are exactly four dataframes and names
    if len(classifier_dfs) != 4 or len(classifier_names) != 4:
        raise ValueError("Exactly four dataframes and four classifier names must be provided.")

    # Ensure each dataframe has 'Dataset' and 'M_config' columns
    for i, df in enumerate(classifier_dfs):
        if 'Dataset' not in df.columns or 'M_config' not in df.columns:
            raise ValueError(f"DataFrame {i+1} is missing 'Dataset' or 'M_config' columns.")

    # Merge dataframes on 'Dataset'
    merged_df = classifier_dfs[0][['Dataset', 'M_config']].rename(columns={'M_config': classifier_names[0]})
    for i in range(1, len(classifier_dfs)):
        merged_df = merged_df.merge(
            classifier_dfs[i][['Dataset', 'M_config']].rename(columns={'M_config': classifier_names[i]}),
            on='Dataset'
        )

    # Prepare data for statistical tests
    friedman_data = merged_df[classifier_names].values  # Shape: (num_datasets, num_classifiers)

    # Perform Friedman test
    statistic, p_value = friedmanchisquare(*friedman_data.T)
    print(f"Friedman test statistic: {statistic:.4f}, p-value: {p_value:.4f}")

    # If significant, perform Nemenyi post-hoc test with Holm correction
    if p_value < 0.05:
        # Nemenyi post-hoc test
        nemenyi_result = sp.posthoc_nemenyi_friedman(friedman_data)
        p_values = nemenyi_result.values[np.triu_indices_from(nemenyi_result, k=1)]
        print("NEMENYI RESULTS ARE:")
        print(nemenyi_result)

        # Holm's correction
        reject, pvals_corrected, _, _ = multipletests(p_values, method='holm')

        # Build corrected p-values matrix
        corrected_p_values_matrix = np.zeros_like(nemenyi_result)
        corrected_p_values_matrix[np.triu_indices_from(corrected_p_values_matrix, k=1)] = pvals_corrected
        corrected_p_values_matrix += corrected_p_values_matrix.T  # Mirror to lower triangle
        np.fill_diagonal(corrected_p_values_matrix, 0)  # Diagonal should be zero
        nemenyi_result_corrected = pd.DataFrame(corrected_p_values_matrix, index=classifier_names, columns=classifier_names)
        # Calculate average ranks
        ranks_per_observation = np.apply_along_axis(lambda x: rankdata(-x), 1, friedman_data)
        average_ranks = ranks_per_observation.mean(axis=0)
        ranks = pd.Series(average_ranks, index=classifier_names)

        # Plot Critical Difference Diagram
        sp.critical_difference_diagram(ranks, nemenyi_result_corrected)
        plt.title(title)
        filename = title.lower().replace(" ", "_") + ".png"
        plt.savefig("results/"+filename)
        plt.clf()

    else:
        print("No significant differences found among classifiers.")
        
        

def run_statistical_tests_instance_reduction_knn():
    alpha, beta, omega = 0.55, 0.05, 0.4

    # Load and preprocess data for each method
    knn_data = pd.read_csv('results/knn_results.csv')
    knn_data['r (if Minkowski)'] = knn_data['r (if Minkowski)'].fillna(0)

    knn_sick = knn_data[knn_data['Dataset'].str.contains('sick')].copy()
    knn_balance = knn_data[knn_data['Dataset'].str.contains('balance')].copy()

    best_knn_sick = knn_sick[
        (knn_sick['k'] == 7) & 
        (knn_sick['Feature Weighting Method'] == 'eq_weight') & 
        (knn_sick['Selection Method'] == 'sheppard') &
        (knn_sick['Distance Metric'] == 'hamming')
    ]

    best_knn_balance = knn_balance[
        (knn_balance['k'] == 7) & 
        (knn_balance['Feature Weighting Method'] == 'eq_weight') & 
        (knn_balance['Selection Method'] == 'majority') &
        (knn_balance['Distance Metric'] == 'hamming')
    ]

    knn_baseline_sick = best_knn_sick.copy()
    knn_baseline_balance = best_knn_balance.copy()

    knn_baseline_sick['Storage percentage'] = 100.0
    knn_baseline_balance['Storage percentage'] = 100.0

    # Load data for other methods
    knn_drop3 = pd.read_csv('results/knn_results_ir_drop3.csv', sep=';')
    knn_drop3['r (if Minkowski)'] = knn_drop3['r (if Minkowski)'].fillna(0)
    knn_drop3_sick = knn_drop3[knn_drop3['Dataset'].str.contains('sick')].copy()
    knn_drop3_balance = knn_drop3[knn_drop3['Dataset'].str.contains('balance')].copy()
    
    knn_ennth = pd.read_csv('results/knn_results_ir_ennth.csv', sep=';')
    knn_ennth['r (if Minkowski)'] = knn_ennth['r (if Minkowski)'].fillna(0)
    knn_ennth_sick = knn_ennth[knn_ennth['Dataset'].str.contains('sick')].copy()
    knn_ennth_balance = knn_ennth[knn_ennth['Dataset'].str.contains('balance')].copy()
    
    knn_gcnn = pd.read_csv('results/knn_results_ir_gcnn.csv', sep=';')
    knn_gcnn['r (if Minkowski)'] = knn_gcnn['r (if Minkowski)'].fillna(0)
    knn_gcnn_sick = knn_gcnn[knn_gcnn['Dataset'].str.contains('sick')].copy()
    knn_gcnn_balance = knn_gcnn[knn_gcnn['Dataset'].str.contains('balance')].copy()

    # Calculate global min and max for normalization
    all_sick_dfs = [knn_baseline_sick, knn_drop3_sick, knn_ennth_sick, knn_gcnn_sick]
    all_balance_dfs = [knn_baseline_balance, knn_drop3_balance, knn_ennth_balance, knn_gcnn_balance]
    
    min_time_sick, max_time_sick = calculate_global_min_max(all_sick_dfs, 'Time (seconds)')
    min_accu_sick, max_accu_sick = calculate_global_min_max(all_sick_dfs, 'Accuracy')
    min_stor_sick, max_stor_sick = calculate_global_min_max(all_sick_dfs, 'Storage percentage')
    
    min_time_balance, max_time_balance = calculate_global_min_max(all_balance_dfs, 'Time (seconds)')
    min_accu_balance, max_accu_balance = calculate_global_min_max(all_balance_dfs, 'Accuracy')
    min_stor_balance, max_stor_balance = calculate_global_min_max(all_balance_dfs, 'Storage percentage')

    # Calculate M_config for each subset
    knn_baseline_sick = calculate_m_config_storage(knn_baseline_sick, min_time_sick, max_time_sick, min_accu_sick, max_accu_sick, min_stor_sick, max_stor_sick, alpha, beta, omega)
    knn_baseline_balance = calculate_m_config_storage(knn_baseline_balance, min_time_balance, max_time_balance, min_accu_balance, max_accu_balance, min_stor_balance, max_stor_balance, alpha, beta, omega)
    knn_drop3_sick = calculate_m_config_storage(knn_drop3_sick, min_time_sick, max_time_sick, min_accu_sick, max_accu_sick, min_stor_sick, max_stor_sick, alpha, beta, omega)
    knn_drop3_balance = calculate_m_config_storage(knn_drop3_balance, min_time_balance, max_time_balance, min_accu_balance, max_accu_balance, min_stor_balance, max_stor_balance, alpha, beta, omega)
    knn_ennth_sick = calculate_m_config_storage(knn_ennth_sick, min_time_sick, max_time_sick, min_accu_sick, max_accu_sick, min_stor_sick, max_stor_sick, alpha, beta, omega)
    knn_ennth_balance = calculate_m_config_storage(knn_ennth_balance, min_time_balance, max_time_balance, min_accu_balance, max_accu_balance, min_stor_balance, max_stor_balance, alpha, beta, omega)
    knn_gcnn_sick = calculate_m_config_storage(knn_gcnn_sick, min_time_sick, max_time_sick, min_accu_sick, max_accu_sick, min_stor_sick, max_stor_sick, alpha, beta, omega)
    knn_gcnn_balance = calculate_m_config_storage(knn_gcnn_balance, min_time_balance, max_time_balance, min_accu_balance, max_accu_balance, min_stor_balance, max_stor_balance, alpha, beta, omega)

    # Perform statistical analysis
    classifier_names = ['Baseline', 'Drop3', 'ENNTH', 'GCNN']
    
    classifier_dfs_sick = [knn_baseline_sick, knn_drop3_sick, knn_ennth_sick, knn_gcnn_sick]
    perform_statistical_analysis_instance_reduction(classifier_dfs_sick, classifier_names, title='BEST IR METHOD for the KNN for the Sick Dataset - Statistical Analysis')
    
    classifier_dfs_balance = [knn_baseline_balance, knn_drop3_balance, knn_ennth_balance, knn_gcnn_balance]
    perform_statistical_analysis_instance_reduction(classifier_dfs_balance, classifier_names, title='BEST IR METHOD for the Balance Dataset - Statistical Analysis')



def run_statistical_tests_instance_reduction_svm():
    alpha, beta, omega = 0.55, 0.05, 0.4

    # Load and preprocess data for each method
    svm_data = pd.read_csv('results/svm_results.csv')
    svm_data['gamma'] = svm_data['gamma'].fillna(0)
    
    svm_sick = svm_data[svm_data['Dataset'].str.contains('sick')].copy()
    svm_balance = svm_data[svm_data['Dataset'].str.contains('balance')].copy()
    
    best_svm_sick = svm_sick[
        (svm_sick['kernel'] == 'rbf') & 
        (svm_sick['C'] == 100.0) & 
        (svm_sick['class_weight'] == 'balanced') &
        (svm_sick['shrinking'] == True) &
        (svm_sick['gamma'] == '1')
    ]

    best_svm_balance = svm_balance[
        (svm_balance['kernel'] == 'linear') & 
        (svm_balance['C'] == 10.0) & 
        (svm_balance['class_weight'] == 'balanced') &
        (svm_balance['shrinking'] == True)
    ]

    svm_baseline_sick = best_svm_sick.copy()
    svm_baseline_balance = best_svm_balance.copy()

    svm_baseline_sick['Storage percentage'] = 100.0
    svm_baseline_balance['Storage percentage'] = 100.0

    # Load data for other methods
    svm_drop3 = pd.read_csv('results/svm_results_ir_drop3.csv')
    svm_drop3['gamma'] = svm_drop3['gamma'].fillna(0)
    svm_drop3_sick = svm_drop3[svm_drop3['Dataset'].str.contains('sick')].copy()
    svm_drop3_balance = svm_drop3[svm_drop3['Dataset'].str.contains('balance')].copy()
    
    svm_ennth = pd.read_csv('results/svm_results_ir_ennth.csv')
    svm_ennth['gamma'] = svm_ennth['gamma'].fillna(0)
    svm_ennth_sick = svm_ennth[svm_ennth['Dataset'].str.contains('sick')].copy()
    svm_ennth_balance = svm_ennth[svm_ennth['Dataset'].str.contains('balance')].copy()
    
    svm_gcnn = pd.read_csv('results/svm_results_ir_gcnn.csv')
    svm_gcnn['gamma'] = svm_gcnn['gamma'].fillna(0)
    svm_gcnn_sick = svm_gcnn[svm_gcnn['Dataset'].str.contains('sick')].copy()
    svm_gcnn_balance = svm_gcnn[svm_gcnn['Dataset'].str.contains('balance')].copy()

    # Calculate global min and max for normalization
    all_sick_dfs = [svm_baseline_sick, svm_drop3_sick, svm_ennth_sick, svm_gcnn_sick]
    all_balance_dfs = [svm_baseline_balance, svm_drop3_balance, svm_ennth_balance, svm_gcnn_balance]
    
    min_time_sick, max_time_sick = calculate_global_min_max(all_sick_dfs, 'Time (seconds)')
    min_accu_sick, max_accu_sick = calculate_global_min_max(all_sick_dfs, 'Accuracy')
    min_stor_sick, max_stor_sick = calculate_global_min_max(all_sick_dfs, 'Storage percentage')
    
    min_time_balance, max_time_balance = calculate_global_min_max(all_balance_dfs, 'Time (seconds)')
    min_accu_balance, max_accu_balance = calculate_global_min_max(all_balance_dfs, 'Accuracy')
    min_stor_balance, max_stor_balance = calculate_global_min_max(all_balance_dfs, 'Storage percentage')

    # Calculate M_config for each subset
    svm_baseline_sick = calculate_m_config_storage(svm_baseline_sick, min_time_sick, max_time_sick, min_accu_sick, max_accu_sick, min_stor_sick, max_stor_sick, alpha, beta, omega)
    svm_baseline_balance = calculate_m_config_storage(svm_baseline_balance, min_time_balance, max_time_balance, min_accu_balance, max_accu_balance, min_stor_balance, max_stor_balance, alpha, beta, omega)
    svm_drop3_sick = calculate_m_config_storage(svm_drop3_sick, min_time_sick, max_time_sick, min_accu_sick, max_accu_sick, min_stor_sick, max_stor_sick, alpha, beta, omega)
    svm_drop3_balance = calculate_m_config_storage(svm_drop3_balance, min_time_balance, max_time_balance, min_accu_balance, max_accu_balance, min_stor_balance, max_stor_balance, alpha, beta, omega)
    svm_ennth_sick = calculate_m_config_storage(svm_ennth_sick, min_time_sick, max_time_sick, min_accu_sick, max_accu_sick, min_stor_sick, max_stor_sick, alpha, beta, omega)
    svm_ennth_balance = calculate_m_config_storage(svm_ennth_balance, min_time_balance, max_time_balance, min_accu_balance, max_accu_balance, min_stor_balance, max_stor_balance, alpha, beta, omega)
    svm_gcnn_sick = calculate_m_config_storage(svm_gcnn_sick, min_time_sick, max_time_sick, min_accu_sick, max_accu_sick, min_stor_sick, max_stor_sick, alpha, beta, omega)
    svm_gcnn_balance = calculate_m_config_storage(svm_gcnn_balance, min_time_balance, max_time_balance, min_accu_balance, max_accu_balance, min_stor_balance, max_stor_balance, alpha, beta, omega)

    # Perform statistical analysis
    classifier_names = ['Baseline', 'Drop3', 'ENNTH', 'GCNN']
    
    classifier_dfs_sick = [svm_baseline_sick, svm_drop3_sick, svm_ennth_sick, svm_gcnn_sick]
    perform_statistical_analysis_instance_reduction(classifier_dfs_sick, classifier_names, title='Best SVM for the Sick Dataset - Statistical Analysis')
    
    classifier_dfs_balance = [svm_baseline_balance, svm_drop3_balance, svm_ennth_balance, svm_gcnn_balance]
    perform_statistical_analysis_instance_reduction(classifier_dfs_balance, classifier_names, title='Best SVM for the Balance Dataset - Statistical Analysis')
    
    
def run_statistical_tests(test_name):
    
    os.makedirs('results', exist_ok=True)
      
    if test_name == "best_knn":
        run_statistical_tests_best_knn()
    elif test_name == "best_svm":
        run_statistical_tests_best_svm()
    elif test_name == "knn_vs_svm":
        run_statistical_tests_svm_vs_knn()
    elif test_name == "ir_knn":
        run_statistical_tests_instance_reduction_knn()
    elif test_name == "ir_svm":
        run_statistical_tests_instance_reduction_svm()