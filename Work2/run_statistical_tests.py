import pandas as pd
import scipy.stats as stats

def run_statistical_tests(experiment):
    if experiment == 'knn':
        results_filename = 'results/knn_results.csv'
        results_ir_filename = 'results/knn_results_ir.csv'
    elif experiment == 'svm':
        results_filename = 'results/svm_results.csv'
        results_ir_filename = 'results/svm_results_ir.csv'
    else:
        print("Statistical tests can be run for 'knn' and 'svm' experiments.")
        return

    # Load results
    results = pd.read_csv(results_filename)
    results_ir = pd.read_csv(results_ir_filename)

    # Merge results on Dataset and other relevant parameters
    merge_columns = ['Dataset']
    if 'k' in results.columns:
        merge_columns.extend(['k', 'Feature Weighting Method', 'Selection Method', 'Distance Metric', 'r (if Minkowski)'])
    elif 'Kernel' in results.columns:
        merge_columns.append('Kernel')

    merged_results = pd.merge(results, results_ir, on=merge_columns, suffixes=('_normal', '_ir'))

    # Perform paired t-test on the accuracies
    t_stat, p_value = stats.ttest_rel(merged_results['Accuracy_normal'], merged_results['Accuracy_ir'])

    print(f"Statistical test results for {experiment}:")
    print(f"T-test statistic: {t_stat:.4f}, p-value: {p_value:.4f}")

    if p_value < 0.05:
        print("There is a significant difference between the normal and instance-reduced datasets.")
        if merged_results['Accuracy_normal'].mean() > merged_results['Accuracy_ir'].mean():
            print("The normal dataset performs better.")
        else:
            print("The instance-reduced dataset performs better.")
    else:
        print("No significant difference between the normal and instance-reduced datasets.")
