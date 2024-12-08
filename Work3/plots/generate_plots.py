from tqdm import tqdm
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

import warnings
warnings.filterwarnings("ignore")

from preprocessing import DataLoader, DataProcessor

# Import clustering classes and functions for different experiments
from fuzzy import GSFuzzyCMeans
from optics.optics import opticsAlgorithm  # Function-based
from spectral.spectral import spectralAlgorithm  # Function-based
from kmeans.kmeans import KMeans
from kmeans.xmeans import XMeans
from kmeans.global_kmeans import fast_global_k_means 

# Mapping from experiment name to clustering class (class-based algorithms)
CLUSTERING_CLASSES = {
    'fuzzy': GSFuzzyCMeans,
    'kmeans': KMeans,
    'xmeans': XMeans,
    'globalkmeans': fast_global_k_means,
}

# Mapping from experiment name to clustering function (function-based algorithms)
CLUSTERING_FUNCTIONS = {
    'optics': opticsAlgorithm,
    'spectral': spectralAlgorithm,
}

# Define metric aliases to handle different naming conventions
METRIC_ALIASES = {
    'adjusted rand index': ['adjusted rand index', 'ari'],
    'davies-bouldin index': ['davies-bouldin index', 'dbi'],
    'silhouette score': ['silhouette score', 'silhouette'],
    'purity score': ['purity score', 'purity'],
}

# Define the required parameters for each algorithm (all lowercase)
ALGORITHM_PARAMS = {
    'fuzzy': ['k', 'm', 'xi', 'seed'],
    'optics': ['metric', 'algorithm', 'min samples'],  # No 'seed'
    'spectral': ['n neighbors', 'affinity', 'eigen solver', 'assign labels', 'n clusters', 'seed'],
    'kmeans': ['k', 'distance', 'seed'],
    'xmeans': ['k_max', 'best_k', 'seed'],
    'globalkmeans': ['k', 'distance'],  # No 'seed'
}

# Define CSV parsing configurations per algorithm
CSV_PARSERS = {
    'fuzzy': {
        'required_columns': [
            'k', 'm', 'xi',
            'adjusted rand index', 'adjusted rand index seed',
            'davies-bouldin index', 'davies-bouldin index seed',
            'silhouette score', 'silhouette score seed',
            'purity score', 'purity score seed',
            'time (s)', 'time (s) seed',
            'iterations', 'iterations seed',
            'dataset'
        ],
        'param_mapping': {'k': 'k', 'm': 'm', 'xi': 'xi', 'seed': 'adjusted rand index seed'},  # Seed is specific per metric
        'metric_columns': ['adjusted rand index', 'davies-bouldin index', 'silhouette score', 'purity score'],
        'has_seed': True  # Seeds are present per metric
    },
    'globalkmeans': {
        'required_columns': [
            'k', 'distance',
            'adjusted rand index', 'davies-bouldin index',
            'silhouette score', 'purity score',
            'time (s)', 'iterations', 'error',
            'dataset'
        ],
        'param_mapping': {'k': 'k', 'distance': 'distance'},
        'metric_columns': ['adjusted rand index', 'davies-bouldin index', 'silhouette score', 'purity score'],
        'has_seed': False  # Deterministic
    },
    'kmeans': {
        'required_columns': [
            'k', 'distance', 'seed',
            'adjusted rand index', 'davies-bouldin index',
            'silhouette score', 'purity score',
            'time (s)', 'iterations', 'error',
            'dataset'
        ],
        'param_mapping': {'k': 'k', 'distance': 'distance', 'seed': 'seed'},
        'metric_columns': ['adjusted rand index', 'davies-bouldin index', 'silhouette score', 'purity score'],
        'has_seed': True
    },
    'optics': {
        'required_columns': [
            'dataset', 'metric', 'algorithm', 'min samples',
            'silhouette', 'ari', 'dbi', 'purity',
            'num clusters', 'time (s)'
        ],
        'param_mapping': {'metric': 'metric', 'algorithm': 'algorithm', 'min samples': 'min samples'},
        'metric_columns': ['ari', 'dbi', 'silhouette', 'purity'],
        'has_seed': False
    },
    'spectral': {
        'required_columns': [
            'dataset', 'n neighbors', 'affinity', 'eigen solver',
            'assign labels', 'n clusters', 'seed',
            'ari', 'dbi', 'silhouette', 'purity',
            'time (s)'
        ],
        'param_mapping': {
            'n neighbors': 'n neighbors',
            'affinity': 'affinity',
            'eigen solver': 'eigen solver',
            'assign labels': 'assign labels',
            'n clusters': 'n clusters',
            'seed': 'seed'
        },
        'metric_columns': ['ari', 'dbi', 'silhouette', 'purity'],
        'has_seed': True
    },
    'xmeans': {
        'required_columns': [
            'dataset', 'k_max', 'best_k', 'seed',
            'ari', 'silhouette', 'purity', 'dbi',
            'time (s)'
        ],
        'param_mapping': {'k_max': 'k_max', 'best_k': 'best_k', 'seed': 'seed'},
        'metric_columns': ['ari', 'dbi', 'silhouette', 'purity'],
        'has_seed': True
    }
}

def get_all_aliases(metric_standard_name):
    metric_standard_name = metric_standard_name.lower()
    for standard_name, aliases in METRIC_ALIASES.items():
        if metric_standard_name == standard_name or metric_standard_name in aliases:
            # Return all aliases for the matched standard name
            return [standard_name] + aliases
    # Return the metric name itself as a fallback
    return [metric_standard_name]

def ensure_directory(path):
    """
    Ensures that the specified directory exists. If not, it creates the directory.
    """
    os.makedirs(path, exist_ok=True)

def read_results(experiment):
    """
    Reads the consolidated CSV results file for the given experiment.

    Parameters:
        experiment (str): The name of the experiment.

    Returns:
        pd.DataFrame: The DataFrame containing the results.
    """
    file_name = f"{experiment}_results.csv"
    file_path = os.path.join("results", file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The results file '{file_path}' does not exist.")
    df = pd.read_csv(file_path)

    # Standardize column names to lowercase to handle 'Dataset' vs 'dataset'
    df.columns = [col.strip().lower() for col in df.columns]

    # Verify required columns based on the algorithm's CSV parser
    parser_config = CSV_PARSERS.get(experiment.lower())
    if not parser_config:
        raise ValueError(f"No CSV parser configuration found for experiment '{experiment}'.")

    missing_columns = [col for col in parser_config['required_columns'] if col not in df.columns]
    if missing_columns:
        raise ValueError(f"The CSV file for '{experiment}' is missing required columns: {missing_columns}")

    return df

def find_metric_column(df, metric_standard_name):
    """
    Finds the actual metric column name in the DataFrame based on possible aliases.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the results.
        metric_standard_name (str): The standardized metric name.

    Returns:
        str: The actual column name in the DataFrame.

    Raises:
        ValueError: If none of the aliases are found in the DataFrame.
    """
    aliases = get_all_aliases(metric_standard_name)

    for alias in aliases:
        if alias.lower() in df.columns:
            return alias.lower()
    raise ValueError(f"None of the aliases for metric '{metric_standard_name}' were found in the DataFrame.")

def get_best_configs(df, metrics, algorithm):
    """
    Extracts the best configurations for each dataset and metric based on optimization direction.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the results.
        metrics (dict): A dictionary of metrics and their optimization directions.
        algorithm (str): The name of the clustering algorithm.

    Returns:
        dict: A nested dictionary with best configurations.
              Structure: {dataset: {metric_standard: {'params': {...}, 'metric_value': value}}}
    """
    best_configs = {}
    datasets = df['dataset'].unique()
    param_cols = ALGORITHM_PARAMS.get(algorithm.lower(), [])
    parser_config = CSV_PARSERS.get(algorithm.lower(), {})

    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset].copy().reset_index(drop=True)
        best_configs[dataset] = {}

        for metric_standard, direction in metrics.items():
            metric_lower = metric_standard.lower()
            try:
                actual_metric_col = find_metric_column(dataset_df, metric_standard)
            except ValueError as ve:
                raise ValueError(f"For dataset '{dataset}', {ve}")

            if algorithm.lower() == 'fuzzy':
                # For fuzzy, each row contains the best configuration per metric
                # Select the row where the metric is maximized or minimized
                if direction == "max":
                    best_row = dataset_df.loc[dataset_df[actual_metric_col].idxmax()]
                elif direction == "min":
                    best_row = dataset_df.loc[dataset_df[actual_metric_col].idxmin()]
                else:
                    raise ValueError(f"Unknown optimization direction '{direction}' for metric '{metric_standard}'.")

                # Extract the best parameters and seed specific to the metric
                config = {}
                for param in param_cols:
                    if param == 'seed':
                        # Each metric has its own seed column
                        seed_column = f"{actual_metric_col} seed"
                        if seed_column not in dataset_df.columns:
                            raise ValueError(f"Missing '{seed_column}' for dataset '{dataset}' and metric '{metric_standard}'.")
                        seed_value = best_row.get(seed_column, None)
                        if pd.isnull(seed_value):
                            raise ValueError(f"Missing '{seed_column}' for dataset '{dataset}' and metric '{metric_standard}'.")
                        config['seed'] = int(seed_value)
                    else:
                        param_value = best_row.get(param, None)
                        if pd.isnull(param_value):
                            raise ValueError(f"Missing parameter '{param}' for dataset '{dataset}' and metric '{metric_standard}'.")
                        # Convert to int if possible
                        if isinstance(param_value, float) and param_value.is_integer():
                            param_value = int(param_value)
                        config[param] = param_value

                # Also include metric value
                metric_value = best_row.get(actual_metric_col, None)

                # Remove any NaN parameters (shouldn't be any due to above checks)
                config = {k: v for k, v in config.items() if pd.notnull(v)}

                # Store configuration and metric value
                best_configs[dataset][metric_standard] = {
                    'params': config,
                    'metric_value': metric_value
                }

                print(f"Best configuration for '{dataset}' based on '{metric_standard}': {config} with {metric_standard} = {metric_value}")

            else:
                # For other algorithms
                if parser_config.get('has_seed'):
                    # Algorithms with seed: kmeans, spectral, xmeans
                    # Multiple rows per configuration (different seeds)
                    # Select the row with the best metric based on direction
                    if direction == "max":
                        best_row = dataset_df.loc[dataset_df[actual_metric_col].idxmax()]
                    elif direction == "min":
                        best_row = dataset_df.loc[dataset_df[actual_metric_col].idxmin()]
                    else:
                        raise ValueError(f"Unknown optimization direction '{direction}' for metric '{metric_standard}'.")

                    # Extract parameters based on param_mapping
                    config = {}
                    for param_key, param_col in parser_config['param_mapping'].items():
                        param_value = best_row.get(param_col, None)
                        if pd.isnull(param_value):
                            raise ValueError(f"Missing parameter '{param_col}' for dataset '{dataset}' and metric '{metric_standard}'.")
                        # Convert to int if possible
                        if isinstance(param_value, float) and param_value.is_integer():
                            param_value = int(param_value)
                        config[param_key] = param_value

                    # Include metric value
                    metric_value = best_row.get(actual_metric_col, None)

                    # Remove any NaN parameters
                    config = {k: v for k, v in config.items() if pd.notnull(v)}

                    # Store configuration and metric value
                    best_configs[dataset][metric_standard] = {
                        'params': config,
                        'metric_value': metric_value
                    }

                    print(f"Best configuration for '{dataset}' based on '{metric_standard}': {config} with {metric_standard} = {metric_value}")

                else:
                    # Algorithms without seed: globalkmeans, optics
                    # Each row is a unique configuration
                    # Select the row with the best metric based on direction
                    if direction == "max":
                        best_row = dataset_df.loc[dataset_df[actual_metric_col].idxmax()]
                    elif direction == "min":
                        best_row = dataset_df.loc[dataset_df[actual_metric_col].idxmin()]
                    else:
                        raise ValueError(f"Unknown optimization direction '{direction}' for metric '{metric_standard}'.")

                    # Extract parameters based on param_mapping
                    config = {}
                    for param_key, param_col in parser_config['param_mapping'].items():
                        param_value = best_row.get(param_col, None)
                        if pd.isnull(param_value):
                            raise ValueError(f"Missing parameter '{param_col}' for dataset '{dataset}' and metric '{metric_standard}'.")
                        # Convert to int if possible
                        if isinstance(param_value, float) and param_value.is_integer():
                            param_value = int(param_value)
                        config[param_key] = param_value

                    # Include metric value
                    metric_value = best_row.get(actual_metric_col, None)

                    # Remove any NaN parameters
                    config = {k: v for k, v in config.items() if pd.notnull(v)}

                    # Store configuration and metric value
                    best_configs[dataset][metric_standard] = {
                        'params': config,
                        'metric_value': metric_value
                    }

                    print(f"Best configuration for '{dataset}' based on '{metric_standard}': {config} with {metric_standard} = {metric_value}")

    return best_configs

def load_and_preprocess(dataset):
    """
    Loads and preprocesses the dataset.

    Parameters:
        dataset (str): The name of the dataset.

    Returns:
        tuple: Preprocessed data (X) and true labels (y_true).
    """
    data_loader = DataLoader()
    data_processor = DataProcessor()

    data, labels = data_loader.load_arff_data(dataset)  # Assuming datasets are in ARFF format
    X = data_processor.preprocess_dataset(data)
    y_true = labels  # Ground truth labels
    return X, y_true

def perform_clustering(clustering_class_or_function, X, config, algorithm):
    """
    Performs clustering using the specified clustering class or function and configuration.

    Parameters:
        clustering_class_or_function (class/function): The clustering algorithm class or function.
        X (np.ndarray): The preprocessed data.
        config (dict): The configuration parameters for clustering.
        algorithm (str): The name of the clustering algorithm.

    Returns:
        np.ndarray: The predicted cluster labels.
    """
    if algorithm.lower() in CLUSTERING_FUNCTIONS:
        # Function-based algorithms: optics, spectral
        if algorithm.lower() == 'optics':
            # Extract parameters specific to opticsAlgorithm
            metric = config.get('metric')
            alg = config.get('algorithm')
            min_samples = config.get('min samples')
            n_jobs = -1  # Default value; modify if necessary

            # Call the opticsAlgorithm function
            y_pred = clustering_class_or_function(
                X,
                metric=metric,
                algorithm=alg,
                min_samples=int(min_samples),
                n_jobs=n_jobs
            )
            return y_pred

        elif algorithm.lower() == 'spectral':
            # Extract parameters specific to spectralAlgorithm
            n_neighbors = config.get('n neighbors')
            affinity = config.get('affinity')
            eigen_solver = config.get('eigen solver')
            assign_labels = config.get('assign labels')
            n_clusters = config.get('n clusters')
            seed = config.get('seed')
            n_jobs = -1  # Default value; modify if necessary

            # Call the spectralAlgorithm function
            y_pred = clustering_class_or_function(
                X,
                eigen_solver=eigen_solver,
                affinity=affinity,
                n_neighbors=int(n_neighbors),
                assign_labels=assign_labels,
                n_clusters=int(n_clusters),
                seed=int(seed),
                n_jobs=n_jobs
            )
            return y_pred

        else:
            raise ValueError(f"Unknown function-based algorithm: {algorithm}")

    else:
        # Class-based algorithms: fuzzy, kmeans, xmeans, globalkmeans
        if algorithm.lower() == 'fuzzy':
            clusterer = clustering_class_or_function(
                n_clusters=int(config['k']),
                m=config['m'],
                xi=config['xi'],
                max_iter=100,
                random_state=int(config['seed'])
            )
        elif algorithm.lower() == 'kmeans':
            clusterer = clustering_class_or_function(
                k=int(config['k']),
                distance=config['distance'],
                seed=int(config['seed'])
            )
        elif algorithm.lower() == 'xmeans':
            clusterer = clustering_class_or_function(
                k_max=int(config['k_max']),
                max_iters=100,
                seed=int(config['seed'])
            )
        elif algorithm.lower() == 'globalkmeans':
            clusterer = clustering_class_or_function(
                k=int(config['k']),
                distance=config['distance']
                # Add other parameters if needed
            )
        else:
            raise ValueError(f"Unknown class-based algorithm: {algorithm}")

        # Determine the appropriate method to obtain cluster labels
        if hasattr(clusterer, 'fit_predict'):
            # Use fit_predict if available
            y_pred = clusterer.fit_predict(X)
        else:
            # Fallback to fit and then retrieve labels
            clusterer.fit(X)
            if hasattr(clusterer, 'labels_'):
                y_pred = clusterer.labels_
            elif hasattr(clusterer, 'predict'):
                y_pred = clusterer.predict(X)
            else:
                raise AttributeError(f"The '{algorithm}' class does not have 'labels_', 'predict', or 'fit_predict' methods.")

        return y_pred

def plot_pca(dataset_labels, metrics, experiment, save_dir):
    """
    Plots PCA scatter plots in a single row with 2x2 plots per dataset.

    Parameters:
        dataset_labels (dict): Nested dictionary containing data and labels.
        metrics (dict): Dictionary of metrics and their optimization directions.
        experiment (str): The name of the experiment.
        save_dir (str): Directory to save PCA plots.
    """
    sns.set_style('whitegrid')
    datasets = list(dataset_labels.keys())
    num_datasets = len(datasets)
    num_metrics = len(metrics)

    # Total number of subplots: 2 rows x (len(datasets) * 2) columns for 2x2 per dataset
    fig, axes = plt.subplots(2, num_datasets * 2, figsize=(num_datasets * 10, 10), sharex=False, sharey=False)
    fig.suptitle('PCA Clustering Results for All Datasets (2x2 Per Dataset)', fontsize=18, y=1.02)

    for dataset_idx, dataset in enumerate(datasets):
        # Subplots indices for the dataset
        dataset_axes = axes[:, dataset_idx * 2:(dataset_idx + 1) * 2].flat

        # Add a title centered above the 2x2 grid for the current dataset
        x_center = (dataset_idx * 2 + 1) / (num_datasets * 2)
        fig.text(x_center, 0.96, f'{dataset.capitalize()} Dataset', ha='center', va='center', fontsize=16, weight='bold')

        for ax, (metric, _) in zip(dataset_axes, metrics.items()):
            # Retrieve data and labels for the specific metric
            data = dataset_labels[dataset][metric]['data']
            labels = dataset_labels[dataset][metric]['labels']

            # Apply PCA
            pca = PCA(n_components=2)
            data_reduced = pca.fit_transform(data)
            variance_ratio = pca.explained_variance_ratio_ * 100  # Variance percentages

            # Create a DataFrame for plotting
            plot_df = pd.DataFrame(data_reduced, columns=['Principal Component 1', 'Principal Component 2'])
            plot_df['Cluster'] = labels
            # Ensure the legend is ordered numerically
            plot_df['Cluster'] = pd.Categorical(plot_df['Cluster'], categories=sorted(plot_df['Cluster'].unique()), ordered=True)

            # Plot without legend
            sns.scatterplot(
                data=plot_df, 
                x='Principal Component 1', 
                y='Principal Component 2',
                hue='Cluster', 
                palette='tab10', 
                ax=ax, 
                s=50, 
                alpha=0.8, 
                legend=False  # Disable legend
            )
            ax.set_title(f'{metric} Metric')
            ax.set_xlabel(f'Principal Component 1 ({variance_ratio[0]:.2f}% Variance)')
            ax.set_ylabel(f'Principal Component 2 ({variance_ratio[1]:.2f}% Variance)')

    # Adjust the layout to leave room for titles
    plt.tight_layout(rect=[0, 0, 1, 0.93])  # Adjust top margin to leave space for the suptitle
    pca_plot_path = os.path.join(save_dir, f"{experiment}_pca.png")  # Updated filename
    plt.savefig(pca_plot_path, bbox_inches='tight')
    plt.close()
    print(f"PCA plots saved to '{pca_plot_path}'")

def plot_confusion_matrices(dataset_labels, metrics, experiment, save_dir):
    """
    Plots diagonalized confusion matrices in a single row with 2x2 plots per dataset.

    Parameters:
        dataset_labels (dict): Nested dictionary containing data and labels.
        metrics (dict): Dictionary of metrics and their optimization directions.
        experiment (str): The name of the experiment.
        save_dir (str): Directory to save confusion matrix plots.
    """
    sns.set_style('whitegrid')
    datasets = list(dataset_labels.keys())
    num_datasets = len(datasets)
    num_metrics = len(metrics)

    # Total number of subplots: 2 rows x (len(datasets) * 2) columns for 2x2 per dataset
    fig, axes = plt.subplots(2, num_datasets * 2, figsize=(num_datasets * 10, 10), sharex=False, sharey=False)
    fig.suptitle('Diagonalized Confusion Matrices for All Datasets (2x2 Per Dataset)', fontsize=18, y=1.02)

    for dataset_idx, dataset in enumerate(datasets):
        # Subplots indices for the dataset
        dataset_axes = axes[:, dataset_idx * 2:(dataset_idx + 1) * 2].flat

        # Add a title centered above the 2x2 grid for the current dataset
        x_center = (dataset_idx * 2 + 1) / (num_datasets * 2)
        fig.text(x_center, 0.96, f'{dataset.capitalize()} Dataset', ha='center', va='center', fontsize=16, weight='bold')

        for ax, (metric, _) in zip(dataset_axes, metrics.items()):
            # Retrieve true labels and predicted labels
            y_true = dataset_labels[dataset][metric]['true_labels']
            y_pred = dataset_labels[dataset][metric]['labels']

            # Map true labels and predicted labels to consecutive integers
            y_true_unique = np.unique(y_true)
            y_pred_unique = np.unique(y_pred)

            y_true_mapping = {label: idx for idx, label in enumerate(y_true_unique)}
            y_pred_mapping = {label: idx for idx, label in enumerate(y_pred_unique)}

            y_true_mapped = np.array([y_true_mapping[label] for label in y_true])
            y_pred_mapped = np.array([y_pred_mapping[label] for label in y_pred])

            # Compute confusion matrix
            cm = confusion_matrix(y_true_mapped, y_pred_mapped)

            # Apply the Hungarian algorithm to find the best assignment
            cost_matrix = cm.max() - cm  # Convert to cost matrix for maximization
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Reorder the confusion matrix
            cm_reordered = cm[:, col_ind]

            # Remove rows and columns that sum to 0
            non_zero_rows = np.any(cm_reordered > 0, axis=1)
            non_zero_cols = np.any(cm_reordered > 0, axis=0)
            cm_reordered = cm_reordered[non_zero_rows, :][:, non_zero_cols]

            # Normalize confusion matrix to [0,1]
            cm_row_sums = cm_reordered.sum(axis=1, keepdims=True)
            cm_row_sums[cm_row_sums == 0] = 1  # Avoid division by zero
            cm_normalized = cm_reordered.astype('float') / cm_row_sums

            # Generate labels for heatmap axes
            true_labels_names = [str(y_true_unique[i]) for i in range(len(y_true_unique)) if non_zero_rows[i]]
            pred_labels_names = [str(y_pred_unique[col_ind[j]]) for j in range(len(col_ind)) if non_zero_cols[j]]

            # Plot heatmap
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                        xticklabels=pred_labels_names, yticklabels=true_labels_names, cbar=False)
            ax.set_title(f'{metric} Metric')
            ax.set_xlabel('Predicted Cluster')
            ax.set_ylabel('True Label')

    # Adjust the layout to leave room for titles
    plt.tight_layout(rect=[0, 0, 1, 0.93])  # Adjust top margin to leave space for the suptitle
    cm_plot_path = os.path.join(save_dir, f"{experiment}_cm.png")  # Updated filename
    plt.savefig(cm_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix plots saved to '{cm_plot_path}'")

def generate_plots(experiment):
    """
    Generates PCA scatter plots and diagonalized confusion matrices for the specified experiment.

    Parameters:
        experiment (str): The name of the experiment (e.g., 'fuzzy').
    """
    # Define metrics and their optimization directions
    metrics = {
        "Adjusted Rand Index": "max",      # Maximized
        "Davies-Bouldin Index": "min",    # Minimized
        "Silhouette Score": "max",        # Maximized
        "Purity Score": "max"             # Maximized
    }

    # Ensure the experiment is supported
    if experiment.lower() not in CLUSTERING_CLASSES and experiment.lower() not in CLUSTERING_FUNCTIONS:
        raise ValueError(f"Experiment '{experiment}' is not supported. Please add it to the CLUSTERING_CLASSES or CLUSTERING_FUNCTIONS dictionary.")

    # Determine if the algorithm is class-based or function-based
    if experiment.lower() in CLUSTERING_CLASSES:
        clustering_class_or_function = CLUSTERING_CLASSES[experiment.lower()]
    else:
        clustering_class_or_function = CLUSTERING_FUNCTIONS[experiment.lower()]

    # Define directories for saving plots
    pca_save_dir = os.path.join("plots", "plots_pca")
    cm_save_dir = os.path.join("plots", "plots_cm")
    ensure_directory(pca_save_dir)
    ensure_directory(cm_save_dir)

    # Read the results CSV
    df = read_results(experiment)

    # Extract best configurations
    best_configs = get_best_configs(df, metrics, experiment)

    # Dictionary to store datasets and labels for each metric
    dataset_labels = {}
    datasets = ['satimage', 'splice', 'vowel']

    print("Starting clustering computation with optimal parameters...")

    for dataset in tqdm(datasets, desc="Datasets"):
        X, y_true = load_and_preprocess(dataset)
        dataset_labels[dataset] = {}

        for metric in tqdm(metrics.keys(), desc=f"Metrics for {dataset}", leave=False):
            config_entry = best_configs[dataset][metric]
            config = config_entry['params']
            
            y_pred = perform_clustering(clustering_class_or_function, X, config, experiment)
            dataset_labels[dataset][metric] = {'data': X, 'labels': y_pred, 'true_labels': y_true}

    # Plot PCA scatter plots
    plot_pca(dataset_labels, metrics, experiment, pca_save_dir)

    # Plot confusion matrices
    plot_confusion_matrices(dataset_labels, metrics, experiment, cm_save_dir)

    print("All plots have been generated and saved successfully.")
