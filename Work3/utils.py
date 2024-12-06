import csv
import numpy as np
from sklearn.metrics import cluster

def purity_score(y_true, y_pred):
    """
    Calculate the purity score for the given clustering.

    Args:
        y_true (array-like): Ground truth class labels.
        y_pred (array-like): Cluster labels assigned by the clustering algorithm.
    Returns:
        purity (float): Purity score ranging from 0 to 1.
    """
    # Compute confusion matrix (contingency matrix)
    contingency_matrix = cluster.contingency_matrix(y_true, y_pred)
    
    # Sum the maximum counts for each cluster
    max_counts = np.amax(contingency_matrix, axis=0)
    return np.sum(max_counts) / np.sum(contingency_matrix) 



def save_kmeans_results(data_row, csv_file):
    """
    Saves per-run results to a CSV file.

    Args:
        data_row (dict): Data to write to the CSV.
        csv_file (str): Filename of the CSV file.
    """
    # Only include specified parameters and results in the CSV columns
    csv_columns = [
        'Dataset',
        'k',
        'Distance',
        'Seed',
        'ARI',
        'Silhouette',
        'DBI',
        'Time (s)'
    ]

    try:
        with open(csv_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            # Write header if file is empty
            if csvfile.tell() == 0:
                writer.writeheader()
            writer.writerow(data_row)
    except IOError:
        print("I/O error when writing KMeans results.")
    

def save_xmeans_results(data_row, csv_file):
    """
    Saves per-run results to a CSV file.

    Args:
        data_row (dict): Data to write to the CSV.
        csv_file (str): Filename of the CSV file.
    """
    # Only include specified parameters and results in the CSV columns
    csv_columns = [
        'Dataset',
        'k_max',
        'best_k',
        'Seed',
        'ARI',
        'Silhouette',
        'Purity',
        'DBI',
        'Time (s)'
    ]

    try:
        with open(csv_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            # Write header if file is empty
            if csvfile.tell() == 0:
                writer.writeheader()
            writer.writerow(data_row)
    except IOError:
        print("I/O error when writing KMeans results.")

def save_optics_results(data_row, csv_file):
    """
    Saves per-run results to a CSV file.

    Args:
        data_row (dict): Data to write to the CSV.
        csv_file (str): Filename of the CSV file.
    """
    # Only include specified parameters and results in the CSV columns
    csv_columns = [
        'Dataset',
        'Metric',
        'Algorithm',
        'Min Samples',
        'Silhouette',
        'ARI',
        'DBI',
        'Purity',
        'Num Clusters',
        'Time (s)'
    ]

    try:
        with open(csv_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            # Write header if file is empty
            if csvfile.tell() == 0:
                writer.writeheader()
            writer.writerow(data_row)
    except IOError:
        print("I/O error when writing OPTICS results.")


def save_spectral_results(data_row, csv_file):
    """
    Saves per-run results to a CSV file.

    Args:
        data_row (dict): Data to write to the CSV.
        csv_file (str): Filename of the CSV file.
    """
    # Only include specified parameters and results in the CSV columns
    csv_columns = [
        'Dataset',
        'N Neighbors',
        'Affinity',
        'Eigen Solver',
        'Assign Labels',
        'N Clusters',
        'Seed',
        'ARI',
        'DBI',
        'Silhouette',
        'Purity',
        'Time (s)'
    ]

    try:
        with open(csv_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            # Write header if file is empty
            if csvfile.tell() == 0:
                writer.writeheader()
            writer.writerow(data_row)
    except IOError:
        print("I/O error when writing Spectral results.")
