import csv

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
        'Eigen Solver',
        'Affinity',
        'Assign Labels',
        'N Neighbors',
        'N Clusters',
        'Silhouette',
        'ARI',
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
        print("I/O error when writing Spectral results.")

