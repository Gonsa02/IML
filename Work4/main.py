
from PCA.PCA import main_PCA, main_incremental_PCA
from exercise_4 import main_our_pca_clustering
from exercise_5 import main_sklearn_kernelpca_clustering
from clustering_plots.clustering_plots import main_umap_and_pca

import argparse


def run_all_experiments():
    """
    Runs all clustering experiments.
    """
    print("Running all experiments...")
    try:
        main_PCA()
        print("PCA Experiments Completed.\n")
    except Exception as e:
        print(f"Error running PCA experiments: {e}\n")

    try:
        main_incremental_PCA()
        print("Incremental PCA Experiments Completed.\n")
    except Exception as e:
        print(f"Error running Incremental PCA experiment: {e}\n")
    
    try:
        main_our_pca_clustering()
        print("Clustering with dimensionality reduction (Our PCA) Completed.\n")
    except Exception as e:
        print(f"Error running clustering with dimensionality reduction (Our PCA) experiment: {e}\n")
    
    try:
        main_sklearn_kernelpca_clustering()
        print("Clustering with dimensionality reduction (sklearn kernelPCA) Completed.\n")
    except Exception as e:
        print(f"Error running clustering with dimensionality reduction (sklearn kernelPCA): {e}\n")
    
    try:
        main_umap_and_pca()
        print("Clustering with different dimensionality reduction algorithms (our PCA and UMAP) Completed.\n")
    except Exception as e:
        print(f"Error running clustering with different dimensionality reduction algorithms (our PCA and UMAP) experiment: {e}\n")
    
    print("All experiments have been executed.\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run experiments')

    # Experiment option
    parser.add_argument(
        '--experiment', 
        nargs='+',
        choices=['all', 'pca', 'incremental_pca', 'clustering_our_pca', 'clustering_sklearn_pca', 'clustering_pca_and_umap'],
        help="Type of experiment to run: 'pca', 'incremental_pca','clustering_our_pca', 'clustering_sklearn_pca', 'clustering_pca_and_umap', or 'all'"
    )

    args = parser.parse_args()

    # Ensure at least one action is specified
    if not args.experiment:
        parser.error(
            "You must specify at least one action: --experiment.")

    if args.experiment:
        if 'all' in args.experiment:
            run_all_experiments()
        else:
            for experiment in args.experiment:
                print(f"Running {experiment} experiment...")
                if experiment == 'pca':
                    main_PCA()
                elif experiment == 'incremental_pca':
                    main_incremental_PCA()
                elif experiment == 'clustering_our_pca':
                    main_our_pca_clustering()
                elif experiment == 'clustering_sklearn_kernelpca':
                    main_sklearn_kernelpca_clustering()
                elif experiment == 'clustering_pca_and_umap':
                    main_umap_and_pca()
                print(f"{experiment} experiment completed.\n")

if __name__ == '__main__':
    main()