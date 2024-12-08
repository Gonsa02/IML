# main.py

import argparse
import sys
import os

# Import the generate_plots function
from plots.generate_plots import generate_plots

# Import run functions for each algorithm
from optics.run_optics import run_optics
from spectral.run_spectral import run_spectral
from kmeans.run_kmeans import run_kmeans
from kmeans.run_gkmeans import run_gkmeans
from kmeans.run_xmeans import run_xmeans
from kmeans.run_gkmeans import run_gkmeans
from fuzzy.run_fuzzy import run_fuzzy

def ensure_directories():
    """
    Ensures that the necessary directories for results and plots exist.
    """
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots/plots_pca", exist_ok=True)
    os.makedirs("plots/plots_cm", exist_ok=True)

def run_all_experiments():
    """
    Runs all clustering experiments.
    """
    print("Running all experiments...")
    try:
        run_optics()
        print("Optics experiment completed.\n")
    except Exception as e:
        print(f"Error running Optics experiment: {e}\n")
    
    try:
        run_spectral()
        print("Spectral experiment completed.\n")
    except Exception as e:
        print(f"Error running Spectral experiment: {e}\n")
    
    try:
        run_kmeans()
        print("KMeans experiment completed.\n")
    except Exception as e:
        print(f"Error running KMeans experiment: {e}\n")
    
    try:
        run_xmeans()
        print("XMeans experiment completed.\n")
    except Exception as e:
        print(f"Error running XMeans experiment: {e}\n")
    
    try:
        run_fuzzy()
        print("Fuzzy experiment completed.\n")
    except Exception as e:
        print(f"Error running Fuzzy experiment: {e}\n")
    
    print("All experiments have been executed.\n")

def run_all_plots():
    """
    Generates plots for all clustering algorithms.
    """
    print("Generating plots for all algorithms...")
    algorithms = ['optics', 'spectral', 'kmeans', 'xmeans', 'fuzzy']
    for algo in algorithms:
        print(f"Generating plots for {algo}...")
        try:
            generate_plots(algo)
            print(f"Plots for {algo} generated successfully.\n")
        except Exception as e:
            print(f"Error generating plots for {algo}: {e}\n")
    print("All plots have been generated.\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run clustering experiments and generate plots')

    # Experiment option
    parser.add_argument(
        '--experiment', 
        nargs='+',
        choices=['all', 'optics', 'spectral', 'kmeans', 'xmeans', 'globalkmeans', 'fuzzy'],
        help="Type of experiment to run: 'optics', 'spectral', 'kmeans', 'xmeans', 'globalkmeans', 'fuzzy', or 'all'"
    )

    # Generate plots option
    parser.add_argument(
        '--generate_plots',
        nargs='+',  # Allow multiple plot generations
        choices=['all', 'optics', 'spectral', 'kmeans', 'xmeans', 'globalkmeans', 'fuzzy'],
        help="Generate plots for the specified algorithm(s): 'optics', 'spectral', 'kmeans', 'xmeans', 'globalkmeans' 'fuzzy', or 'all'"
    )

    args = parser.parse_args()

    # Ensure at least one action is specified
    if not args.experiment and not args.generate_plots:
        parser.error(
            "You must specify at least one action: --experiment or --generate_plots.")

    # Handle experiment execution
    if args.experiment:
        print(f"Running {args.experiment} experiment...")
        if args.experiment == 'optics':
            run_optics()
        elif args.experiment == 'spectral':
            run_spectral()
        elif args.experiment == 'kmeans':
            run_kmeans()
        elif args.experiment == 'gkmeans':
            run_gkmeans()
        elif args.experiment == 'xmeans':
            run_xmeans()
        elif args.experiment == 'fuzzy':
            run_fuzzy()
        print(f"{args.experiment} experiment completed.\n")

    # Handle plot generation
    if args.generate_plots:
        plots = args.generate_plots
        if 'all' in plots:
            run_all_plots()
        else:
            for algo in plots:
                print(f"Generating plots for {algo}...")
                try:
                    generate_plots(algo)
                    print(f"Plots for {algo} generated successfully.\n")
                except Exception as e:
                    print(f"Error generating plots for {algo}: {e}\n")


if __name__ == '__main__':
    main()
