import argparse

from optics.run_optics import run_optics
from spectral.run_spectral import run_spectral
from kmeans.run_kmeans import run_kmeans
from kmeans.run_xmeans import run_xmeans
from fuzzy.run_fuzzy import run_fuzzy

# Import the generate_plots function
from plots import generate_plots

def main():
    parser = argparse.ArgumentParser(description='Run clustering experiments and generate plots')

    # Experiment option
    parser.add_argument(
        '--experiment', 
        choices=['optics', 'spectral', 'kmeans', 'xmeans', 'fuzzy'],
        help='Type of experiment to run: optics, spectral, kmeans, xmeans, fuzzy'
    )
    
    # Generate plots option
    parser.add_argument(
        '--generate_plots',
        choices=['optics', 'spectral', 'kmeans', 'xmeans', 'fuzzy'],
        help='Generate plots for the specified algorithm'
    )
    
    args = parser.parse_args()
    
    # Ensure at least one action is specified
    if not args.experiment and not args.generate_plots:
        parser.error("You must specify at least one action: --experiment or --generate_plots.")
    
    # Handle experiment execution
    if args.experiment:
        print(f"Running {args.experiment} experiment...")
        if args.experiment == 'optics':
            run_optics()
        elif args.experiment == 'spectral':
            run_spectral()
        elif args.experiment == 'kmeans':
            run_kmeans()
        elif args.experiment == 'xmeans':
            run_xmeans()
        elif args.experiment == 'fuzzy':
            run_fuzzy()
        print(f"{args.experiment} experiment completed.\n")
    
    # Handle plot generation
    if args.generate_plots:
        print(f"Generating plots for {args.generate_plots}...")
        generate_plots(args.generate_plots)
        print(f"Plots for {args.generate_plots} generated successfully.\n")

if __name__ == '__main__':
    main()
