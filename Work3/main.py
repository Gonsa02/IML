import argparse

from optics.run_optics import run_optics
from spectral.run_spectral import run_spectral
from kmeans.run_kmeans import run_kmeans
from kmeans.run_xmeans import run_xmeans
#from fuzzy.run_fuzzy import run_fuzzy

def main():
    parser = argparse.ArgumentParser(description='Run clustering experiments')

    # Experiment options
    parser.add_argument('--experiment', 
                        choices=['optics', 'spectral', 'kmeans', 'xmeans', 'fuzzy'],
                        help='Type of experiment to run: optics, spectral, kmeans, xmeans, fuzzy')
    
    args = parser.parse_args()
    
    # Ensure an experiment is specified
    if not args.experiment:
        parser.error("You must specify an experiment using --experiment.")

    # Dispatch to the appropriate experiment
    if args.experiment == 'optics':
        run_optics()
    elif args.experiment == 'spectral':
        run_spectral()
    elif args.experiment == 'kmeans':
        run_kmeans()
    elif args.experiment == 'xmeans':
        run_xmeans()
    #elif args.experiment == 'fuzzy':
    #    run_fuzzy()

if __name__ == '__main__':
    main()