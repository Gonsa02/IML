import argparse
import os

from knn.run_knn_experiment import run_knn_experiment
from svm.run_svm_experiment import run_svm_experiment
from run_statistical_tests import run_statistical_tests


def main():
    parser = argparse.ArgumentParser(description='Run experiments')
    parser.add_argument('--experiment', choices=['knn', 'svm'], required=True,
                        help='Type of experiment to run: knn or svm')
    parser.add_argument('--ir_method', choices=['drop3', 'ennth', 'gcnn'], default=None,
                        help='Instance reduction method to apply: drop3, ennth, gcnn. If not specified, no instance reduction is applied.')
    parser.add_argument('--run_tests', action='store_true',
                        help='Run statistical tests on the results')
    args = parser.parse_args()

    if args.experiment == 'knn':
        run_knn_experiment(instance_reduction_method=args.ir_method)
    elif args.experiment == 'svm':
        run_svm_experiment(instance_reduction_method=args.ir_method)

    if args.run_tests:
        run_statistical_tests(args.experiment)

if __name__ == '__main__':
    main()