import argparse
import os

from knn.run_knn_experiment import run_knn_experiment
from svm.run_svm_experiment import run_svm_experiment
from run_statistical_tests import run_statistical_tests


def main():
    parser = argparse.ArgumentParser(description='Run experiments')
    parser.add_argument('--experiment', choices=['knn', 'svm', 'knn_ir', 'svm_ir'], required=True,
                        help='Type of experiment to run: knn, svm, knn_ir (knn with instance reduction), svm_ir (svm with instance reduction)')
    parser.add_argument('--run_tests', action='store_true',
                        help='Run statistical tests on the results')
    args = parser.parse_args()

    if args.experiment == 'knn':
        run_knn_experiment(instance_reduction=False)
    elif args.experiment == 'svm':
        run_svm_experiment(instance_reduction=False)
    elif args.experiment == 'knn_ir':
        run_knn_experiment(instance_reduction=True)
    elif args.experiment == 'svm_ir':
        run_svm_experiment(instance_reduction=True)

    if args.run_tests:
        run_statistical_tests(args.experiment)

if __name__ == '__main__':
    main()
