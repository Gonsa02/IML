import argparse
import os

from knn.run_knn_experiment import run_knn_experiment
from svm.run_svm_experiment import run_svm_experiment
from knn.run_knn_ir_experiment import run_knn_ir_experiment
from svm.run_svm_ir_experiment import run_svm_ir_experiment
from run_statistical_tests import run_statistical_tests


def main():
    parser = argparse.ArgumentParser(description='Run experiments')
    parser.add_argument('--experiment', choices=['knn', 'svm', 'knn_ir', 'svm_ir'], required=True,
                        help='Type of experiment to run: knn, svm, knn_ir, svm_ir')
    parser.add_argument('--ir_method', choices=['drop3', 'ennth', 'gcnn'], default=None,
                        help='Instance reduction method to apply (only for knn_ir and svm_ir): drop3, ennth, gcnn.')
    parser.add_argument('--run_tests', action='store_true',
                        help='Run statistical tests on the results')
    args = parser.parse_args()

    if args.experiment == 'knn':
        if args.ir_method:
            parser.error("--ir_method is not allowed with knn experiment. Did you mean knn_ir?")
        run_knn_experiment()

    elif args.experiment == 'svm':
        if args.ir_method:
            parser.error("--ir_method is not allowed with svm experiment. Did you mean svm_ir?")
        run_svm_experiment()

    elif args.experiment == 'knn_ir':
        if not args.ir_method:
            parser.error("--ir_method is required for knn_ir experiment.")
        run_knn_ir_experiment(ir_method=args.ir_method)

    elif args.experiment == 'svm_ir':
        if not args.ir_method:
            parser.error("--ir_method is required for svm_ir experiment.")
        run_svm_ir_experiment(ir_method=args.ir_method)

    if args.run_tests:
        run_statistical_tests(args.experiment)

if __name__ == '__main__':
    main()