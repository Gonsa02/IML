import argparse
import os

from knn.run_knn_experiment import run_knn_experiment
from svm.run_svm_experiment import run_svm_experiment
from knn.run_knn_ir_experiment import run_knn_ir_experiment
from svm.run_svm_ir_experiment import run_svm_ir_experiment
from run_statistical_tests import run_statistical_tests

def main():
    parser = argparse.ArgumentParser(description='Run experiments or statistical tests')
    
    # Experiment options
    parser.add_argument('--experiment', choices=['knn', 'svm', 'knn_ir', 'svm_ir'], 
                        help='Type of experiment to run: knn, svm, knn_ir, svm_ir')
    parser.add_argument('--ir_method', choices=['drop3', 'ennth', 'gcnn'], default=None,
                        help='Instance reduction method to apply (only for knn_ir and svm_ir): drop3, ennth, gcnn.')

    # Statistical test option
    parser.add_argument('--stat_test', choices=['best_knn', 'best_svm', 'knn_vs_svm', 'ir_knn', 'ir_svm'], 
                        help='Statistical test to run: best_knn, best_svm, knn_vs_svm, ir_knn, ir_svm')

    args = parser.parse_args()

    # Check if an experiment or a test has been specified
    if args.experiment and args.stat_test:
        parser.error("You can only specify either --experiment or --stat_test, not both.")

    # Run an experiment
    if args.experiment:
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

    # Run a statistical test
    elif args.stat_test:
        os.makedirs('results', exist_ok=True)
        run_statistical_tests(args.stat_test)

    else:
        parser.error("You must specify either --experiment or --stat_test.")

if __name__ == '__main__':
    main()
