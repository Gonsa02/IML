# Experiment Runner

This script allows you to run various machine learning experiments using K-Nearest Neighbors (KNN) and Support Vector Machine (SVM) algorithms, with options for instance reduction methods.

## Requirements

Make sure to install the necessary libraries before running the experiments. You can install them using:

```bash
pip install -r requirements.txt
```

## Usage

You can run the script from the command line with different options based on the experiment you wish to conduct.

### Running Experiments

The main command structure is as follows:

```bash
python main.py --experiment <experiment_type> [options]
```

### Experiment Types

- **knn**: Run a K-Nearest Neighbors experiment.
- **svm**: Run a Support Vector Machine experiment.
- **knn_ir**: Run KNN with an instance reduction method.
- **svm_ir**: Run SVM with an instance reduction method.

### Example Commands

#### KNN Experiment

To run a KNN experiment:

```bash
python main.py --experiment knn
```

#### SVM Experiment

To run an SVM experiment:

```bash
python main.py --experiment svm
```

#### KNN with Instance Reduction

To run a KNN experiment with an instance reduction method, specify the `--ir_method` option:

```bash
python main.py --experiment knn_ir --ir_method <method>
```

Where `<method>` can be one of the following:

- `drop3`
- `ennth`
- `gcnn`

Example:

```bash
python main.py --experiment knn_ir --ir_method drop3
```

#### SVM with Instance Reduction

To run an SVM experiment with an instance reduction method, use:

```bash
python main.py --experiment svm_ir --ir_method <method>
```

Example:

```bash
python main.py --experiment svm_ir --ir_method gcnn
```

### Statistical Tests

You can also run statistical tests on the results. Use the following option:

```bash
python main.py --stat_test <test_type>
```

Test Types
- best_knn: Run statistical tests for the best KNN configuration.
- best_svm: Run statistical tests for the best SVM configuration.
- knn_vs_svm: Compare KNN and SVM configurations.
- ir_knn: Compare instance reduction methods within KNN.
- ir_svm: Compare instance reduction methods within SVM.

#### Statistical Test Example

To run a statistical test comparing the best KNN and SVM configurations:

```bash
python main.py --stat_test knn_vs_svm
```

### Notes

- Specify either `--experiment` or `--stat_test`, not both.
- When running `knn_ir` or `svm_ir` experiments, an `--ir_method` option is required.