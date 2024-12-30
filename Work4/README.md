# Dimensionality Reduction Experiment Runner

This script allows you to run dimensionality reduction experiments using the following algorithms: PCA, incrementalPCA, kernelPCA and UMAP. You can generate visual plots of the results for each algorithm.

## Requirements
Ensure that you have the necessary libraries installed before running the experiments.

### Environmental Setup
It is recommended to create a Python 3.9 environment to ensure compatibility. You can set this up using either `conda` or `venv`:

#### Using Conda

```bash
conda create -n dimensionality_reduction_experiment_runner python=3.9
conda activate dimensionality_reduction_experiment_runner
```

#### Using venv

```bash
python3.9 -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```

Once the environment is activated, install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage
You can run the script from the command line to execute dimensionality reduction experiments and generate plots of the experiment results. Visualizations will appear on your screen as the program executes. 

## Running Experiments and Generating Plots
To run a specific dimensionality reduction experiment, use the --experiment option followed by the experiment type.

```bash
python main.py --experiment <algorithm>
```

### Dimensionality Reduction Algorithms 

- **pca**: Run our own PCA implementation. _Parts 1 and 2 of the assignment_.
- **incremental_pca**: Run our own PCA implementation, sklearn PCA and sklearn Incremental PCA. _Part 3 of the assignment_.
- **clustering_our_pca**: Run KMeans and OPTICS using our PCA implementation. _Part 4 of the assignment_.
- **clustering_sklearn_kernelpca**: Run and plot KMeans and OPTICS with sklearn kernelPCA. _Part 5 of the assignment_. 
- **clustering_pca_and_umap**: Run and plot PCA and UMAP for KMeans and OPTICS clustering methods. _Part 6 of the assignment_
- **all**: Run all previous dimensionality reduction experiments.



### Example Commands
#### PCA experiment

To run the experiments from our implementation of PCA:
```bash
python main.py --experiment pca
```

### Notes
- Specify the experiment type using the `--experiment` option.
- Ensure that only valid experiment and algorithm types are specified.


## Getting Help

If you encounter any issues or have questions, feel free to open an issue on the repository.