# Clustering Experiment Runner

This script allows you to run various clustering experiments using algorithms such as OPTICS, Spectral Clustering, K-Means, X-Means, and Fuzzy Clustering.

## Requirements

Ensure that you have the necessary libraries installed before running the experiments.

### Environment Setup

It's recommended to create a Python 3.9 environment to ensure compatibility. You can set this up using either `conda` or `venv`:

#### Using Conda

```bash
conda create -n clustering_experiment_runner python=3.9
conda activate clustering_experiment_runner
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

You can run the script from the command line, specifying the type of clustering experiment you wish to conduct.

## Running Experiments

The main command structure is as follows:

```bash
python main.py --experiment <experiment_type>
```

### Experiment Types

- **optics:** Run an OPTICS clustering experiment.
- **spectral:** Run a Spectral Clustering experiment.
- **kmeans:** Run a K-Means clustering experiment.
- **xmeans:** Run an X-Means clustering experiment.
- **fuzzy:** Run a Fuzzy Clustering experiment.

### Example Commands

#### OPTICS Experiment

To run an OPTICS experiment:

```bash
python main.py --experiment optics
```

### Notes

- Specify the experiment type using the `--experiment` option.
- Ensure that only one experiment type is specified at a time.

## Getting Help

If you encounter any issues or have questions, feel free to open an issue on the repository.