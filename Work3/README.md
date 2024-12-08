# Clustering Experiment Runner

This script allows you to run various clustering experiments using algorithms such as OPTICS, Spectral Clustering, K-Means, X-Means, and Fuzzy Clustering. Additionally, you can generate visual plots of the results for each algorithm.

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

You can run the script from the command line to either execute clustering experiments or generate plots of the experiment results.

## Running Experiments

To run a specific clustering experiment, use the `--experiment` option followed by the experiment type.

```bash
python main.py --experiment <algorithm>
```

### Algorithms

- **optics:** Run an OPTICS clustering experiment.
- **spectral:** Run a Spectral Clustering experiment.
- **kmeans:** Run a K-Means clustering experiment.
- **globalkmeans**: Run a Global K-Means clustering experiment.
- **xmeans:** Run an X-Means clustering experiment.
- **fuzzy:** Run a Fuzzy Clustering experiment.
- **all:** Run all previous clustering experiments.

### Example Commands

#### OPTICS Experiment

To run an OPTICS experiment:

```bash
python main.py --experiment optics
```

## Generating Plots

To generate visual plots for a specific algorithm's experiment results, use the `--generate_plots` option followed by the algorithm name.

```bash
python main.py --generate_plots <algorithm>
```

### Algorithms

- **optics:** Run an OPTICS clustering experiment.
- **spectral:** Run a Spectral Clustering experiment.
- **kmeans:** Run a K-Means clustering experiment.
- **globalkmeans**: Run a Global K-Means clustering experiment.
- **xmeans:** Run an X-Means clustering experiment.
- **fuzzy:** Run a Fuzzy Clustering experiment.
- **all:** Run all previous clustering experiments.

### Example Commands

#### Generate Plots for K-Means

```bash
python main.py --generate_plots kmeans
```

### Notes

- Specify the experiment type using the `--experiment` option.
- Specify the algorithm for plot generation using the `--generate_plots` option.
- Ensure that only valid experiment and algorithm types are specified.

## Getting Help

If you encounter any issues or have questions, feel free to open an issue on the repository.