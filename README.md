# obelix-ml-pipeline
Data and code related to the ML pipeline used in the paper '...' introducing our [OBeLiX workflow](https://github.com/EPiCs-group/obelix).

## Installation
First clone the repository:

```bash
git clone https://github.com/EPiCs-group/obelix-ml-pipeline
```

Next, install or load conda and create a new environment from the environment.yml file:

```bash
conda env create -f environment.yml
```

Then activate the environment:

```bash
conda activate obelix-ml-pipeline
```

Afterwards, install the package:

```bash
pip install -e .
```
Then the notebooks can be run to reproduce the results. In your anaconda prompt, run:

```bash
cd obelix-ml-pipeline/notebooks
jupyter notebook
```

## Data
**filename**: contains ...

## Code
**representation_variables.py**: contains a selection of features for representations of the ligand or substrate.

**utilities.py**: contains functions for data loading and general utilities.

**load_representations.py**: contains functions for loading representations of the ligand or substrate.

**machine_learning.py**: contains functions for machine learning, training and testing.

**predictions_\*.py**: contains the 3 main use cases of the ML pipeline.

## Notes