# obelix-ml-pipeline
This Github repo contains the ML pipeline used in the paper '...'. For the exact 'production' versions used to generate the
figures in the paper, visit ...  

First, homogeneous catalyst structures are featurized using
our [OBeLiX workflow](https://github.com/EPiCs-group/obelix). Various other representations are created for these structures
and used as input for machine learning models. The models are trained to predict the enantioselectivity or conversion. 


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
Then the notebooks can be run to run examples of the pipeline. In your anaconda prompt, run:

```bash
cd obelix-ml-pipeline/notebooks
jupyter notebook
```

## Repository structure
In the paper, 4 different prediction tasks are performed. Fully out-of-domain, partially out-of-domain, in-domain and 
monte-carlo in-domain. The functions to perform these tasks are kept in their own files. Example use cases
are shown in the notebooks. The ligand representations, substrate representations and experimental response are 
loaded from the data folder. A more detailed description of the files is given below.

## Data
* **Experimental response/**  
Contains the experimental response for the different substrates and solvents.  
    * **jnjdata_sm12378_MeOH_16h**: contains the experimental response for 16 hours with one solvent.
    * **jnjdata_sm12378_MeOH**: contains the experimental response for at 1 hour for SM1/2/3 and 16 hours for the other substrates with one solvent.
  
* **Ligand representations/**  
Contains the various representations of the ligands.  
    * **raw_data_processing**: contains the raw data and scripts needed to create each representation.

* **Substrate representations/**  
Contains the various representations of the substrates.  
    * **raw_data_processing**: contains the raw data and scripts needed to create each representation.

## Code
**representation_variables.py**: contains a selection of features for representations of the ligand or substrate. 
When the file itself is run, it will create correlation plots for the features in the file.

**utilities.py**: contains functions for data loading and general utilities.

**load_representations.py**: contains functions for loading representations of the ligand or substrate.

**machine_learning.py**: contains functions for machine learning, training and testing.

**data_classes.py:** contains classes for the data returned in the ML pipeline.

**predictions\_on\_unseen\_substrate.py**: contains the functions to perform the fully out-of-domain prediction task.

**predictions\_on\_unseen\_substrate\_filtered.py**: contains the functions to fully out-of-domain prediction task, except in this case 
if a classification task is performed, ligands that are in the same class across all training substrates will be removed from the set. This was done to test 
how well the models work if ligands that always perform well are removed.

**predictions\_on\_partially\_unseen\_substrate.py**: contains the functions to perform the partially out-of-domain prediction task.

**predictions\_within\_substrate\_class.py**: contains the functions to perform the in-domain prediction task.

**predictions\_within\_substrate\_class\_for\_random\_subset.py**: contains the functions to perform the monte-carlo in-domain prediction task.



## Notes