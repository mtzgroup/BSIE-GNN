# BSIE-GNN

Steps to create environment:
1) If you don't have Tensorflow 2.x installed already, create conda environment and install tensorflow
    > conda create --name BSIE
    > conda activate BSIE
    > conda install -c anaconda tensorflow-gpu

2) Install the this python package
    > pip install -e .


After installing the python package, a few scripts will be available at the command line. These scripts
allows the user to train models and make predictions by giving xyz file(s) as input. To use the scripts,
look the the command line options available for each script, or just try the examples below:

Train a model:
    > train_model.py -l 1.0e-3 -e 2 -b 64 -t 400 -d Datasets/ -a Weights/atomic_const_weights.txt -c 6.0

Make a prediction using trained model
    > predict.py Data/LinearHydrocarbons/XYZ/1.xyz -w Weights/B3LYP/GDB+S66_delta_D5/0.hdf5


Weights/
    Pre-trained weights are included in the repository for quick evaluations. The weights found at
    Weights/B3LYP/GDB+S66_delta_D5 are obtained by training the BSIE-GNN model on the cc-pVDZ/cc-pV5Z
    incompleteness error of the B3LYP method. The dataset used for training was the GDB-BSIE dataset and
    the S66-BSIE dataset.

Data/
    Test datasets included in the repository. All of the training data used to train the BSIE-GNN model
    can be found at the following links:
    https://doi.org/10.5281/zenodo.7402871 (GDB-BSIE)
    https://doi.org/10.5281/zenodo.7402847 (S66-BSIE)


The jupyter-notebook Notebooks/create_test_datasets.ipynb is inluded to show how to make a dataset and how
to create input data. In the notebook Notebooks/evaluate_model.ipynb, the trained models are evaluated on
the test datasets found in Data/.
