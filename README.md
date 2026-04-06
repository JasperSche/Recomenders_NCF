# Neural Collaborative Filtering in PyTorch

This repository contains a small PyTorch implementation of three recommendation models for implicit-feedback movie recommendation:

- **GMF** (`GMF.py`): Generalized Matrix Factorization
- **MLP** (`MLP.py`): Multi-Layer Perceptron recommender
- **NeuMF** (`NeuMF.py`): Hybrid model combining GMF and MLP
- **Dataset** (`Dataset.py`): data loading, train/validation/test split, and negative sampling
- **Experiments** (`Experiment.py`): comparison runs and hyperparameter sweeps

## Data

The code expects a MovieLens-style ratings file at:

`Data/ratings.dat`

Each line should follow the format:

`user_id::item_id::rating::timestamp`

Ratings are converted to implicit feedback by keeping interactions with rating **>= 4** as positive examples.

## How it works

- Users' positive interactions are split into **train / validation / test** sets.
- Training uses **negative sampling**.
- Models are trained with **binary cross-entropy loss** and **Adam**.
- `experiment.py` supports pretraining GMF/MLP and finetuning NeuMF.

## Run

Run the experiment pipeline with:

$ python experiment.py