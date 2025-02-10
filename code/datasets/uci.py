from ola_datasets.uci_reg.load_uci import loadtxt, UCI_DIR, load_standard_splits
import numpy as np
import jax.numpy as jnp
import pandas as pd
import os

_standard_splits = load_standard_splits(True)

dataset_names = ["boston-housing", "concrete", "energy", "kin8nm", "naval-propulsion-plant", "power-plant", "protein-tertiary-structure", "wine-quality-red", "yacht"]

class UCIDataset():
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.splits = _standard_splits[dataset_name]
        self.data = jnp.array(loadtxt(f"{UCI_DIR}/{dataset_name}/data/data.txt"))
        self.normalize()
        self.X = self.data[:,:-1]
        self.y = self.data[:,-1]
        # self.X_train = data["tr"][:,:-1]
        # self.y_train = data["tr"][:,-1]
        # self.X_val = data["val"][:,:-1]
        # self.y_val = data["val"][:,-1]
        # self.X_test = data["te"][:,:-1]
        # self.y_test = data["te"][:,-1]
    def normalize(self):
        # Normalize all independent variables per split
        # Training data is used to calculate mean and std, which are then applied to all splits
        X = self.data[:,:-1]
        y = self.data[:,-1]
        for split in self.splits:
            train_X = X[split["tr"]]
            val_X = X[split["val"]]
            test_X = X[split["te"]]

            mean = train_X.mean(axis=0, keepdims=True)
            std = train_X.std(axis=0, keepdims=True)

            X = X.at[split["tr"]].set((train_X - mean) / std)
            X = X.at[split["val"]].set((val_X - mean) / std)
            X = X.at[split["te"]].set((test_X - mean) / std)
        self.data = jnp.concatenate((X, y.reshape(-1, 1)), axis=1)

            
