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
        self.X = self.data[:,:-1]
        self.y = self.data[:,-1]

    def normalize_X(self, X, split) -> jnp.array:
        train_X = X[split["tr"]]
        val_X = X[split["val"]]
        test_X = X[split["te"]]
        # Normalize all independent variables per split
        # Training data is used to calculate mean and std, which are then applied to all splits
        mean = train_X.mean(axis=0)
        std = train_X.std(axis=0)

        X = X.at[split["tr"]].set((train_X - mean) / std)
        X = X.at[split["val"]].set((val_X - mean) / std)
        X = X.at[split["te"]].set((test_X - mean) / std)

        # Validation step
        train_X = X[split["tr"]]
        val_X = X[split["val"]]
        test_X = X[split["te"]]
        assert np.allclose(train_X.mean(axis=0), 0, atol=1e-2)
        assert np.allclose(train_X.std(axis=0), 1, atol=1e-2)

        old_train_X = self.data[:,:-1][split["tr"]]
        old_val_X = self.data[:,:-1][split["val"]]
        old_test_X = self.data[:,:-1][split["te"]]
        assert np.allclose(train_X * std + mean, old_train_X, atol=1e-2)
        assert np.allclose(val_X * std + mean, old_val_X, atol=1e-2)
        assert np.allclose(test_X * std + mean, old_test_X, atol=1e-2)

        return X

            
