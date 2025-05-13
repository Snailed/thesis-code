import kagglehub
import numpy as np
import jax.numpy as jnp
import jax
import pandas as pd
from sklearn.utils import resample
import os

class ECGDataset():
    dataset_name = "ecg"
    def __init__(self, n=2000, random_state=42, resample_train=False):
        path = kagglehub.dataset_download("shayanfazeli/heartbeat")
        train_df = pd.read_csv(os.path.join(path, "mitbih_train.csv"), header=None)
        test_df = pd.read_csv(os.path.join(path, "mitbih_test.csv"), header=None)

        # Get train_df class distribution
        self.train_class_probs = train_df[187].value_counts(normalize=True)

        # Resample to get balanced classes
        if resample_train:
            df_1=train_df[train_df[187]==1]
            df_2=train_df[train_df[187]==2]
            df_3=train_df[train_df[187]==3]
            df_4=train_df[train_df[187]==4]
            df_0=(train_df[train_df[187]==0]).sample(n=n,random_state=random_state)

            df_1_upsample=resample(df_1,replace=True,n_samples=n,random_state=random_state + 1)
            df_2_upsample=resample(df_2,replace=True,n_samples=n,random_state=random_state + 2)
            df_3_upsample=resample(df_3,replace=True,n_samples=n,random_state=random_state + 3)
            df_4_upsample=resample(df_4,replace=True,n_samples=n,random_state=random_state + 4)

            train=pd.concat([df_0,df_1_upsample,df_2_upsample,df_3_upsample,df_4_upsample])
        else:
            train = train_df
        train=jnp.array(train.sample(frac=1,random_state=random_state).values)
        test = jnp.array(test_df.values)

        self.data = jnp.concat([train, test])
        self.X = jnp.concat([train[:,:-1], test[:, :-1]])
        self.y = jnp.concat([train[:,-1], test[:, -1]])
        #self.y = jax.nn.one_hot(self.y, num_classes=5)
        self.splits = [
            {
                "tr": np.arange(0, train.shape[0]),
                "val": np.arange(train.shape[0], train.shape[0] + test.shape[0]), # val and test are the same
                "te": np.arange(train.shape[0], train.shape[0] + test.shape[0])
            }
        ]
        

    def normalize_X(self, X, split) -> jnp.array:
        train_X = X[split["tr"]]
        test_X = X[split["te"]]
        # Normalize all independent variables per split
        # Training data is used to calculate mean and std, which are then applied to all splits
        mean = train_X.mean(axis=0)
        std = train_X.std(axis=0)

        X = X.at[split["tr"]].set((train_X - mean) / std)
        X = X.at[split["te"]].set((test_X - mean) / std)

        # Validation step
        train_X = X[split["tr"]]
        test_X = X[split["te"]]
        assert np.allclose(train_X.mean(axis=0), 0, atol=1e-2)
        assert np.allclose(train_X.std(axis=0), 1, atol=1e-2)

        old_train_X = self.X[split["tr"]]
        old_test_X = self.X[split["te"]]
        assert np.allclose(train_X * std + mean, old_train_X, atol=1e-2)
        assert np.allclose(test_X * std + mean, old_test_X, atol=1e-2)

        return X
    
    def train_label_distribution(self):
        """
        Returns the distribution of labels in the dataset.
        """
        return jnp.array(self.train_class_probs)
