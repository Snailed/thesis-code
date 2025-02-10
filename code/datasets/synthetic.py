import numpy as np
import jax.numpy as jnp

# From https://github.com/aleatory-science/smi_experiments/blob/main/src/sine_wave_data.py
# adapted to class-based pattern

class SyntheticDataset():
    def __init__(self):
        self.dataset_name = "synthetic"
        self.data_seed = 52
        self.C1_domain = (-1.5, -0.5)
        self.C2_domain = (1.3, 1.7)
        self.X_domain = (-2.0, 2.0)
        self.noise_level = 0.1
        self.X_true = np.linspace(*self.X_domain, 500)
        self.y_true = self._sine_wave_function(self.X_true)
        self.X_train, self.y_train = self._sample_clusters(20,20)

        self.X_test_in, self.y_test_in = self._sample_uniform_clusters(20)
        self.X_test_between, self.y_test_between = self._sample_uniform_interval(-0.5, 1.3, 60)
        self.X_test_entire, self.y_test_entire = self._sample_uniform_interval(*self.X_domain, 120)

    
    def _sine_wave_function(self, t):
        amp = 1.5
        freq = 1
        phase = 2 / 3 * np.pi
        linear_coef = 3.0
        bias = -1.0
        return amp * np.sin(2 * np.pi * freq * t + phase) + linear_coef * t - bias
    
    def _data_wave_function(self, t):
        np.random.seed(self.data_seed)
        return jnp.array(self._sine_wave_function(t) + np.random.randn(t.shape[0])*self.noise_level)
    
    def _sample_clusters(self,nc1, nc2):
        np.random.seed(self.data_seed)
        c1x = np.random.uniform(*self.C1_domain, size=(nc1,))
        c2x = np.random.uniform(*self.C2_domain, size=(nc2,))
        x = np.concatenate([c1x, c2x])
        idx = jnp.argsort(x)
        x = x[idx]
        y = self._data_wave_function(x)
        return jnp.array(x).reshape(-1, 1), jnp.array(y)
    
    def _sample_uniform_interval(self, min_val, max_val, n):
        np.random.seed(self.data_seed)
        x = np.random.uniform(min_val, max_val, size=(n,))
        idx = jnp.argsort(x)
        x = x[idx]
        y = self._data_wave_function(x)
        return jnp.array(x).reshape(-1, 1), jnp.array(y)

    def _sample_uniform_clusters(self,n):
        np.random.seed(self.data_seed)
        xs = []
        for _ in range(n):
            if np.random.binomial(2, 0.5):
                xs.append(np.random.uniform(*self.C1_domain))
            else:
                xs.append(np.random.uniform(*self.C2_domain))
        x = np.array(xs)
        idx = jnp.argsort(x)
        x = x[idx]
        y = self._data_wave_function(x)
        return jnp.array(x).reshape(-1, 1), jnp.array(y)
