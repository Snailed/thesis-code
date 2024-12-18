import jax.numpy as jnp
from jax import random
from abc import ABC

class Dataset(ABC):
    def __init__(self, rng_key):
        self.X = None
        self.Y = None

class SineRegression(Dataset):
    def __init__(self, rng_key, sigma=0.02):
        self.sigma=sigma
        self.x_obs = jnp.hstack([jnp.linspace(-0.2, 0.2, 500), jnp.linspace(0.6, 1, 500)])
        self.noise = 0.02 * random.normal(rng_key, self.x_obs.shape[0])
        self.y_obs = self.x_obs + 0.3 * jnp.sin(2 * jnp.pi * (self.x_obs + self.noise)) + 0.3 * jnp.sin(4 * jnp.pi * (self.x_obs + self.noise)) + self.noise

        self.x_true = jnp.linspace(-0.5, 1.5, 1000)
        self.y_true = self.x_true + 0.3 * jnp.sin(2 * jnp.pi * self.x_true) + 0.3 * jnp.sin(4 * jnp.pi * self.x_true)

        self.X = jnp.array([self.x_obs]).T
        self.Y = jnp.array([self.y_obs]).T
