import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import nn

def BNN(X, y=None, width=4, sigma=1.0, D_Y=None, activation=jnp.tanh, subsample=None):
    # Make sure D_Y is defined
    if y is None and D_Y is None:
        raise ValueError("Either y or D_Y must be provided.")
    if y is not None:
        if y.ndim > 1:
            y = y.flatten()
            D_Y = y.shape[-1]
        else:
            D_Y = 1

    N = X.shape[-2]
    D_X = X.shape[-1]
    D_Z = width

    # First layer
    w0 = numpyro.sample("w0", dist.Normal(0.0, 1).expand((D_X, D_Z)))
    b0 = numpyro.sample("b0", dist.Normal(0.0, 1).expand((D_Z, )))

    # Middle layers:
    w1 = numpyro.sample(f"w1", dist.Normal(0.0, 1).expand((D_Z, D_Z)))
    b1 = numpyro.sample(f"b1", dist.Normal(0.0, 1).expand((D_Z,)))

    # Last layer
    w2 = numpyro.sample(f"w2", dist.Normal(0.0, 1).expand((D_Z, D_Y)))
    b2 = numpyro.sample(f"b2", dist.Normal(0.0, 1).expand((D_Y,)))

    with numpyro.plate("data", N, subsample_size=subsample if subsample is not None else N) as ind:
        X_batch = X[ind]
        y_batch = y[ind] if y is not None else None

        # Forward pass
        z_p = activation(X_batch @ w0 + b0)
        z_p = activation(z_p @ w1 + b1)
        z = (z_p @ w2 + b2).flatten()
        if y_batch is not None:
            assert z.shape == y_batch.shape, f"Shapes (z,y): {(z.shape, y_batch.shape)}"

        y_loc = numpyro.deterministic("y_loc", z)
        numpyro.sample("y", dist.Normal(y_loc, sigma), obs=y_batch)

def UCI_BNN(X, y=None, width=50, D_Y=None, subsample=None):
    prec = numpyro.sample("prec", dist.Gamma(1.0, 0.1))
    _sigma = jnp.sqrt(1 / prec)
    BNN(X, y, width=width, D_Y=D_Y, sigma=_sigma, activation=nn.relu, subsample=subsample)


def UCI_BNN_tanh(X, y=None, width=50, D_Y=None, subsample=None):
    prec = numpyro.sample("prec", dist.Gamma(1.0, 0.1))
    _sigma = jnp.sqrt(1 / prec)
    BNN(X, y, width=width, D_Y=D_Y, sigma=_sigma, activation=jnp.tanh, subsample=subsample)