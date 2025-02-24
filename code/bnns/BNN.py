import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import nn

def BNN(X, y=None, depth=1, width=4, sigma=1.0, D_Y=None, activation=jnp.tanh):
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
    if depth == 1:
        D_Z = D_Y

    # First layer
    w = numpyro.sample("w0", dist.Normal(0.0, 1).expand((D_X, D_Z)))
    b = numpyro.sample("b0", dist.Normal(0.0, 1).expand((D_Z, )))
    z = X @ w + b.flatten()
    z_p = activation(z)

    # Middle layers:
    for i in range(1, depth):
        w = numpyro.sample(f"w{i}", dist.Normal(0.0, 1).expand((D_Z, D_Z)))
        b = numpyro.sample(f"b{i}", dist.Normal(0.0, 1).expand((D_Z,)))
        z = z_p @ w + b
        z_p = activation(z)

    # Last layer
    w = numpyro.sample(f"w{depth}", dist.Normal(0.0, 1).expand((D_Z, D_Y)))
    b = numpyro.sample(f"b{depth}", dist.Normal(0.0, 1).expand((D_Y,)))
    z = (z_p @ w + b).flatten() # (N, 1) -> (N,)
    if y is not None:
        assert z.shape == y.shape, f"Shapes (z,y): {(z.shape, y.shape)}"
    with numpyro.plate("data", N):
        y_loc = numpyro.deterministic("y_loc", z)
        numpyro.sample("y", dist.Normal(y_loc, sigma), obs=y)

def UCI_BNN(X, y=None, depth=2, width=50, D_Y=None):
    prec = numpyro.sample("prec", dist.Gamma(1.0, 0.1))
    _sigma = jnp.sqrt(1 / prec)
    BNN(X, y, depth=depth, width=width, D_Y=D_Y, sigma=_sigma, activation=nn.relu)


def UCI_BNN_tanh(X, y=None, depth=2, width=50, D_Y=None):
    prec = numpyro.sample("prec", dist.Gamma(1.0, 0.1))
    _sigma = jnp.sqrt(1 / prec)
    BNN(X, y, depth=depth, width=width, D_Y=D_Y, sigma=_sigma, activation=jnp.tanh)