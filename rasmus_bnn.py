import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import nn

LATENT = [
    "prec",
    "w0",
    "b0",
    "w1",
    "b1",
    "w2",
    "b2",
]
OUT = ["y_loc", "y"]


def BNN(X, y=None, depth=1, width=4, sigma=1.0, D_Y=None, activation=jnp.tanh, subsample=None):
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
    # OLD version (no index)
    # w = numpyro.sample("w0", dist.Normal(0.0, 1).expand((D_X, D_Z)))
    # b = numpyro.sample("b0", dist.Normal(0.0, 1).expand((D_Z, )))

    # OR: Added variable index to allow subsampling
    w0 = numpyro.sample("w0", dist.Normal(0.0, 1).expand((D_X, D_Z)))
    b0 = numpyro.sample("b0", dist.Normal(0.0, 1).expand((D_Z, )))

    # OLD version (no subsampling)
    # z = X @ w + b.flatten()
    # z_p = activation(z)

    # Middle layers:
    # OLD version
    # for i in range(1, depth):
    #     w = numpyro.sample(f"w{i}", dist.Normal(0.0, 1).expand((D_Z, D_Z)))
    #     b = numpyro.sample(f"b{i}", dist.Normal(0.0, 1).expand((D_Z,)))
    #     z = z_p @ w + b
    #     z_p = activation(z)

    # OR: added variable index to allow subsampling
    # OR: made sample site index  explicit(got None type error when tracing in [predictive])
    # OR: did not investigate the problem in detail by probably i gets tra
    w1 = numpyro.sample(f"w{1}", dist.Normal(0.0, 1).expand((D_Z, D_Z)))
    b1 = numpyro.sample(f"b{1}", dist.Normal(0.0, 1).expand((D_Z,)))
    # OLD version (no subsampling)
    # z = z_p @ w + b
    # z_p = activation(z)

    # Last layer
    # OLD version (no variable index)
    # w = numpyro.sample(f"w{2}", dist.Normal(0.0, 1).expand((D_Z, D_Y)))
    # b = numpyro.sample(f"b{2}", dist.Normal(0.0, 1).expand((D_Y,)))

    # OR: added variable index to allow subsampling
    w2 = numpyro.sample(f"w{2}", dist.Normal(0.0, 1).expand((D_Z, D_Y)))
    b2 = numpyro.sample(f"b{2}", dist.Normal(0.0, 1).expand((D_Y,)))
    # OLD version (no subsampling)
    # z = (z_p @ w + b).flatten() # (N, 1) -> (N,)

    # OLD version (no subsampling in plate)
    # with numpyro.plate("data", N):
    with numpyro.plate(
        "data",
        X.shape[0],
        subsample_size=subsample if subsample is not None else X.shape[0],
    ) as idx:
        # OR: introduce subsampling by index (y=> y_batch and X => x_batch)
        x_batch = X[idx] if len(X.shape) > 1 else X
        y_batch = y[idx] if y is not None and len(y.shape) > 0 else y
        # OR: moved forward computation inside plate to allow subsampling
        z = x_batch @ w0 + b0.flatten()
        z_p = activation(z)
        z = z_p @ w1 + b1
        z_p = activation(z)
        z = (z_p @ w2 + b2).flatten() # (N, 1) -> (N,)
        if y_batch is not None:
            assert z.shape == y_batch.shape, f"Shapes (z,y): {(z.shape, y_batch.shape)}"

        y_loc = numpyro.deterministic("y_loc", z)
        numpyro.sample("y", dist.Normal(y_loc, sigma), obs=y_batch)

def model(x, y=None, depth=2, width=50, D_Y=1, subsample=None):
    # OR: Introduced subsampling 
    prec = numpyro.sample("prec", dist.Gamma(1.0, 0.1))
    _sigma = jnp.sqrt(1 / prec)
    BNN(x, y, depth=depth, width=width, D_Y=D_Y, sigma=_sigma, activation=nn.relu, subsample=subsample)


def UCI_BNN_tanh(X, y=None, depth=2, width=50, D_Y=None):
    prec = numpyro.sample("prec", dist.Gamma(1.0, 0.1))
    _sigma = jnp.sqrt(1 / prec)
    BNN(X, y, depth=depth, width=width, D_Y=D_Y, sigma=_sigma, activation=jnp.tanh)