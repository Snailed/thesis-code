import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import jax
from jax.numpy.fft import fft, ifft

@jax.jit
def circ_mult(w,x): # w is a vector
    return jnp.real(fft(ifft(x.T)*fft(w)).T)

def CBNN(X, y=None, depth=1, width=4, sigma=1.0, D_Y=None, activation=jnp.tanh):
    # Make sure D_Y is defined
    if y is None and D_Y is None:
        raise ValueError("Either y or D_Y must be provided.")
    if y is not None:
        if y.ndim == 1:
            y = y[..., None]
        D_Y = y.shape[-1]

    N = X.shape[-2]
    D_X = X.shape[-1]
    D_Z = width
    if depth == 1:
        D_Z = D_Y

    # Circulant matrix vmap
    circ_vmap = jax.vmap(lambda col, ind: jnp.roll(col, ind), in_axes=(None,0), out_axes=1)

    # First layer
    w = numpyro.sample("w0", dist.Normal(0, 1).expand((D_X, D_Z)))
    b = numpyro.sample("b0", dist.Normal(0, 1).expand((D_Z,)))
    z = X @ w + b
    z_p = activation(z)

    # Middle layers:
    for i in range(1, depth):
        w_vector = numpyro.sample(f"w{i}", dist.Normal(0, 1).expand((D_Z,)))
        w = circ_vmap(w_vector.reshape(1,-1), jnp.arange(D_Z))

        b = numpyro.sample(f"b{i}", dist.Normal(0, 1).expand((D_Z,)))
        z = z_p @ w + b
        z_p = activation(z)

    # Last layer
    w = numpyro.sample(f"w{depth}", dist.Normal(0, 1).expand((D_Z, D_Y)))
    b = numpyro.sample(f"b{depth}", dist.Normal(0, 1).expand((D_Y,)))
    z = z_p @ w + b
    with numpyro.plate("data", N):
        return numpyro.sample("y", dist.Normal(z, sigma).to_event(1), obs=y)

def FFT_CBNN(X, y=None, depth=1, width=4, sigma=1.0, D_Y=None, activation=jnp.tanh):
    # Make sure D_Y is defined
    if y is None and D_Y is None:
        raise ValueError("Either y or D_Y must be provided.")
    if y is not None:
        if y.ndim == 1:
            y = y[..., None]
        D_Y = y.shape[-1]

    N = X.shape[-2]
    D_X = X.shape[-1]
    D_Z = width
    if depth == 1:
        D_Z = D_Y

    # Circulant matrix vmap
    circ_vmap = jax.vmap(lambda col, ind: jnp.roll(col, ind), in_axes=(None,0), out_axes=1)

    # First layer
    w = numpyro.sample("w0", dist.Normal(0, 1).expand((D_X, D_Z)))
    b = numpyro.sample("b0", dist.Normal(0, 1).expand((D_Z,)))
    z = X @ w + b
    z_p = activation(z)

    # Middle layers:
    for i in range(1, depth):
        w_vector = numpyro.sample(f"w{i}", dist.Normal(0, 1).expand((D_Z,)))
        #w = circ_vmap(w_vector.reshape(1,-1), jnp.arange(D_Z))

        b = numpyro.sample(f"b{i}", dist.Normal(0, 1).expand((D_Z,)))
        z = jnp.matrix_transpose(circ_mult(w_vector, z_p.T)) + b
        z_p = activation(z)

    # Last layer
    w = numpyro.sample(f"w{depth}", dist.Normal(0, 1).expand((D_Z, D_Y)))
    b = numpyro.sample(f"b{depth}", dist.Normal(0, 1).expand((D_Y,)))
    z = z_p @ w + b
    with numpyro.plate("data", N):
        return numpyro.sample("y", dist.Normal(z, sigma).to_event(1), obs=y)

def UCI_CBNN(X, y=None, depth=2, width=50, D_Y=None):
    prec = numpyro.sample("prec", dist.Gamma(1.0, 0.1))
    _sigma = jnp.sqrt(1 / prec)
    return CBNN(X, y, depth=depth, width=width, D_Y=D_Y, sigma=_sigma, activation=jnp.tanh)

def UCI_FFT_CBNN(X, y=None, depth=2, width=50, D_Y=None):
    prec = numpyro.sample("prec", dist.Gamma(1.0, 0.1))
    _sigma = jnp.sqrt(1 / prec)
    return FFT_CBNN(X, y, depth=depth, width=width, D_Y=D_Y, sigma=_sigma, activation=jnp.tanh)