import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import jax
from jax.numpy.fft import fft, ifft, ifftn, fftn
import jax.nn as nn

@jax.jit
def circ_mult(w,x): # w is a vector
    return jnp.real(ifft(fft(w, axis=-1) * fft(x, axis=-1), axis=-1))

# old
#def circ_mult(w, x):
#    return jnp.real(fft(fft(w) * ifft(x)))

@jax.jit
def expand_circ_mult(w,x): # w has (num_circ, D_X), x has (N, D_X)
    x_fft = fft(x, axis=-1)[..., None, None, :]
    if w.ndim == 4:
        # w has (num_chain, num_sample, num_circ, D_X)
        x_fft = x_fft[..., None, None, :]
    elif w.ndim == 3:
        x_fft = x_fft[..., None, :]
    w_fft = fft(w, axis=-1)
    num_circ = w.shape[-2]
    D_X = w.shape[-1]
    return jnp.real(ifft(x_fft * w_fft, axis=-1)).transpose(0, 3, 1,2).reshape(w.shape[:-2] + (x.shape[0], num_circ * D_X))

circ_vmap = jax.vmap(lambda col, ind: jnp.roll(col, ind), in_axes=(None,0), out_axes=1)
recursive_circ_vmap = jax.vmap(circ_vmap, in_axes=(0, None), out_axes=0)

def CBNN(X, y=None, width=4, sigma=1.0, D_Y=None, activation=jnp.tanh, subsample=None):
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
    w1_vector = numpyro.sample(f"w1", dist.Normal(0, 1).expand((D_Z,)))
    w1 = circ_vmap(w1_vector, jnp.arange(D_Z))
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

def Full_CBNN(X, y=None, width=4, sigma=1.0, D_Y=None, activation=jnp.tanh, subsample=None):
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

    # First layer. Expansion of signal, use ceil(D_Z/D_X) circulant matrices
    num_circ = D_Z // D_X + 1
    w0_vectors = numpyro.sample("w0", dist.Normal(0.0, 1).expand((num_circ, D_X)))
    w0 = recursive_circ_vmap(w0_vectors, jnp.arange(D_X)) # (num_circ, D_X, D_X)
    b0 = numpyro.sample("b0", dist.Normal(0.0, 1).expand((D_Z, )))

    # Middle layers:
    w1_vector = numpyro.sample(f"w1", dist.Normal(0, 1).expand((D_Z,)))
    w1 = circ_vmap(w1_vector, jnp.arange(D_Z))
    b1 = numpyro.sample(f"b1", dist.Normal(0.0, 1).expand((D_Z,)))

    # Last layer
    w2 = numpyro.sample(f"w2", dist.Normal(0.0, 1).expand((D_Z, D_Y)))
    b2 = numpyro.sample(f"b2", dist.Normal(0.0, 1).expand((D_Y,)))

    with numpyro.plate("data", N, subsample_size=subsample if subsample is not None else N) as ind:
        X_batch = X[ind]
        y_batch = y[ind] if y is not None else None

        # Forward pass
        z_0 = jnp.einsum("nx,cxy->ncy", X_batch, w0) # Layer one (N, D_X) @ (num_circ, D_X, D_X) -> (N, num_circ, D_X)
        z_0 = z_0.reshape(X_batch.shape[0], -1) # (N, num_circ * D_X)
        z_0 = z_0[:, :D_Z] # (N, D_Z)
        z_p = activation(z_0 + b0)

        z_p = activation(z_p @ w1 + b1)
        z = (z_p @ w2 + b2).flatten()
        if y_batch is not None:
            assert z.shape == y_batch.shape, f"Shapes (z,y): {(z.shape, y_batch.shape)}"

        y_loc = numpyro.deterministic("y_loc", z)
        numpyro.sample("y", dist.Normal(y_loc, sigma), obs=y_batch)

def Sign_Flipped_CBNN(X, y=None, width=4, sigma=1.0, D_Y=None, activation=jnp.tanh, subsample=None):
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

    # First layer. Expansion of signal, use ceil(D_Z/D_X) circulant matrices
    num_circ = D_Z // D_X + 1
    w0_vectors = numpyro.sample("w0", dist.Normal(0.0, 1).expand((num_circ, D_X)))
    w0 = recursive_circ_vmap(w0_vectors, jnp.arange(D_X)) # (num_circ, D_X, D_X)
    d0 = numpyro.sample("d0", dist.Bernoulli(0.5).expand((D_X, D_X)), infer={"enumerate": "parallel"}) * 2 - 1
    b0 = numpyro.sample("b0", dist.Normal(0.0, 1).expand((D_Z, )))

    # Middle layers:
    w1_vector = numpyro.sample(f"w1", dist.Normal(0, 1).expand((D_Z,)))
    w1 = circ_vmap(w1_vector, jnp.arange(D_Z))
    d1 = numpyro.sample("d1", dist.Bernoulli(0.5).expand((D_Z, D_Z)), infer={"enumerate": "parallel"}) * 2 - 1
    b1 = numpyro.sample(f"b1", dist.Normal(0.0, 1).expand((D_Z,)))

    # Last layer
    w2 = numpyro.sample(f"w2", dist.Normal(0.0, 1).expand((D_Z, D_Y)))
    d2 = numpyro.sample("d2", dist.Bernoulli(0.5).expand((D_Z, D_Z)), infer={"enumerate": "parallel"}) * 2 - 1
    b2 = numpyro.sample(f"b2", dist.Normal(0.0, 1).expand((D_Y,)))

    with numpyro.plate("data", N, subsample_size=subsample if subsample is not None else N) as ind:
        X_batch = X[ind]
        y_batch = y[ind] if y is not None else None

        # Forward pass
        z_0 = jnp.einsum("nx,cxy->ncy", X_batch @ d0, w0) # Layer one (N, D_X) @ (num_circ, D_X, D_X) -> (N, num_circ, D_X)
        z_0 = z_0.reshape(X_batch.shape[0], -1) # (N, num_circ * D_X)
        z_0 = z_0[:, :D_Z] # (N, D_Z)
        z_p = activation(z_0 + b0)

        z_p = activation(z_p @ d1 @ w1 + b1)
        z = (z_p @ d2 @ w2 + b2).flatten()
        if y_batch is not None:
            assert z.shape == y_batch.shape, f"Shapes (z,y): {(z.shape, y_batch.shape)}"

        y_loc = numpyro.deterministic("y_loc", z)
        numpyro.sample("y", dist.Normal(y_loc, sigma), obs=y_batch)

def FFT_CBNN(X, y=None, width=4, sigma=1.0, D_Y=None, activation=jnp.tanh, subsample=None):
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
    w1_vector = numpyro.sample(f"w1", dist.Normal(0, 1).expand((D_Z,)))
    b1 = numpyro.sample(f"b1", dist.Normal(0.0, 1).expand((D_Z,)))

    # Last layer
    w2 = numpyro.sample(f"w2", dist.Normal(0.0, 1).expand((D_Z, D_Y)))
    b2 = numpyro.sample(f"b2", dist.Normal(0.0, 1).expand((D_Y,)))

    with numpyro.plate("data", N, subsample_size=subsample if subsample is not None else N) as ind:
        X_batch = X[ind]
        y_batch = y[ind] if y is not None else None

        # Forward pass
        z_p = activation(X_batch @ w0 + b0)
        z_p = activation(circ_mult(w1_vector, z_p) + b1)
        z = (z_p @ w2 + b2).flatten()
        if y_batch is not None:
            assert z.shape == y_batch.shape, f"Shapes (z,y): {(z.shape, y_batch.shape)}"

        y_loc = numpyro.deterministic("y_loc", z)
        numpyro.sample("y", dist.Normal(y_loc, sigma), obs=y_batch)

def Full_FFT_CBNN(X, y=None, width=4, sigma=1.0, D_Y=None, activation=jnp.tanh, subsample=None):
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

    # First layer. Expansion of signal, use ceil(D_Z/D_X) circulant matrices
    num_circ = D_Z // D_X + 1
    w0_vectors = numpyro.sample("w0", dist.Normal(0.0, 1).expand((num_circ, D_X)))
    b0 = numpyro.sample("b0", dist.Normal(0.0, 1).expand((D_Z, )))

    # Middle layers:
    w1_vector = numpyro.sample(f"w1", dist.Normal(0, 1).expand((D_Z,)))
    b1 = numpyro.sample(f"b1", dist.Normal(0.0, 1).expand((D_Z,)))

    # Last layer
    w2 = numpyro.sample(f"w2", dist.Normal(0.0, 1).expand((D_Z, D_Y)))
    b2 = numpyro.sample(f"b2", dist.Normal(0.0, 1).expand((D_Y,)))

    with numpyro.plate("data", N, subsample_size=subsample if subsample is not None else N) as ind:
        X_batch = X[ind]
        y_batch = y[ind] if y is not None else None

        # Forward pass
        z_p = activation(expand_circ_mult(w0_vectors, X_batch)[:, :D_Z] + b0)
        z_p = activation(circ_mult(w1_vector, z_p) + b1)
        z = (z_p @ w2 + b2).flatten()
        if y_batch is not None:
            assert z.shape == y_batch.shape, f"Shapes (z,y): {(z.shape, y_batch.shape)}"

        y_loc = numpyro.deterministic("y_loc", z)
        numpyro.sample("y", dist.Normal(y_loc, sigma), obs=y_batch)

def UCI_CBNN(X, y=None, width=50, D_Y=None, subsample=None):
    prec = numpyro.sample("prec", dist.Gamma(1.0, 0.1))
    _sigma = jnp.sqrt(1 / prec)
    return CBNN(X, y, width=width, D_Y=D_Y, sigma=_sigma, activation=nn.relu, subsample=subsample)

def UCI_Full_CBNN(X, y=None, width=50, D_Y=None, subsample=None):
    prec = numpyro.sample("prec", dist.Gamma(1.0, 0.1))
    _sigma = jnp.sqrt(1 / prec)
    return Full_CBNN(X, y, width=width, D_Y=D_Y, sigma=_sigma, activation=nn.relu, subsample=subsample)

def UCI_Sign_Flipped_CBNN(X, y=None, width=50, D_Y=None, subsample=None):
    prec = numpyro.sample("prec", dist.Gamma(1.0, 0.1))
    _sigma = jnp.sqrt(1 / prec)
    return Sign_Flipped_CBNN(X, y, width=width, D_Y=D_Y, sigma=_sigma, activation=nn.relu, subsample=subsample)

def UCI_FFT_CBNN(X, y=None, width=50, D_Y=None, subsample=None):
    prec = numpyro.sample("prec", dist.Gamma(1.0, 0.1))
    _sigma = jnp.sqrt(1 / prec)
    return FFT_CBNN(X, y, width=width, D_Y=D_Y, sigma=_sigma, activation=nn.relu, subsample=subsample)

def UCI_Full_FFT_CBNN(X, y=None, width=50, D_Y=None, subsample=None):
    prec = numpyro.sample("prec", dist.Gamma(1.0, 0.1))
    _sigma = jnp.sqrt(1 / prec)
    return Full_FFT_CBNN(X, y, width=width, D_Y=D_Y, sigma=_sigma, activation=nn.relu, subsample=subsample)