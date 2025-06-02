import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import jax
from jax.numpy.fft import fft, ifft, ifftn, fftn
import jax.nn as nn

@jax.jit
def circ_mult(w_hat,x): # w is a vector
    return jnp.real(ifft(w_hat * fft(x, axis=-1), axis=-1))

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

def is_hermitian(v):
    v2 = v[1:]
    res = v2 - v2[::-1].conj()
    return jnp.allclose(jnp.imag(res), 0, atol=1e-5) and jnp.allclose(jnp.real(res), 0, atol=1e-5)
def S(k, alpha=1e2, beta=1e5):
    return beta*jnp.exp(-(1/alpha) * k**2)

#def sample_w_hat(i: int, n: int, S=S, alpha=1e2, beta=1e5):
#    w_hat = jnp.zeros(n, dtype=jnp.complex64)
#
#    normals = numpyro.sample(f"w_hat_{i}_normals", dist.Normal(0.0, 1.0).expand((n,)))
#    w_hat = w_hat.at[0].set(normals[0]) # w^ <- N(0, 1)
#    for k in range(1, n//2):
#        real = normals[2*k] * S(k, alpha=alpha, beta=beta) * 0.5
#        img = normals[2*k+1] * S(k, alpha=alpha, beta=beta) * 0.5
#        w_hat = w_hat.at[k].set(real + 1j * img)
#        w_hat = w_hat.at[n-k].set(real - 1j * img)
#    if n % 2 == 0:
#        w_hat = w_hat.at[n//2].set(normals[n-1] * S(n//2))# Nyquist
#    #assert is_hermitian(w_hat), "The generated vector is not hermitian"
#    #assert jnp.allclose(jnp.imag(jnp.fft.ifft(w_hat)), 0, atol=1e-3)
#    w_hat = numpyro.deterministic(f"w_hat_{i}", w_hat / jnp.sqrt(n))
#    return w_hat

@jax.jit
def inner_sample_w_hat(normals, S_k):
    n = normals.shape[-1]
    k = jnp.arange(n)

    real = normals[2 * k] * S_k * 0.5
    img = normals[2 * k + 1] * S_k * 0.5

    w_hat_real = jnp.where((k > 0) & (k < n // 2), real, 0.0)
    w_hat_img = jnp.where((k > 0) & (k < n // 2), img, 0.0)

    w_hat = w_hat_real + 1j * w_hat_img
    w_hat = w_hat.at[0].set(normals[0] * S_k[0])
    w_hat = w_hat.at[n // 2].set(jnp.where(n % 2 == 0, normals[n - 1] * S_k[n // 2], 0.0))
    w_hat = w_hat.at[n - k].set(jnp.where((k > 0) & (k < n // 2), w_hat_real - 1j * w_hat_img, w_hat[n - k]))

    return w_hat

def sample_w_hat(i: int, n: int, S=S, alpha=1e2, beta=1e5):
    w_hat = jnp.zeros(n, dtype=jnp.complex64)

    normals = numpyro.sample(f"w_hat_{i}_normals", dist.Normal(0.0, 1.0).expand((n,)))
    S_k = S(jnp.arange(n), alpha=alpha, beta=beta)
    w_hat = inner_sample_w_hat(normals, S_k)
    w_hat = numpyro.deterministic(f"w_hat_{i}", w_hat / jnp.sqrt(n))
    return w_hat

def Spectral_BNN(X, y=None, width=4, sigma=1.0, D_Y=None, activation=jnp.tanh, subsample=None):
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
    alpha0 = numpyro.sample("alpha1", dist.Gamma(1.0, 0.1))
    beta0 = numpyro.sample("beta1", dist.Gamma(10.0, 0.1))
    w1_hat = sample_w_hat(i=0, n=D_Z, alpha=alpha0, beta=beta0)
    #w1 = numpyro.sample(f"w1", dist.Normal(0.0, 1).expand((D_Z, D_Z)))
    b1 = numpyro.sample(f"b1", dist.Normal(0.0, 1).expand((D_Z,)))

    # Last layer
    w2 = numpyro.sample(f"w2", dist.Normal(0.0, 1).expand((D_Z, D_Y)))
    b2 = numpyro.sample(f"b2", dist.Normal(0.0, 1).expand((D_Y,)))

    with numpyro.plate("data", N, subsample_size=subsample if subsample is not None else N) as ind:
        X_batch = X[ind]
        y_batch = y[ind] if y is not None else None

        # Forward pass
        z_p = activation(X_batch @ w0 + b0)
        z_p = activation(circ_mult(w1_hat, z_p) + b1)
        z = (z_p @ w2 + b2).flatten()
        if y_batch is not None:
            assert z.shape == y_batch.shape, f"Shapes (z,y): {(z.shape, y_batch.shape)}"

        y_loc = numpyro.deterministic("y_loc", z)
        numpyro.sample("y", dist.Normal(y_loc, sigma), obs=y_batch)

def Full_Spectral_BNN(X, y=None, width=4, sigma=1.0, D_Y=None, activation=jnp.tanh, subsample=None):
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
    #w0 = numpyro.sample("w0", dist.Normal(0.0, 1).expand((D_X, D_Z)))
    num_circ = D_Z // D_X + 1
    alpha0 = numpyro.sample("alpha0", dist.Gamma(1.0, 0.1))
    beta0 = numpyro.sample("beta0", dist.Gamma(10.0, 0.1))
    w0_vectors = jnp.array([sample_w_hat(i, D_X, alpha=alpha0, beta=beta0) for i in range(num_circ)])
    b0 = numpyro.sample("b0", dist.Normal(0.0, 1).expand((D_Z, )))

    # Middle layers:
    alpha1 = numpyro.sample("alpha1", dist.Gamma(1.0, 0.1))
    beta1 = numpyro.sample("beta1", dist.Gamma(10.0, 0.1))
    w1_hat = sample_w_hat(i=num_circ, n=D_Z, alpha=alpha1, beta=beta1)
    #w1 = numpyro.sample(f"w1", dist.Normal(0.0, 1).expand((D_Z, D_Z)))
    b1 = numpyro.sample(f"b1", dist.Normal(0.0, 1).expand((D_Z,)))

    # Last layer
    w2 = numpyro.sample("w2", dist.Normal(0.0, 1).expand((D_Z, D_Y)))
    b2 = numpyro.sample(f"b2", dist.Normal(0.0, 1).expand((D_Y,)))

    with numpyro.plate("data", N, subsample_size=subsample if subsample is not None else N) as ind:
        X_batch = X[ind]
        y_batch = y[ind] if y is not None else None

        # Forward pass
        z_p = activation(expand_circ_mult(w0_vectors, X_batch)[:, :D_Z] + b0)
        z_p = activation(circ_mult(w1_hat, z_p) + b1)
        z = (z_p @ w2 + b2).flatten()
        if y_batch is not None:
            assert z.shape == y_batch.shape, f"Shapes (z,y): {(z.shape, y_batch.shape)}"

        y_loc = numpyro.deterministic("y_loc", z)
        numpyro.sample("y", dist.Normal(y_loc, sigma), obs=y_batch)

def UCI_Spectral_BNN(X, y=None, width=50, D_Y=None, subsample=None):
    prec = numpyro.sample("prec", dist.Gamma(1.0, 0.1))
    _sigma = jnp.sqrt(1 / prec)
    Spectral_BNN(X, y, width=width, D_Y=D_Y, sigma=_sigma, activation=nn.relu, subsample=subsample)


def UCI_Full_Spectral_BNN(X, y=None, width=50, D_Y=None, subsample=None):
    prec = numpyro.sample("prec", dist.Gamma(1.0, 0.1))
    _sigma = jnp.sqrt(1 / prec)
    Full_Spectral_BNN(X, y, width=width, D_Y=D_Y, sigma=_sigma, activation=nn.relu, subsample=subsample)
