import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import jax
from jax.numpy.fft import fft, ifft, ifftn, fftn
import jax.nn as nn

@jax.jit
def circ_mult(w_hat,x): # w is a vector
    return jnp.real(fft(w_hat * ifft(x)))
def is_hermitian(v):
    v2 = v[1:]
    res = v2 - v2[::-1].conj()
    return jnp.allclose(jnp.imag(res), 0, atol=1e-5) and jnp.allclose(jnp.real(res), 0, atol=1e-5)
def S(k, alpha=1e2, beta=1e5):
    return beta*jnp.exp(-(1/alpha) * k**2)

def sample_w_hat(i: int, n: int, S=S, alpha=1e2, beta=1e5):
    w_hat = jnp.zeros(n, dtype=jnp.complex64)
    normals = numpyro.sample("w_hat_normals", dist.Normal(0, 1).expand((n,)))
    w_hat = w_hat.at[0].set(normals[0]) # w^ <- N(0, 1)
    for k in range(1, n//2):
        real = normals[2*k] * S(k, alpha=alpha, beta=beta) * 0.5
        img = normals[2*k+1] * S(k, alpha=alpha, beta=beta) * 0.5
        w_hat = w_hat.at[k].set(real + 1j * img)
        w_hat = w_hat.at[n-k].set(real - 1j * img)
    if n % 2 == 0:
        w_hat = w_hat.at[n//2].set(normals[n-1] * S(n//2))# Nyquist
    #assert is_hermitian(w_hat), "The generated vector is not hermitian"
    #assert jnp.allclose(jnp.imag(jnp.fft.ifft(w_hat)), 0, atol=1e-3)
    w_hat = numpyro.deterministic("w_hat", w_hat / jnp.sqrt(n))
    return w_hat

def Spectral_BNN(X, y=None, depth=1, width=4, sigma=1.0, D_Y=None, activation=jnp.tanh):
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

    # First layer
    w = numpyro.sample("w0", dist.Normal(0, 1).expand((D_X, D_Z)))
    b = numpyro.sample("b0", dist.Normal(0, 1).expand((D_Z,)))
    z = X @ w + b
    z_p = activation(z)

    # Middle layers:
    for i in range(1, depth):
        #w_vector = numpyro.sample(f"w{i}", dist.Normal(0, 1).expand((D_Z,)))
        w_hat = sample_w_hat(i=i, n=D_Z)

        b = numpyro.sample(f"b{i}", dist.Normal(0, 1).expand((D_Z,)))
        z = circ_mult(w_hat, z_p) + b
        z_p = activation(z)

    # Last layer
    w = numpyro.sample(f"w{depth}", dist.Normal(0, 1).expand((D_Z, D_Y)))
    b = numpyro.sample(f"b{depth}", dist.Normal(0, 1).expand((D_Y,)))
    z = z_p @ w + b
    if y is not None:
        assert z.shape == y.shape
    else:
        assert z.shape[-1] == D_Y
    with numpyro.plate("data", N):
        y_loc = numpyro.deterministic("y_loc", z)
        return numpyro.sample("y", dist.Normal(y_loc, sigma).to_event(1), obs=y)

def UCI_Spectral_BNN(X, y=None, depth=2, width=50, D_Y=None):
    prec = numpyro.sample("prec", dist.Gamma(1.0, 0.1))
    _sigma = jnp.sqrt(1 / prec)
    Spectral_BNN(X, y, depth=depth, width=width, D_Y=D_Y, sigma=_sigma, activation=nn.relu)