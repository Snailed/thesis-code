import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import nn
import jax
from jax.numpy.fft import fft, ifft
from functools import partial

def ECG_BNN(X, y=None, width=4, subsample=None, prior_probs=None):
    N = X.shape[-2]
    D_X = X.shape[-1]
    D_Z = width

    # First layer
    w0 = numpyro.sample("w0", dist.Normal(0.0, 1).expand((D_X, D_Z)))
    b0 = numpyro.sample("b0", dist.Normal(0.0, 1).expand((D_Z, )))

    # Middle layers:
    w1 = numpyro.sample(f"w1", dist.Normal(0.0, 1).expand((D_Z, D_Z)))
    b1 = numpyro.sample(f"b1", dist.Normal(0.0, 1).expand((D_Z,)))

    w2 = numpyro.sample(f"w2", dist.Normal(0.0, 1).expand((D_Z, D_Z)))
    b2 = numpyro.sample(f"b2", dist.Normal(0.0, 1).expand((D_Z,)))

    w3 = numpyro.sample(f"w3", dist.Normal(0.0, 1).expand((D_Z, 5)))
    b3 = numpyro.sample(f"b3", dist.Normal(0.0, 1).expand((5,)))

    with numpyro.plate("data", N, subsample_size=subsample if subsample is not None else N) as ind:
        X_batch = X[ind]
        y_batch = y[ind] if y is not None else None

        # Forward pass
        z_p = nn.tanh(X_batch @ w0 + b0)
        z_p = nn.tanh(z_p @ w1 + b1)
        z_p = nn.tanh(z_p @ w2 + b2)

        z = (z_p @ w3 + b3)
        if y_batch is not None:
            assert z.shape == (subsample if subsample else N, 5), f"Shapes z: {z.shape}"

        # if prior_probs is not None:
        #     # Use prior_probs as the prior for the output layer
        #     assert prior_probs.shape == (5,), f"Shapes prior_probs: {prior_probs.shape}"
        #     z = nn.softmax(z) * prior_probs
        #     z /= jnp.sum(z, axis=-1, keepdims=True)
        y_probs = numpyro.deterministic("y_probs", z)
        numpyro.sample("y", dist.Categorical(logits=y_probs), obs=y_batch.astype(jnp.int32) if y_batch is not None else None)

# @jax.jit
# def circ_mult(w,x): # w is a vector
#     return jnp.real(fft(fft(w) * ifft(x)))
@jax.jit
def circ_mult(w,x): # w is a vector
    return jnp.real(ifft(fft(w, axis=-1) * fft(x, axis=-1), axis=-1))

# @jax.jit
# def expand_circ_mult(w,x): # w has (num_circ, D_X), x has (N, D_X)
#     x_fft = ifft(x)
#     x_fft = jnp.repeat(x_fft[:, None, :], w.shape[0], axis=1)
#     return jnp.real(fft(fft(w) * x_fft)).reshape(x.shape[0], -1) # (N, num_circ * D_X)

# Performs W @ x.T
@jax.jit
def expand_circ_mult(w,x): # w has (num_circ, D_X), x has (N, D_X)
    x_fft = fft(x, axis=-1)[..., None, None, :]
    w_fft = fft(w, axis=-1)
    return jnp.real(ifft(x_fft * w_fft, axis=-1)).transpose(1, 2,3,0)

circ_vmap = jax.vmap(lambda col, ind: jnp.roll(col, ind), in_axes=(None,0), out_axes=1)
recursive_circ_vmap = jax.vmap(circ_vmap, in_axes=(0, None), out_axes=0)

def ECG_CBNN(X, y=None, width=4, subsample=None, prior_probs=None):
    N = X.shape[-2]
    D_X = X.shape[-1]
    D_Z = width

    # # First layer
    # if D_X % D_Z == 0:
    #     num_circ = D_X // D_Z
    # else:
    #     num_circ = D_X // D_Z + 1
    # w0_vectors = numpyro.sample("w0", dist.Normal(0.0, 1).expand((num_circ, D_X)))
    w0_vector = numpyro.sample("w0", dist.Normal(0.0, 1).expand((D_X,)))
    #w0 = recursive_circ_vmap(w0_vectors, jnp.arange(D_X)) # (num_circ, D_X, D_X)
    b0 = numpyro.sample("b0", dist.Normal(0.0, 1).expand((D_Z, )))

    # Middle layers:
    w1_vector = numpyro.sample(f"w1", dist.Normal(0, 1).expand((D_Z,)))
    #w1 = circ_vmap(w1_vector, jnp.arange(D_Z))
    b1 = numpyro.sample(f"b1", dist.Normal(0.0, 1).expand((D_Z,)))

    w2_vector = numpyro.sample(f"w2", dist.Normal(0, 1).expand((D_Z,)))
    #w2 = circ_vmap(w2_vector, jnp.arange(D_Z))
    b2 = numpyro.sample(f"b2", dist.Normal(0.0, 1).expand((D_Z,)))

    # Last layer
    w3 = numpyro.sample(f"w3", dist.Normal(0.0, 1).expand((D_Z, 5)))
    b3 = numpyro.sample(f"b3", dist.Normal(0.0, 1).expand((5,)))

    with numpyro.plate("data", N, subsample_size=subsample if subsample is not None else N) as ind:
        X_batch = X[ind]
        y_batch = y[ind] if y is not None else None

        # Forward pass
        z_p = nn.relu(circ_mult(w0_vector, X_batch)[...,:D_Z] + b0)
        z_p = nn.relu(circ_mult(w1_vector, z_p) + b1)
        z_p = nn.relu(circ_mult(w2_vector, z_p) + b2)

        z = (z_p @ w3 + b3)
        if z.ndim == 3:
            z = z[0]
        if y_batch is not None:
            assert z.shape == (subsample if subsample else N, 5), f"Shapes z: {z.shape}"

        if prior_probs is not None:
            # Use prior_probs as the prior for the output layer
            assert prior_probs.shape == (5,), f"Shapes prior_probs: {prior_probs.shape}"
            z = nn.softmax(z) * prior_probs
            z /= jnp.sum(z, axis=-1, keepdims=True)
            y_probs = numpyro.deterministic("y_probs", z)
            numpyro.sample("y", dist.Categorical(probs=y_probs), obs=y_batch.astype(jnp.int32) if y_batch is not None else None)
        else:
            y_probs = numpyro.deterministic("y_probs", z)
            numpyro.sample("y", dist.Categorical(logits=y_probs), obs=y_batch.astype(jnp.int32) if y_batch is not None else None)




def is_hermitian(v):
    v2 = v[1:]
    res = v2 - v2[::-1].conj()
    return jnp.allclose(jnp.imag(res), 0, atol=1e-5) and jnp.allclose(jnp.real(res), 0, atol=1e-5)
def S(k, alpha=1e2, beta=1e5):
    return beta*jnp.exp(-(1/alpha) * k**2)

@jax.jit
def spectral_circ_mult(w_hat,x): # w is a vector
    return jnp.real(ifft(w_hat * fft(x)))

# @jax.jit
# def spectral_expand_circ_mult(w_hat,x): # w has (num_circ, D_X), x has (N, D_X)
#     x_fft = ifft(x)
#     x_fft = jnp.repeat(x_fft[:, None, :], w_hat.shape[0], axis=1)
#     return jnp.real(fft(w_hat * x_fft)).reshape(x.shape[0], -1) # (N, num_circ * D_X)


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
    #assert is_hermitian(w_hat), "The generated vector is not hermitian"
    #assert jnp.allclose(jnp.imag(jnp.fft.ifft(w_hat)), 0, atol=1e-3)
    w_hat = numpyro.deterministic(f"w_hat_{i}", w_hat / jnp.sqrt(n))
    return w_hat

def ECG_Spectral_BNN(X, y=None, width=4, subsample=None, prior_probs=None):
    N = X.shape[-2]
    D_X = X.shape[-1]
    D_Z = width

    # Hyperprior, sample 4 deep cond. independent a,b
    alpha = numpyro.sample("alpha0", dist.Gamma(5.0, 0.5).expand((3,)))
    beta = numpyro.sample("beta0", dist.Gamma(50.0, 0.5).expand((3,)))
    #alpha = jnp.array([1e1, 1e1, 1e1, 1e1])
    #beta = jnp.array([1e2, 1e2, 1e2, 1e2])

    # First layer, assumes D_X > D_Z
    w0_hat = sample_w_hat(i=0, n=D_X, alpha=alpha[0], beta=beta[0])
    b0 = numpyro.sample("b0", dist.Normal(0.0, 1).expand((D_Z, )))

    # Middle layers:
    w1_hat = sample_w_hat(i=1, n=D_Z, alpha=alpha[1], beta=beta[1])
    b1 = numpyro.sample(f"b1", dist.Normal(0.0, 1).expand((D_Z,)))

    w2_hat = sample_w_hat(i=2, n=D_Z, alpha=alpha[2], beta=beta[2])
    b2 = numpyro.sample(f"b2", dist.Normal(0.0, 1).expand((D_Z,)))

    # Last layer
    w4 = numpyro.sample("w3", dist.Normal(0.0, 1).expand((D_Z, 5)))
    b4 = numpyro.sample(f"b3", dist.Normal(0.0, 1).expand((5,)))


    with numpyro.plate("data", N, subsample_size=subsample if subsample is not None else N) as ind:
        X_batch = X[ind]
        y_batch = y[ind] if y is not None else None

        # Forward pass
        z_p = spectral_circ_mult(w0_hat, X_batch)[:, :D_Z]
        z_p = nn.relu(z_p + b0)
        z_p = nn.relu(spectral_circ_mult(w1_hat, z_p) + b1)
        z_p = nn.relu(spectral_circ_mult(w2_hat, z_p) + b2)
        # z_p = nn.relu(circ_mult(w3_hat, z_p) + b3)
        z = z_p @ w4 + b4

        if z.ndim == 3:
            z = z[0]
        if y_batch is not None:
            assert z.shape == (subsample if subsample else N, 5), f"Shapes z: {z.shape}"

        if prior_probs is not None:
            # Use prior_probs as the prior for the output layer
            assert prior_probs.shape == (5,), f"Shapes prior_probs: {prior_probs.shape}"
            #assert jnp.all(jnp.isfinite(z)), "z contains non-finite values before softmax"
            z = nn.softmax(z) * prior_probs
            z /= jnp.sum(z, axis=-1, keepdims=True)
            y_probs = numpyro.deterministic("y_probs", z)
            numpyro.sample("y", dist.Categorical(probs=y_probs), obs=y_batch.astype(jnp.int32) if y_batch is not None else None)
        else:
            y_probs = numpyro.deterministic("y_probs", z)
            numpyro.sample("y", dist.Categorical(logits=y_probs), obs=y_batch.astype(jnp.int32) if y_batch is not None else None)





def ECG_Small_CBNN(X, y=None, width=4, subsample=None, prior_probs=None):
    N = X.shape[-2]
    D_X = X.shape[-1]
    D_Z = width

    # # First layer
    # if D_X % D_Z == 0:
    #     num_circ = D_X // D_Z
    # else:
    #     num_circ = D_X // D_Z + 1
    # w0_vectors = numpyro.sample("w0", dist.Normal(0.0, 1).expand((num_circ, D_X)))
    w0_vector = numpyro.sample("w0", dist.Normal(0.0, 1).expand((D_X,)))
    #w0 = recursive_circ_vmap(w0_vectors, jnp.arange(D_X)) # (num_circ, D_X, D_X)
    b0 = numpyro.sample("b0", dist.Normal(0.0, 1).expand((D_Z, )))

    # Middle layers:
    w1_vector = numpyro.sample(f"w1", dist.Normal(0, 1).expand((D_Z,)))
    #w1 = circ_vmap(w1_vector, jnp.arange(D_Z))
    b1 = numpyro.sample(f"b1", dist.Normal(0.0, 1).expand((D_Z,)))

    # Last layer
    w2 = numpyro.sample(f"w4", dist.Normal(0.0, 1).expand((D_Z, 5)))
    b2 = numpyro.sample(f"b4", dist.Normal(0.0, 1).expand((5,)))

    with numpyro.plate("data", N, subsample_size=subsample if subsample is not None else N) as ind:
        X_batch = X[ind]
        y_batch = y[ind] if y is not None else None

        # Forward pass
        z_p = nn.relu(circ_mult(w0_vector, X_batch)[...,:D_Z] + b0)
        z_p = nn.relu(circ_mult(w1_vector, z_p) + b1)

        z = (z_p @ w2 + b2)
        if z.ndim == 3:
            z = z[0]
        if y_batch is not None:
            assert z.shape == (subsample if subsample else N, 5), f"Shapes z: {z.shape}"

        if prior_probs is not None:
            # Use prior_probs as the prior for the output layer
            assert prior_probs.shape == (5,), f"Shapes prior_probs: {prior_probs.shape}"
            z = nn.softmax(z) * prior_probs
            z /= jnp.sum(z, axis=-1, keepdims=True)
            y_probs = numpyro.deterministic("y_probs", z)
            numpyro.sample("y", dist.Categorical(probs=y_probs), obs=y_batch.astype(jnp.int32) if y_batch is not None else None)
        else:
            y_probs = numpyro.deterministic("y_probs", z)
            numpyro.sample("y", dist.Categorical(logits=y_probs), obs=y_batch.astype(jnp.int32) if y_batch is not None else None)


def ECG_CBNN_tanh(X, y=None, width=4, subsample=None, prior_probs=None):
    N = X.shape[-2]
    D_X = X.shape[-1]
    D_Z = width

    # # First layer
    w0_vector = numpyro.sample("w0", dist.Normal(0.0, 1).expand((D_X,)))
    b0 = numpyro.sample("b0", dist.Normal(0.0, 1).expand((D_Z, )))

    # Middle layers:
    w1_vector = numpyro.sample(f"w1", dist.Normal(0, 1).expand((D_Z,)))
    b1 = numpyro.sample(f"b1", dist.Normal(0.0, 1).expand((D_Z,)))

    w2_vector = numpyro.sample(f"w2", dist.Normal(0, 1).expand((D_Z,)))
    b2 = numpyro.sample(f"b2", dist.Normal(0.0, 1).expand((D_Z,)))

    # Last layer
    w3 = numpyro.sample(f"w3", dist.Normal(0.0, 1).expand((D_Z, 5)))
    b3 = numpyro.sample(f"b3", dist.Normal(0.0, 1).expand((5,)))

    with numpyro.plate("data", N, subsample_size=subsample if subsample is not None else N) as ind:
        X_batch = X[ind]
        y_batch = y[ind] if y is not None else None

        # Forward pass
        z_p = nn.tanh(circ_mult(w0_vector, X_batch)[...,:D_Z] + b0)
        z_p = nn.tanh(circ_mult(w1_vector, z_p) + b1)
        z_p = nn.tanh(circ_mult(w2_vector, z_p) + b2)

        z = (z_p @ w3 + b3)
        if z.ndim == 3:
            z = z[0]
        if y_batch is not None:
            assert z.shape == (subsample if subsample else N, 5), f"Shapes z: {z.shape}"

        y_probs = numpyro.deterministic("y_probs", z)
        numpyro.sample("y", dist.Categorical(logits=y_probs), obs=y_batch.astype(jnp.int32) if y_batch is not None else None)



def ECG_Spectral_BNN_tanh(X, y=None, width=4, subsample=None, prior_probs=None):
    N = X.shape[-2]
    D_X = X.shape[-1]
    D_Z = width

    # Hyperprior, sample 4 deep cond. independent a,b
    alpha = numpyro.sample("alpha0", dist.Gamma(5.0, 0.5).expand((3,)))
    beta = numpyro.sample("beta0", dist.Gamma(50.0, 0.5).expand((3,)))
    #alpha = jnp.array([1e1, 1e1, 1e1, 1e1])
    #beta = jnp.array([1e2, 1e2, 1e2, 1e2])

    # First layer, assumes D_X > D_Z
    w0_hat = sample_w_hat(i=0, n=D_X, alpha=alpha[0], beta=beta[0])
    b0 = numpyro.sample("b0", dist.Normal(0.0, 1).expand((D_Z, )))

    # Middle layers:
    w1_hat = sample_w_hat(i=1, n=D_Z, alpha=alpha[1], beta=beta[1])
    b1 = numpyro.sample(f"b1", dist.Normal(0.0, 1).expand((D_Z,)))

    w2_hat = sample_w_hat(i=2, n=D_Z, alpha=alpha[2], beta=beta[2])
    b2 = numpyro.sample(f"b2", dist.Normal(0.0, 1).expand((D_Z,)))

    # Last layer
    w4 = numpyro.sample("w3", dist.Normal(0.0, 1).expand((D_Z, 5)))
    b4 = numpyro.sample(f"b3", dist.Normal(0.0, 1).expand((5,)))


    with numpyro.plate("data", N, subsample_size=subsample if subsample is not None else N) as ind:
        X_batch = X[ind]
        y_batch = y[ind] if y is not None else None

        # Forward pass
        z_p = spectral_circ_mult(w0_hat, X_batch)[:, :D_Z]
        z_p = nn.tanh(z_p + b0)
        z_p = nn.tanh(spectral_circ_mult(w1_hat, z_p) + b1)
        z_p = nn.tanh(spectral_circ_mult(w2_hat, z_p) + b2)
        # z_p = nn.relu(circ_mult(w3_hat, z_p) + b3)
        z = z_p @ w4 + b4

        if z.ndim == 3:
            z = z[0]
        if y_batch is not None:
            assert z.shape == (subsample if subsample else N, 5), f"Shapes z: {z.shape}"

        if prior_probs is not None:
            # Use prior_probs as the prior for the output layer
            assert prior_probs.shape == (5,), f"Shapes prior_probs: {prior_probs.shape}"
            #assert jnp.all(jnp.isfinite(z)), "z contains non-finite values before softmax"
            z = nn.softmax(z) * prior_probs
            z /= jnp.sum(z, axis=-1, keepdims=True)
            y_probs = numpyro.deterministic("y_probs", z)
            numpyro.sample("y", dist.Categorical(probs=y_probs), obs=y_batch.astype(jnp.int32) if y_batch is not None else None)
        else:
            y_probs = numpyro.deterministic("y_probs", z)
            numpyro.sample("y", dist.Categorical(logits=y_probs), obs=y_batch.astype(jnp.int32) if y_batch is not None else None)