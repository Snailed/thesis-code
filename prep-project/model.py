import jax
import jax.random as random
import jax.numpy as jnp
import numpyro
from numpyro import sample
import numpyro.distributions as dist

def SimpleBNN(X, Y=None, D_H=2, D_Y=1, sigma=None):
    N, D_X = X.shape[-2], X.shape[-1]
    w1 = sample("w1", dist.Normal(jnp.zeros((D_X, D_H)), jnp.ones((D_X, D_H)))) # (D_X, D_H)
    z1 = jnp.tanh(X @ w1) # => (D_N, D_H)

    w2 = sample("w2", dist.Normal(jnp.zeros((D_H, D_Y)), jnp.ones((D_H, D_Y)))) # (D_X, D_H)
    z2 = z1 @ w2 # => (D_N, D_H)

    precision_obs = sample("precision_obs", dist.Gamma(3.0,1.0), obs=1/sigma**2 if sigma is not None else None)
    sigma_obs = 1/jnp.sqrt(precision_obs)
    with numpyro.plate("data", N):
        return sample("Y", dist.Normal(z2, sigma_obs).to_event(1), obs=Y)

def SimpleBNN2(X, Y=None, sigma=None):
    return SimpleBNN(X, Y=Y, D_H=20, sigma=sigma)


def SimpleBNN3(X, Y=None, sigma=None):
    return SimpleBNN(X, Y=Y, D_H=40, sigma=sigma)

def SimpleBNN_1_deep(X, Y=None, sigma=None):
    return SimpleBNN(X, Y=Y, D_H=20, sigma=sigma)


def RegularBNN(X, Y=None, D_H=2, D_Y=1, depth=1, sigma=None):
    N, D_X = X.shape[-2], X.shape[-1]
    D_in = D_X
    D_out = D_H
    z_prime = X
    z = X
    for i in range(depth):
        w = sample(f"w{i+1}", dist.Normal(jnp.zeros((D_in, D_out)), jnp.ones((D_in, D_out)))) # (D_X, D_H)
        z_prime = z @ w # => (D_N, D_H)
        z = jnp.tanh(z_prime)

        D_in = D_H
        if i == depth - 1:
            D_out = D_Y

    precision_obs = sample("precision_obs", dist.Gamma(3.0,1.0), obs=1/sigma**2 if sigma is not None else None)
    sigma_obs = 1/jnp.sqrt(precision_obs)
    with numpyro.plate("data", N):
        return sample("Y", dist.Normal(z_prime, sigma_obs).to_event(1), obs=Y)


def rotate(w, r, axis=0):
    """
        Rotates matrix w by r.
        Example:
            >> rotate([a,b,c], 2)
            ... [c, a, b]
    """
    assert r >= 0
    return jnp.roll(w, shift=r, axis=axis)

def expand_roll(w):
    size = w.shape[-1]
    axis = 0 if w.ndim == 1 else 1
    rolled = jnp.array([rotate(w, r, axis=axis) for r in range(size)])
    if len(rolled.shape) > 2:
        rolled = jnp.rollaxis(rolled, 0, rolled.ndim - 1)
    
    return rolled.mT
    #D = w.shape[axis]
    #return jnp.array([w[(i + r) % D] for i in range(D)])

def CirculantBNN(X, Y=None, D_H=2, depth=2, D_Y=1, sigma=None):
    N, D_X = X.shape[-2], X.shape[-1]
    assert depth >= 0
    w_sampled = expand_roll(sample("w", dist.Normal(jnp.zeros(D_H), jnp.ones(D_H))))

    if len(w_sampled.shape) == 2:
        w_sampled = jnp.array([w_sampled])

    if not (w_sampled.shape[-2], w_sampled.shape[-1]) == (D_H, D_H):
        print(w_sampled.shape, (D_H, D_H))
    assert (w_sampled.shape[-2], w_sampled.shape[-1]) == (D_H, D_H)

    D_in = D_X
    D_out = D_H
    z_prime = X
    z = X
    for i in range(depth):
        w = rotate(w_sampled, i, axis=1)[..., :D_in, :D_out]
        z_prime = z @ w
        z = jnp.tanh(z_prime) # => (D_N, D_H)
        D_in = D_H
        if i == depth - 1:
            D_out = D_Y

    precision_obs = sample("precision_obs", dist.Gamma(3.0,1.0), obs=1/sigma**2 if sigma is not None else None)
    sigma_obs = 1/jnp.sqrt(precision_obs)
    with numpyro.plate("data", N):
        return sample("Y", dist.Normal(z_prime, sigma_obs).to_event(1), obs=Y)

def CirculantBNN_medium_width(X, Y=None, sigma=None):
    return CirculantBNN(X, Y=Y, D_H=20, sigma=sigma)

def CirculantBNN_large_width(X, Y=None, sigma=None):
    return CirculantBNN(X, Y=Y, D_H=40, sigma=sigma)

def CirculantBNN_medium_width_3_deep(X, Y=None, sigma=None):
    return CirculantBNN(X, Y=Y, D_H=20, depth=3, sigma=sigma)

def CirculantBNN_medium_width_4_deep(X, Y=None, sigma=None):
    return CirculantBNN(X, Y=Y, D_H=20, depth=4, sigma=sigma)

def Exp1Reg_1_deep(X, Y=None, sigma=None):
    return RegularBNN(X, Y=Y, D_H=32, D_Y=1, depth=1, sigma=sigma)

def Exp1Reg_2_deep(X, Y=None, sigma=None):
    return RegularBNN(X, Y=Y, D_H=16, D_Y=1, depth=2, sigma=sigma)

def Exp1Reg_4_deep(X, Y=None, sigma=None):
    return RegularBNN(X, Y=Y, D_H=8, D_Y=1, depth=4, sigma=sigma)

def Exp1Reg_8_deep(X, Y=None, sigma=None):
    return RegularBNN(X, Y=Y, D_H=4, D_Y=1, depth=8, sigma=sigma)

def Exp1Circ_1_deep(X, Y=None, sigma=None):
    return CirculantBNN(X, Y=Y, D_H=32, depth=1, sigma=sigma)

def Exp1Circ_2_deep(X, Y=None, sigma=None):
    return CirculantBNN(X, Y=Y, D_H=32, depth=2, sigma=sigma)

def Exp1Circ_3_deep(X, Y=None, sigma=None):
    return CirculantBNN(X, Y=Y, D_H=32, depth=3, sigma=sigma)

def Exp1Circ_4_deep(X, Y=None, sigma=None):
    return CirculantBNN(X, Y=Y, D_H=32, depth=4, sigma=sigma)

def Exp1Circ_8_deep(X, Y=None, sigma=None):
    return CirculantBNN(X, Y=Y, D_H=32, depth=4, sigma=sigma)



def Exp2Reg_4_deep_20_wide(X, Y=None, sigma=None):
    return RegularBNN(X, Y=Y, D_H=20, D_Y=1, depth=4, sigma=sigma)

def Exp2Circ_4_deep_20_wide(X, Y=None, sigma=None):
    return RegularBNN(X, Y=Y, D_H=20, D_Y=1, depth=4, sigma=sigma)