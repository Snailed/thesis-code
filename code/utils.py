import jax.numpy as jnp
def normalize(val, mean, std):
    """Normalize data to zero mean, unit variance"""
    if mean is None and std is None:
        # Only use training data to estimate mean and std.
        std = jnp.std(val, 0, keepdims=True)
        std = jnp.where(std == 0, 1.0, std)
        mean = jnp.mean(val, 0, keepdims=True)
    return (val - mean) / std, mean, std

def check_test_normed(x):
    # x (from test) is centered with respect to train so x.mean(1) isn't
    # necessarily close to 0.
    assert jnp.abs(jnp.mean(x)).item() < 1.0, "Test x far from centered train x."
    # x (from test) is standardized with respect to train so x.std(1) isn't
    # necessarily close to 1.
    assert jnp.std(x).item() < 2.0, "Test x std large compared to train x."