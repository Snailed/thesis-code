import numpyro
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist
import jax.numpy as jnp
import jax
import arviz as az
import os
from time import time

def run_hmc(model, dataset, split, args, post_warmup_state=None):
    kernel = NUTS(model, init_strategy=numpyro.infer.util.init_to_uniform(radius=0.1))
    mcmc = MCMC(kernel, num_warmup=args.n_warmup, num_samples=args.n_samples, num_chains=args.n_chains, jit_model_args=True)
    if post_warmup_state is not None:
        mcmc.post_warmup_state = post_warmup_state
    rng_key = jax.random.PRNGKey(args.seed)
    X_train = dataset.X[split["tr"]]
    y_train = dataset.y[split["tr"]]
    time_before = time()
    mcmc.run(
        rng_key, 
        X=X_train, 
        y=y_train, 
        D_Y=y_train.shape[-1],
        sigma=dataset.noise_level if hasattr(dataset, "noise_level") else None,
        extra_fields=("adapt_state.step_size", "diverging", "i", "num_steps", "accept_prob", "mean_accept_prob")
    )
    time_spanned = time() - time_before
    return mcmc, time_spanned

def save_mcmc(mcmc, model_name, dataset_name, split_ind: int, args):
    inference_data = az.from_numpyro(
        mcmc
    )
    if not os.path.exists(f"{args.write_dir}/HMC"):
        os.mkdir(f"{args.write_dir}/HMC")
    if not os.path.exists(f"{args.write_dir}/HMC/{dataset_name}"):
        os.mkdir(f"{args.write_dir}/HMC/{dataset_name}")
    inference_data.to_netcdf(f"{args.write_dir}/HMC/{dataset_name}/{model_name}_{split_ind}.nc")

def save_metadata(model_name: str, dataset_name: str, metadata, split_ind: int, args):
    if not os.path.exists(f"{args.write_dir}/HMC"):
        os.mkdir(f"{args.write_dir}/HMC")
    if not os.path.exists(f"{args.write_dir}/{dataset_name}"):
        os.mkdir(f"{args.write_dir}/HMC/{dataset_name}")
    with open(f"{args.write_dir}/HMC/{dataset_name}/{model_name}_{split_ind}_metadata.txt", "w") as f:
        f.write(str(metadata))
    
