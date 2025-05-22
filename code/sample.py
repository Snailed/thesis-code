import numpyro
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist
import jax.numpy as jnp
import jax
import arviz as az
import os
from time import time
import dill
from utils import normalize

def run_hmc(model, dataset, split, args, initial_point=None):
    if initial_point is None:
        init_strategy = numpyro.infer.util.init_to_uniform(radius=0.1)
    else:
        print("Initial point keys...", initial_point.keys())
        init_strategy = numpyro.infer.util.init_to_value(values=initial_point)

    kernel = NUTS(
            model, 
            init_strategy=init_strategy,
            max_tree_depth=args.tree_depth,
        )
    mcmc = MCMC(
        kernel, 
        num_warmup=args.n_warmup, 
        num_samples=args.n_samples, 
        num_chains=args.n_chains, 
        jit_model_args=True, 
        progress_bar=args.progress_bar,
        chain_method=args.chain_method
    )

    rng_key = jax.random.PRNGKey(args.seed)
    X_train = dataset.X[split["tr"]]
    X_train, X_mean, X_std = normalize(X_train, None, None)
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
    print(f"Finished sampling in {time_spanned:.2f} seconds, with throughput {args.n_samples * args.n_chains / time_spanned:.2f} samples per second")
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
    if not os.path.exists(f"{args.write_dir}/HMC/{dataset_name}"):
        os.mkdir(f"{args.write_dir}/HMC/{dataset_name}")
    with open(f"{args.write_dir}/HMC/{dataset_name}/{model_name}_{split_ind}_metadata.dill", "wb") as f:
        f.write(dill.dumps(metadata))
    
