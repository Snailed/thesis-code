import os
import jax
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.svi import SVIRunResult
from numpyro.infer.autoguide import AutoDelta, AutoGuide, init_to_uniform
from numpyro.optim import Adam
from functools import partial
import pickle

LEARNING_RATES = {
    "boston-housing": 5e-5,
    "concrete": 5e-4,
    "energy": 5e-4,
    "kin8nm": 5e-5,
    "naval-propulsion-plant": 5e-4,
    "power-plant": 5e-4,
    "protein-tertiary-structure": 5e-3,
    "wine-quality-red": 5e-5,
    "yacht": 5e-5,
    "ecg": 5e-3,
}

def run_svi(model, dataset, split, steps, args) -> SVIRunResult:
    if dataset.dataset_name in LEARNING_RATES:
        lr = LEARNING_RATES[dataset.dataset_name]
    else:
        lr = args.learning_rate
    optimizer = Adam(lr)
    guide = AutoDelta(model, init_loc_fn=partial(init_to_uniform, radius=0.1))
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    key = jax.random.PRNGKey(args.seed)
    #X = dataset.data[split["tr"]][:,:-1]
    X = dataset.normalize_X(dataset.X, split)[split["tr"]]
    #y = dataset.data[split["tr"]][:,-1]
    y = dataset.y[split["tr"]]
    #assert args.subsample_size is not None
    sigma = dataset.noise_level if "noise_level" in dataset.__dict__ else None
    svi_result = svi.run(key, steps, X=X, y=y, sigma=sigma, subsample=args.subsample_size, progress_bar=args.progress_bar)
    return svi_result, guide
    

def save_svi(model_name: str, dataset_name: str, metadata, split_ind: int, args):
    if not os.path.exists(f"{args.write_dir}/SVI"):
        os.mkdir(f"{args.write_dir}/SVI")
    if not os.path.exists(f"{args.write_dir}/SVI/{dataset_name}"):
        os.mkdir(f"{args.write_dir}/SVI/{dataset_name}")
    with open(f"{args.write_dir}/SVI/{dataset_name}/{model_name}_{split_ind}.pickle", "wb") as f:
        pickle.dump(metadata, f)