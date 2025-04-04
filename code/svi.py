import os
import jax
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.svi import SVIRunResult
from numpyro.infer.autoguide import AutoDelta, AutoGuide, init_to_uniform
from numpyro.optim import Adam
from functools import partial
import pickle

def run_svi(model, dataset, split, args) -> SVIRunResult:
    optimizer = Adam(args.learning_rate)
    guide = AutoDelta(model, init_loc_fn=partial(init_to_uniform, radius=0.1))
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    key = jax.random.PRNGKey(args.seed)
    #X = dataset.data[split["tr"]][:,:-1]
    X = dataset.normalize_X(dataset.X, split)[split["tr"]]
    #y = dataset.data[split["tr"]][:,-1]
    y = dataset.y[split["tr"]]
    #assert args.subsample_size is not None
    svi_result = svi.run(key, args.n_steps, X=X, y=y, subsample=args.subsample_size, progress_bar=args.progress_bar)
    return svi_result, guide
    

def save_svi(model_name: str, dataset_name: str, metadata, split_ind: int, args):
    if not os.path.exists(f"{args.write_dir}/SVI"):
        os.mkdir(f"{args.write_dir}/SVI")
    if not os.path.exists(f"{args.write_dir}/SVI/{dataset_name}"):
        os.mkdir(f"{args.write_dir}/SVI/{dataset_name}")
    with open(f"{args.write_dir}/SVI/{dataset_name}/{model_name}_{split_ind}.pickle", "wb") as f:
        pickle.dump(metadata, f)