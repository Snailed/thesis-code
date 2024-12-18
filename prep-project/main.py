import argparse
import jax.random as random
from dataset import *
from model import *
from experiment import *
from datetime import datetime
import os

from typing import List

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", type=bool, default=False)
    parser.add_argument("--dataset", nargs="+", type=str, default=["SineRegression"])
    parser.add_argument("--model", nargs="+", type=str, default=["SimpleBNN"])
    parser.add_argument("--experiment", nargs="+", type=str, default=["HMCInfer"])
    parser.add_argument("--num_samples", "-n", type=int, default=500)
    parser.add_argument("--num_warmup", type=int, default=2000)
    parser.add_argument("--num_chains", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=bool, default=False) # Defaults to True when dry_run == False
    parser.add_argument("--save_dir", type=str) # Where to save is --save is enabled
    parser.add_argument("--show", type=bool, default=True)
    parser.add_argument("--metrics", type=str, nargs="+", default=["MSE"])
    args = parser.parse_args()

    # Set dry run args (overriding)
    if args.dry_run:
        args.num_samples = 20 
        args.num_warmup = 10
        args.num_chains = 1
    
    if args.num_chains > 1:
        numpyro.set_host_device_count(args.num_chains)
    if not args.dry_run:
        args.save = True
    if args.save and args.save_dir == None:
        args.save_dir = os.path.join("results", str(datetime.now()))

    if args.save_dir is not None:
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        else:
            raise Exception("Save directory already exists, exiting...")

    key = random.PRNGKey(args.seed)
    key, *key_list = random.split(key, len(args.dataset) + 1)

    # Instantiate datasets
    datasets = [eval(ds)(key_list[i]) for i,ds in enumerate(args.dataset)]

    # Instantiate models
    models = [eval(m) for m in args.model]

    # Instantiate experimental setups
    experiments: List[Experiment] = [eval(m)() for m in args.experiment]

    for experiment in experiments:
        key, key_ = random.split(key)
        experiment.run(key_, models, datasets, args)
    
