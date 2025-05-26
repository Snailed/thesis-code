import argparse
import os
from datetime import datetime
from datasets.synthetic import SyntheticDataset
from datasets.uci import UCIDataset, dataset_names
from datasets.ecg import ECGDataset
import matplotlib.pyplot as plt
import bnns.model_configs
from sample import save_mcmc, run_hmc, save_metadata
from svi import run_svi, save_svi
from numpyro import set_platform

def _mkdir_sample_dir(args):
    # Make write directory
    args.write_dir = os.path.join("samples", datetime.now().isoformat())
    os.mkdir(args.write_dir) 
    with open(os.path.join(args.write_dir, "args.txt"), "w") as f:
        f.write(str(args.__dict__))

def _load_datasets(args):
    # Load datasets
    datasets = []
    if len(datasets) == 1 and datasets[0] == "all":
        datasets = [SyntheticDataset()] + [UCIDataset(dataset) for dataset in dataset_names]
    for dataset in args.dataset:
        if dataset in dataset_names:
            datasets.append(UCIDataset(dataset))
        elif dataset == "synthetic":
            datasets.append(SyntheticDataset())
        elif dataset == "ecg":
            print("Loading ECG dataset, resampling:", args.resample_ecg)
            datasets.append(ECGDataset(resample_train=args.resample_ecg))
    return datasets

def _load_models(args):
    # Load models
    models = []
    model_names = [name for name in dir(bnns.model_configs)]
    model_list = [getattr(bnns.model_configs, name) for name in model_names]
    for model_name in args.models:
        try:
            index = model_names.index(model_name)
            models.append(model_list[index])
        except ValueError:
            raise ValueError(f"Model {model_name} not found")
    return models, model_names

def sample(args):
    set_platform(args.device)
    _mkdir_sample_dir(args)
    datasets = _load_datasets(args)
    models, model_names = _load_models(args)

    # Sample
    for dataset in datasets:
        for model in models:
            for split_ind, split in enumerate(dataset.splits):
                if split_ind < args.start_split:
                    print("Skipping split", split_ind)
                    continue
                if split_ind >= args.max_splits:
                    print("Reached max splits, stopping")
                    break
                if args.init_map_iters is not None:
                    print(f"Computing MAP estimate for initial point for {args.init_map_iters} iterations")
                    #args.subsample_size = 100 
                    args.subsample_size = None
                    svi_result, guide = run_svi(model, dataset, split, args.init_map_iters, args)
                    initial_point = svi_result.params
                    initial_point = {k.removesuffix('_auto_loc'): v for k, v in initial_point.items()}
                else:
                    initial_point = None

                print(f"Sampling {model.__name__} on {dataset.dataset_name} split {split_ind}")
                mcmc, time_spanned = run_hmc(model, dataset, split, args, initial_point=initial_point)
                save_mcmc(mcmc, model.__name__, dataset.dataset_name, split_ind, args)
                extra_fields = mcmc.get_extra_fields()
                extra_fields["time_spanned"] = time_spanned
                save_metadata(model.__name__, dataset.dataset_name, extra_fields, split_ind, args)

def map(args):
    set_platform(args.device)
    _mkdir_sample_dir(args)
    datasets = _load_datasets(args)
    models, model_names = _load_models(args)

    # MAP-estimate
    for dataset in datasets:
        for model in models:
            for split_ind, split in enumerate(dataset.splits):
                if split_ind >= args.max_splits:
                    print("Reached max splits, stopping")
                    break
                print("MAP-estimating", model.__name__, "on", dataset.dataset_name, "split", split_ind)
                svi_result, guide = run_svi(model, dataset, split, args.n_steps, args)
                svi_result = {
                    "params": svi_result.params,
                    "losses": svi_result.losses,
                    "state": svi_result.state,
                    "guide": guide
                }
                save_svi(model.__name__, dataset.dataset_name, svi_result, split_ind, args)
                

def plot(args):
    print("Plot", args)

def main():
    parser = argparse.ArgumentParser(description='Thesis script.')

    subparsers = parser.add_subparsers(help='Action')

    parser_sample = subparsers.add_parser('sample', help='Collect samples from posterior distribution using HMC')
    parser_sample.add_argument('--n_samples', type=int, default=1000, help='Number of samples to collect')
    parser_sample.add_argument('--n_chains', type=int, default=1, help='Number of chains to run')
    parser_sample.add_argument('--n_warmup', type=int, default=1000, help='Number of warmup iterations')
    parser_sample.add_argument('--models', nargs='+', default=[], help='Models to sample from')
    parser_sample.add_argument('--dataset', nargs='+', default=["synthetic"], help='Datasets to sample from')
    parser_sample.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser_sample.add_argument("--progress-bar", action="store_true", help="Show progress bar")
    parser_sample.add_argument("--chain_method", default="vectorized", help="MCMC chain method (parallel, sequential, vectorized)")
    parser_sample.add_argument("--max_splits", default=20, type=int, help="Maximum number of dataset splits to consider")
    parser_sample.add_argument("--start_split", default=0, type=int, help="Start from split")
    parser_sample.add_argument("--device", default="cpu", choices=["cpu", "gpu"], help="Device to use. Either cpu or gpu.")
    parser_sample.add_argument("--init_map_iters", default=None, type=int, help="If provided, use MAP estimate as initial point")
    parser_sample.add_argument("--resample-ecg", action="store_true", help="Should ECG data be resampled?")
    parser_sample.add_argument("--tree-depth", default=10, type=int, help="Max tree depth of doubling scheme for NUTS")
    parser_sample.add_argument("--step-size", default=1, type=float, help="Step size. If None then use adaptive step sizes from NUTS")
    parser_sample.add_argument("--dense-mass-matrix", default=False, help="Should use dense mass matrix. Default: Diagonal.")

    parser_sample.set_defaults(func=sample)

    parser_plot = subparsers.add_parser('plot', help='Plot samples')
    parser_plot.set_defaults(func=plot)

    parser_map = subparsers.add_parser('map', help='MAP-estimate')
    parser_map.add_argument('--n_steps', type=int, default=1000, help='Number of optimization steps')
    parser_map.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate')
    parser_map.add_argument('--models', nargs='+', default=[], help='Models to sample from')
    parser_map.add_argument('--dataset', nargs='+', default=["synthetic"], help='Datasets to sample from')
    parser_map.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser_map.add_argument("--progress-bar", action="store_true", help="Show progress bar")
    parser_map.add_argument("--subsample-size", type=int, help="Subsample size if supported", default=None)
    parser_map.add_argument("--max_splits", default=20, type=int, help="Maximum number of dataset splits to consider")
    parser_map.add_argument("--device", default="cpu", choices=["cpu", "gpu"], help="Device to use. Either cpu or gpu.")
    parser_map.add_argument("--resample-ecg", action="store_true", help="Should ECG data be resampled?")
    parser_map.set_defaults(func=map)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
