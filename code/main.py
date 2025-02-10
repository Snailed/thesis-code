import argparse
import os
from datetime import datetime
from datasets.synthetic import SyntheticDataset
from datasets.uci import UCIDataset, dataset_names
import matplotlib.pyplot as plt
import bnns.model_configs
from sample import save_mcmc, run_hmc, save_metadata

def sample(args):
    # Make write directory
    args.write_dir = os.path.join("samples", datetime.now().isoformat())
    os.mkdir(args.write_dir) 
    with open(os.path.join(args.write_dir, "args.txt"), "w") as f:
        f.write(str(args.__dict__))
    
    # Load datasets
    datasets = []
    if len(datasets) == 1 and datasets[0] == "all":
        datasets = [SyntheticDataset()] + [UCIDataset(dataset) for dataset in dataset_names]
    for dataset in args.dataset:
        if dataset in dataset_names:
            datasets.append(UCIDataset(dataset))
        elif dataset == "synthetic":
            datasets.append(SyntheticDataset())
    
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


    # Sample
    for dataset in datasets:
        for model in models:
            for split_ind, split in enumerate(dataset.splits):
                print(f"Sampling {model.__name__} on {dataset.dataset_name} split {split_ind}")
                mcmc, time_spanned = run_hmc(model, dataset, split, args)
                save_mcmc(mcmc, model.__name__, dataset.dataset_name, split_ind, args)
                extra_fields = mcmc.get_extra_fields()
                extra_fields["time_spanned"] = time_spanned
                save_metadata(model.__name__, dataset.dataset_name, extra_fields, split_ind, args)


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
    parser_sample.set_defaults(func=sample)

    parser_plot = subparsers.add_parser('plot', help='Plot samples')
    parser_plot.set_defaults(func=plot)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()