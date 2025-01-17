import os
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
from numpyro.infer import init_to_mean, init_to_sample, init_to_uniform

def save_txt(save_dir, folder_name, file_name, content: str):
    path = save_dir
    if os.path.exists(path) and os.path.isdir(path):
        path = os.path.join(path, folder_name)
        if not os.path.exists(path):
            os.mkdir(path)
        path = os.path.join(path, file_name)
        with open(path, "w") as f:
            f.write(content)
    else:
        raise Exception("Save directory does not exist")


def save_csv(save_dir, folder_name, file_name, content: pd.DataFrame):
    path = save_dir
    if os.path.exists(path) and os.path.isdir(path):
        path = os.path.join(path, folder_name)
        if not os.path.exists(path):
            os.mkdir(path)
        path = os.path.join(path, file_name)
        content.to_csv(path)
    else:
        raise Exception("Save directory does not exist")


def save_arviz_posterior(save_dir, folder_name, file_name, mcmc):
    inference_data = az.from_numpyro(mcmc)
    az.plot_posterior(inference_data)
    fig = plt.gcf()
    path = save_dir
    if os.path.exists(path) and os.path.isdir(path):
        path = os.path.join(path, folder_name)
        if not os.path.exists(path):
            os.mkdir(path)
        path = os.path.join(path, file_name)
        fig.savefig(path)
    else:
        raise Exception("Save directory does not exist")

def save_arviz_p_value(save_dir, folder_name, file_name, mcmc, posterior_predictive):
    inference_data = az.from_numpyro(mcmc, posterior_predictive=posterior_predictive)
    az.plot_bpv(inference_data, kind="p_value")
    fig = plt.gcf()
    path = save_dir
    if os.path.exists(path) and os.path.isdir(path):
        path = os.path.join(path, folder_name)
        if not os.path.exists(path):
            os.mkdir(path)
        path = os.path.join(path, file_name)
        fig.savefig(path)
    else:
        raise Exception("Save directory does not exist")


def save_fig(save_dir, folder_name, file_name, fig):
    path = save_dir
    if os.path.exists(path) and os.path.isdir(path):
        path = os.path.join(path, folder_name)
        if not os.path.exists(path):
            os.mkdir(path)
        path = os.path.join(path, file_name)
        fig.savefig(path)
    else:
        raise Exception("Save directory does not exist")

def colour_map(name):
    if "WBNN" in name:
        return "orange"
    else:
        return "blue"

def get_model_depth(model_name):
    split = model_name.split("_")
    deep_index = split.index("deep")
    depth = int(split[deep_index - 1])
    return depth

def get_init_strategy(args):
    if args.init_strategy == "init_to_mean":
        return init_to_mean
    elif args.init_strategy == "init_to_sample":
        return init_to_sample
    else:
        return init_to_uniform

def get_num_params(mcmc):
    return sum([v.shape[-1] for v in mcmc.get_samples().values()])