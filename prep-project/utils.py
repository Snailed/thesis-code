import os
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt

def save_txt(save_dir, folder_name, file_name, content: str):
    path = save_dir
    if os.path.exists(path) and os.path.isdir(path):
        path = os.path.join(path, folder_name)
        if os.path.exists(path):
            path = os.path.join(path, file_name)
            with open(path, "w") as f:
                f.write(content)
        else:
            os.mkdir(path)
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
    if "Circulant" in name:
        return "orange"
    else:
        return "blue"

def get_model_depth(model_name):
    split = model_name.split("_")
    deep_index = split.index("deep")
    depth = int(split[deep_index - 1])
    return depth