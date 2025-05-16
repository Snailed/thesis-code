
import jax
import jax.numpy as jnp
import jax.random as random
import numpyro
from numpyro.infer import Predictive
import pandas as pd
import dill
import matplotlib.pyplot as plt
import os
import sys
import arviz as az
from numpyro import set_platform
from tqdm.notebook import tqdm
# Add ../code to PYTHON_PATH
sys.path.insert(0, "../code")
sys.path.insert(0, "..")
import bnns.model_configs
from datasets.ecg import ECGDataset
jax.config.update("jax_enable_x64", True)
from numpyro.infer import log_likelihood
from sklearn.utils import resample
import argparse

global SAMPLE_PATH
global SEED
global MAX_DATA

def load_samples():
    methods = os.listdir(SAMPLE_PATH)
    samples = []
    for method in methods:
        path = os.path.join(SAMPLE_PATH, method, "ecg")
        if method == "SVI":
            splits = os.listdir(path)
            for split in splits:
                with open(os.path.join(path, split), "rb") as f:
                    svi_result = dill.load(f)
                model_name = "_".join(split.split(".")[0].split("_")[:-1])
                split_ind = split.split(".")[0].split("_")[-1]
                model = getattr(bnns.model_configs, model_name)
                params = svi_result["params"]
                guide = svi_result["guide"]
                predictive = Predictive(guide, params=params, num_samples=1)
                post_draws = predictive(jax.random.PRNGKey(SEED), None)
                samples.append({
                    "model": model,
                    "method": method,
                    "model": model_name,
                    "split": split_ind,
                    "post_draws": post_draws
                })
        if method == "HMC":
            splits = os.listdir(path)
            for split in splits:
                if ".nc" not in split:
                    continue
                inference_data = az.from_netcdf(os.path.join(path, split))
                model_name = "_".join(split.split(".")[0].split("_")[:-1])
                split_ind = split.split(".")[0].split("_")[-1]
                model = getattr(bnns.model_configs, model_name)

                post_draws = inference_data.to_dict()["posterior"]
                post_draws.pop("y_probs", None)
                samples.append({
                    "model": model,
                    "method": method,
                    "model": model_name,
                    "split": split_ind,
                    "post_draws": post_draws
                })
    return samples

def evaluate(model, post_draws, X, y, batch_ndims, adjust=True):
    predictive = Predictive(model, post_draws, batch_ndims=batch_ndims)
    if adjust:
        prior_probs = dataset.train_label_distribution()
    else:
        prior_probs = None
    predictions = predictive(jax.random.PRNGKey(SEED), X, prior_probs=prior_probs)["y_probs"]

    y_pred = jnp.argmax(predictions, -1)[0]
    assert y.shape[-1] == y_pred.shape[-1], f"y_test shape: {y.shape}, y_pred shape: {y_pred.shape}"
    return y_pred, predictions

def batch_nll(model, post_draws, X, y, batch_ndims, adjust=True):
    batch_size = min(5120, X.shape[0])
    n_batches = X.shape[0] // batch_size + 1 if X.shape[0] > 5120 else 1
    nlls = []
    if adjust:
        prior_probs = dataset.train_label_distribution()
    else:
        prior_probs = None

    for i in range(n_batches):
        start = i * batch_size
        end = max((i + 1) * batch_size, X.shape[0] - 1)
        nll = log_likelihood(model, post_draws, X[start:end], y[start:end], prior_probs=prior_probs, batch_ndims=batch_ndims)["y"]
        #nll = jax.scipy.special.logsumexp(nll, axis=-2) - jnp.log(nll.shape[-2])
        if batch_ndims == 2:
            nll = jax.scipy.special.logsumexp(nll, axis=-2).astype(jnp.float64) - jnp.log(nll.shape[-2])
        nlls.append(nll)
    nlls = jnp.concat(nlls, axis=-1)
    nlls = jax.scipy.special.logsumexp(nlls, axis=-2)
    return -jnp.mean(nlls.astype(jnp.float64)).astype(jnp.float64)

def evaluate(samples):
    results = []
    preds = []
    for sample_dict in tqdm(samples):
        model = getattr(bnns.model_configs, sample_dict["model"])
        post_draws = sample_dict["post_draws"]
        batch_ndims = 1 if sample_dict["method"] == "SVI" else 2

        split = dataset.splits[int(sample_dict["split"])]
        X = dataset.normalize_X(dataset.X, split)

        X_train = X[split["tr"]]
        y_train = dataset.y[split["tr"]]

        X_test = X[split["te"]]
        y_test = dataset.y[split["te"]]

        # Get subset for testing
        if MAX_DATA is not None:
            X_train, y_train = resample(X_train, y_train, n_samples=MAX_DATA, random_state=SEED)
            X_test, y_test = resample(X_test, y_test, n_samples=MAX_DATA, random_state=SEED)

        train_nll = batch_nll(model, post_draws, X_train, y_train, batch_ndims, adjust=False)
        test_nll = batch_nll(model, post_draws, X_test, y_test, batch_ndims, adjust=True)

        y_pred_train, y_pred_train_probs = evaluate(model, post_draws, X_train, y_train, batch_ndims, adjust=False)
        train_accuracy = (y_train == y_pred_train).mean()

        y_pred_test, y_pred_test_probs = evaluate(model, post_draws, X_test, y_test, batch_ndims, adjust=True)
        y_pred_test_unadjusted, y_pred_test_probs_unadjusted = evaluate(model, post_draws, X_test, y_test, batch_ndims, adjust=False)
        test_accuracy = (y_test == y_pred_test).mean()
        test_accuracy_unadjusted = (y_test == y_pred_test_unadjusted).mean()

        results.append({
            "method": sample_dict["method"],
            "model": sample_dict["model"],
            "split": sample_dict["split"],
            "train_nll": train_nll.mean(),
            "test_nll": test_nll.mean(),
            "train accuracy": f"{train_accuracy * 100 : .3f}%",
            "test accuracy": f"{test_accuracy * 100 : .3f}%",
            "test accuracy, udadjusted": f"{test_accuracy_unadjusted * 100 : .3f}%"
        })
        preds.append({
            "y_train": y_train,
            "y_pred_train": y_pred_train,
            "y_pred_train_probs": y_pred_train_probs,
            "y_test": y_test,
            "y_pred_test": y_pred_test,
            "y_pred_test_unadjusted": y_pred_test_unadjusted,
            "y_pred_test_probs": y_pred_test_probs,
            "y_pred_test_probs_unadjusted": y_pred_test_probs_unadjusted
        })
    results = pd.DataFrame(results)
    return results, preds

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--max_data", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    args = parser.parse_args()
    set_platform(args.device)

    SAMPLE_PATH = "../samples/" + args.sample_path
    SEED = args.seed
    MAX_DATA = args.max_data

    methods = [d for d in os.listdir(SAMPLE_PATH) if os.path.isdir(os.path.join(SAMPLE_PATH, d))]
    dataset = ECGDataset(resample_train=False)

    samples = load_samples()
    results, preds = evaluate(samples)
    results.to_csv("ecg_results.csv", index=False)
    print(results)
    # Save the predictions
    with open("ecg_predictions.pkl", "wb") as f:
        dill.dump(preds, f)