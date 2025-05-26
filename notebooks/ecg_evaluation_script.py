
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
from tqdm import tqdm
# Add ../code to PYTHON_PATH
sys.path.insert(0, "../code")
sys.path.insert(0, "..")
import bnns.model_configs
from datasets.ecg import ECGDataset
jax.config.update("jax_enable_x64", True)
from numpyro.infer import log_likelihood
from sklearn.utils import resample
import argparse
from scipy.stats import mannwhitneyu
from numpyro.diagnostics import summary

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
    batch_size = min(1028, X.shape[0])
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
    return -jnp.mean(nlls.astype(jnp.float64)).astype(jnp.float64), nlls.std().astype(jnp.float64)

def compute_nlls(model, post_draws, X, y, batch_ndims):
    nll = log_likelihood(model, post_draws, X, y, prior_probs=None, batch_ndims=batch_ndims)["y"]
    return -nll.mean(axis=-1) # (1, 3000)

def compute_accuracy(model, post_draws, X, y, batch_ndims):
    predictive = Predictive(model, post_draws, batch_ndims=batch_ndims)
    predictions = predictive(jax.random.PRNGKey(SEED), X, prior_probs=None)["y_probs"]
    y_pred = jnp.argmax(predictions, -1)[0]
    return (y == y_pred).mean(axis=-1).flatten()

def bold_if_significant(value, better, significant):
    if better and significant:
        return f"$\\mathbf{{\\underlined{{{value}}}}}"
    elif better:
        return f"$\\mathbf{{{value}}}"
    elif significant:
        return f"$\\underlined{{{value}}}"
    else:
        return str(value)
def produce_table(samples):
    results_table = []
    split = dataset.splits[0]
    X = dataset.normalize_X(dataset.X, split)
    X_train = X[split["tr"]]
    y_train = dataset.y[split["tr"]]
    batch_ndims = 2

    X_test = X[split["te"]]
    y_test = dataset.y[split["te"]]

    # Get subset for testing
    if MAX_DATA is not None:
        X_train, y_train = resample(X_train, y_train, n_samples=MAX_DATA, random_state=SEED)
        X_test, y_test = resample(X_test, y_test, n_samples=MAX_DATA, random_state=SEED)
    
    bnn_index = 0
    assert samples[bnn_index]["model"] == "ECG_BNN_128"

    sample_dict = samples[bnn_index]
    model = getattr(bnns.model_configs, sample_dict["model"])
    post_draws = sample_dict["post_draws"]
    #baseline_nll_train = compute_nlls(model, post_draws, X_train, y_train, batch_ndims)
    nll_baseline = compute_nlls(model, post_draws, X_test, y_test, batch_ndims)
    accuracy_baseline = compute_accuracy(model, post_draws, X_test, y_test, batch_ndims)
    assert accuracy_baseline.shape == (3000,), f"Accuracy has shape {accuracy_baseline.shape}"
    
    summary_baseline = summary(post_draws)
    ess_baseline = jnp.array([jnp.nanmean(value["n_eff"]) for _, value in summary.items()])
    rhat_baseline = jnp.array([jnp.nanmean(value["r_hat"]) for _, value in summary.items()])
    time_baseline = sample_dict.get("time_spanned", None)
    ess_per_s_baseline = ess_baseline / time_baseline

    # Compute NLLs
    # Baseline
    for sample_dict in samples:
        model = getattr(bnns.model_configs, sample_dict["model"])
        #comparison_nll_train = compute_nlls(model, post_draws, X_train, y_train, batch_ndims)
        if sample_dict["model"] == "ECG_BNN_128":
            nll_comparison = nll_baseline
            accuracy_comparison = accuracy_baseline
            time_comparison = time_baseline

            ess_comparison = ess_baseline
            rhat_comparison = rhat_baseline
            ess_per_s_comparison = ess_per_s_baseline
        else:
            post_draws = sample_dict["post_draws"]
            nll_comparison = compute_nlls(model, post_draws, X_test, y_test, batch_ndims)
            accuracy_comparison = compute_accuracy(model, post_draws, X_test, y_test, batch_ndims)
            time_comparison = sample_dict.get("time_spanned", None)

            summary = summary(post_draws)
            ess_comparison = jnp.array([jnp.nanmean(value["n_eff"]) for _, value in summary.items()])
            rhat_comparison = jnp.array([jnp.nanmean(value["r_hat"]) for _, value in summary.items()])
            ess_per_s_comparison = ess_comparison / time_comparison
        accept_prob = sample_dict.get("accept_prob", None)

        #nll_stat, nll_p = mannwhitneyu(nll_baseline.flatten(), nll_comparison.flatten(), alternative="greater")
        #accuracy_stat, accuracy_p = mannwhitneyu(accuracy_baseline, accuracy_comparison, alternative="greater")
        #ess_stat, ess_p = mannwhitneyu(ess_baseline, ess_comparison, alternative="less")
        #rhat_stat, rhat_p = mannwhitneyu(rhat_baseline, rhat_comparison, alternative="greater")
        #ess_s_stat, ess_s_p = mannwhitneyu(ess_s_baseline, ess_s_comparison, alternative="less")

        # Significance
        #nll_significant = nll_p < 0.05
        nll_better = nll_comparison.mean() < nll_baseline.mean()
        accuracy_better = accuracy_comparison.mean() > accuracy_baseline.mean()
        time_better = time_comparison < time_baseline
        ess_better = ess_comparison.mean() > ess_baseline.mean()
        rhat_better = rhat_comparison.mean() < rhat_baseline.mean()
        ess_per_s_better = ess_per_s_comparison.mean() > ess_per_s_baseline.mean()
        #rmse_significant = rmse_p < 0.05
        #ess_significant = ess_p < 0.05
        #rhat_significant = rhat_p < 0.05
        #ess_s_significant = ess_s_p < 0.05

        results_table.append({
            "model": sample_dict["model"],
            "dataset": "ECG",
            "NLL": bold_if_significant(f"{nll_comparison.mean() : .3f}\\pm{nll_comparison.std() : .3f}", nll_better, False),
            "Accuracy": bold_if_significant(f"{accuracy_comparison.mean() : .3f}\\% \\pm{accuracy_comparison.std() : .3f} \\%", accuracy_better, False),
            "Time": bold_if_significant(f"{time_comparison.mean() : .3f}\\% \\pm{time_comparison.std() : .3f} \\%", time_better, False),
            "Speedup": f"{time_comparison / time_baseline : .3f}",
            "ESS": bold_if_significant(f"{ess_comparison.mean() : .3f}\\pm{ess_comparison.std() : .3f}", ess_better, False),
            "R-hat": bold_if_significant(f"{rhat_comparison.mean() : .3f}\\pm{rhat_comparison.std() : .3f}", rhat_better, False),
            "ESS/s": bold_if_significant(f"{ess_per_s_comparison.mean() : .3f}\\pm{ess_per_s_comparison.std() : .3f}", ess_per_s_better, False)
            #"RMSE": bold_if_significant(f"{rmse_comparison.mean() : .3f}\\pm{rmse_comparison.std() : .3f}", rmse_significant),
            #"Time": f"{comparison['time_spanned'].mean() : .3f}",
            #"Speedup": f"{baseline['time_spanned'].mean() / comparison['time_spanned'].mean() : .3f}",
            #"Acceptance Prob": f"{comparison['accept_prob'].mean() : .3f}",
            #"ESS": bold_if_significant(f"{comparison['ess'].mean() : .3f}\\pm{comparison['ess'].std() : .3f}", ess_significant),
            #"R-hat": bold_if_significant(f"{comparison['rhat'].mean() : .3f}\\pm{comparison['rhat'].std() : .3f}", rhat_significant),
            #"ESS/s": bold_if_significant(f"{comparison['ess_per_s'].mean() : .3f}\\pm{comparison['ess_per_s'].std() : .3f}", ess_s_significant),
        })
    results_df = pd.DataFrame(results_table)
    results_df.to_csv("ECG_pretty_table.csv")
    print(results_df)

    


def evaluate_all(samples):
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
        print("Computing negative log-likelihood")
        train_nll, train_nll_std = batch_nll(model, post_draws, X_train, y_train, batch_ndims, adjust=False)
        test_nll, test_nll_std = batch_nll(model, post_draws, X_test, y_test, batch_ndims, adjust=True)

        print("Computing accuracy")
        y_pred_train, y_pred_train_probs = evaluate(model, post_draws, X_train, y_train, batch_ndims, adjust=False)
        train_accuracy = (y_train == y_pred_train).mean()
        train_accuracy_std = (y_train == y_pred_train).std()

        y_pred_test, y_pred_test_probs = evaluate(model, post_draws, X_test, y_test, batch_ndims, adjust=True)
        y_pred_test_unadjusted, y_pred_test_probs_unadjusted = evaluate(model, post_draws, X_test, y_test, batch_ndims, adjust=False)
        test_accuracy = (y_test == y_pred_test).mean()
        test_accuracy_unadjusted = (y_test == y_pred_test_unadjusted).mean()
        test_accuracy_unadjusted_std = (y_test == y_pred_test_unadjusted).std()

        results.append({
            "method": sample_dict["method"],
            "model": sample_dict["model"],
            "split": sample_dict["split"],
            "train_nll": train_nll.mean(),
            "train_nll_std": train_nll_std.mean(),
            "test_nll": test_nll.mean(),
            "test_nll_std": test_nll_std.mean(),
            "train accuracy": f"{train_accuracy * 100 : .3f}%",
            "test accuracy": f"{test_accuracy * 100 : .3f}%",
            "test accuracy, udadjusted": f"{test_accuracy_unadjusted * 100 : .3f}%",
            "test accuracy, udadjusted std": f"{test_accuracy_unadjusted_std * 100 : .3f}%"
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

    produce_table(samples)
    #results, preds = evaluate_all(samples)
    #results.to_csv("ecg_results.csv", index=False)
    #print(results)
    ## Save the predictions
    #with open("ecg_predictions.pkl", "wb") as f:
    #    dill.dump(preds, f)
