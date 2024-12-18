from jax import random
import jax
from numpyro.infer import NUTS, MCMC, Predictive, log_likelihood
from abc import ABC
import matplotlib.pyplot as plt
from numpyro import handlers
import numpyro
import jax.numpy as jnp
import numpy as np
import os
from utils import save_txt, save_csv, save_arviz_posterior, save_arviz_p_value, colour_map, get_model_depth, save_fig
import pandas as pd
import arviz as az
import time

class Experiment(ABC):
    def run(self, rng_key, model, dataset, args):
        pass

"""
    Preliminary results
"""
class HMCInfer(Experiment):
    def run(self, key, models, datasets, args):
        for dataset in datasets:
            mcmcs = []
            for model in models:
                kernel = NUTS(model)
                mcmc = MCMC(kernel, num_chains=1, num_samples=args.num_samples, num_warmup=args.num_warmup)
                key, key_ = random.split(key)
                mcmc.run(key_, dataset.X, dataset.Y)
                print(f"{model.__name__}")
                if args.save:
                    save_csv(
                        args.save_dir, 
                        "mcmc_summaries", 
                        f"{model.__name__}.csv", 
                        pd.DataFrame.from_dict(numpyro.diagnostics.summary(mcmc.get_samples(group_by_chain=True)))
                        )
                    save_arviz_posterior(
                        args.save_dir,
                        "arviz_posterior",
                        f"{model.__name__}.png",
                        mcmc
                    )
                    key, key_ = random.split(key)
                    save_arviz_p_value(
                        args.save_dir,
                        "arviz_p_value",
                        f"{model.__name__}.png",
                        mcmc,
                        Predictive(model, posterior_samples=mcmc.get_samples())(key_, X=dataset.X, Y=dataset.Y)
                    )
                mcmcs += [mcmc]
            
            if dataset.__class__.__name__ == "SineRegression":
                key, key_ = random.split(key)
                HMCInfer.plot_sine(key_, models, mcmcs, dataset, args)
        HMCInfer.plot_model_scatter(models, mcmcs, datasets, args)
            
    @staticmethod 
    def predict(model, rng_key, samples, X):
        model = handlers.substitute(handlers.seed(model, rng_key), samples)
        model_trace = handlers.trace(model).get_trace(X=X, Y=None)
        return model_trace["Y"]["value"]

    @staticmethod
    def plot_sine(rng_key, models, mcmcs, dataset, args):
        if len(mcmcs) > 20:
            print("WARNING: Too many samples to plot for SineRegression, skipping...")
            return
        fig, axs = plt.subplots(1, len(mcmcs), figsize=(len(mcmcs)*10, 10))
        axs = np.array([axs]).ravel()
        
        keys = random.split(rng_key, len(mcmcs))
        for i in range(len(mcmcs)):
            ax = axs[i]
            ax.plot(dataset.x_true, dataset.y_true, "b-", linewidth=2)
            ax.plot(dataset.X, dataset.Y, "ko", markersize=4)
            predictions = HMCInfer.predict(models[i], keys[i], mcmcs[i].get_samples(), jnp.array([dataset.x_true]).T)
            predictions = predictions[..., 0]
            ax.plot(dataset.x_true, jnp.mean(predictions, axis=0))
            percentiles = jnp.percentile(predictions, jnp.array([5.0, 95.0]), axis=0)
            ax.fill_between(
                dataset.x_true, percentiles[0, :], percentiles[1, :], color="lightblue"
            )
            ax.set_title(models[i].__name__)
        fig.tight_layout()
        if args.save:
            fig.savefig(os.path.join(args.save_dir, "sine-regression.png"))
        if args.show:
            plt.show()
    
    @staticmethod
    def log_likelihood(model, mcmc, dataset):
        return log_likelihood(model, mcmc.get_samples(), dataset.X, dataset.Y)
    
    @staticmethod
    def plot_model_scatter(models, mcmcs, datasets, args):
        fig, axs = plt.subplots(1, len(datasets), figsize=(len(datasets)*10, 10))
        axs = np.array([axs]).ravel()
        for i, dataset in enumerate(datasets):
            ax = axs[i]
            params = [sum([z.size for z in mcmc.last_state.z.values()]) for mcmc in mcmcs]
            print("Params/Names", list(zip([model.__name__ for model in models], params)))
            mean_likelihoods = [jnp.mean(HMCInfer.log_likelihood(models[j], mcmcs[j], dataset)["Y"]) for j in range(len(models))]
            colours = [colour_map(model.__name__) for model in models]
            if "log_likelihood":
                ax.scatter(params, mean_likelihoods, c=colours)
                ax.set_xlabel("Parameters")
                ax.set_ylabel("Log Likelihood")
        if args.save:
            fig.savefig(os.path.join(args.save_dir, "log_likelihood_scatter.png"))
        if args.show:
            plt.show()
            

class HMC(Experiment):
    def run(self, rng_key, models, datasets, args):
        assert args.dataset == ["SineRegression"]
        self.dataset = datasets[0]
        mcmcs = []
        depths = []
        durations = []
        for model in models:
            depths += [get_model_depth(model.__name__)]

        rng_key, *keys = random.split(rng_key, len(models) + 1)
        for model, key in zip(models, keys):
            mcmc, duration = self.mcmc(key, model, self.dataset, args)
            mcmcs += [mcmc]
            durations += [duration]
        rng_key, loo_key, waic_key, prior_predictive_key, posterior_predictive_key = random.split(rng_key, 5)
        self.plot_loo(loo_key, models, mcmcs, depths, args)
        self.plot_waic(waic_key, models, mcmcs, depths, args)
        self.plot_prior_predictive(prior_predictive_key, models, mcmcs, args)
        self.plot_posterior_predictive(posterior_predictive_key, models, mcmcs, args)
        self.plot_duration(models, depths, durations, args)
        self.plot_summary(models, mcmcs, depths, args)
        if args.show:
            plt.show()

    def mcmc(self, rng_key, model, dataset, args):
        kernel = NUTS(model)
        mcmc = MCMC(kernel, num_chains=args.num_chains, num_samples=args.num_samples, num_warmup=args.num_warmup)
        time_before = time.time()
        mcmc.run(rng_key, dataset.X, dataset.Y, sigma=dataset.sigma)
        time_after = time.time()
        delta = time_after - time_before
        return mcmc, delta

    def plot_loo(self, rng_key, models, mcmcs, depths, args):
        loos = []
        for mcmc in mcmcs:
            inference_data = az.from_numpyro(mcmc)
            loo = az.loo(inference_data, var_name="Y")
            loos += [loo]
        fig = plt.figure()
        ax = fig.gca()
        circulant_indices = [i for i, model in enumerate(models) if "Circ" in model.__name__]
        non_circulant_indices = [i for i, model in enumerate(models) if "Circ" not in model.__name__]
        ax.set_xlabel("Depth")
        ax.set_ylabel("LOO")
        ax.plot([depths[i] for i in circulant_indices], [loos[i]["elpd_loo"] for i in circulant_indices], "^-", label="Circulant")
        ax.plot([depths[i] for i in non_circulant_indices], [loos[i]["elpd_loo"] for i in non_circulant_indices], "^-", label="Regular")
        ax.legend()
        ax.set_xticks(ax.get_xticks().astype(int))
        if args.save:
            save_fig(args.save_dir, "", "loo.png", fig)

    def plot_waic(self, rng_key, models, mcmcs, depths, args):
        waics = []
        for mcmc in mcmcs:
            inference_data = az.from_numpyro(mcmc)
            waic = az.waic(inference_data, var_name="Y")
            waics += [waic]
        fig = plt.figure()
        ax = fig.gca()
        circulant_indices = [i for i, model in enumerate(models) if "Circ" in model.__name__]
        non_circulant_indices = [i for i, model in enumerate(models) if "Circ" not in model.__name__]
        ax.set_xlabel("Depth")
        ax.set_ylabel("WAIC")
        ax.plot([depths[i] for i in circulant_indices], [waics[i]["elpd_waic"] for i in circulant_indices], "^-", label="Circulant")
        ax.plot([depths[i] for i in non_circulant_indices], [waics[i]["elpd_waic"] for i in non_circulant_indices], "^-", label="Regular")
        ax.legend()
        ax.set_xticks(ax.get_xticks().astype(int))
        if args.save:
            save_fig(args.save_dir, "", "waic.png", fig)

    def plot_posterior_predictive(self, rng_key, models, mcmcs, args):
        for model, mcmc in zip(models, mcmcs):
            rng_key, key = random.split(rng_key)
            fig = plt.figure()
            ax = fig.gca()
            ax.plot(self.dataset.X, self.dataset.Y, "ko", markersize=3)
            ax.plot(self.dataset.x_true, self.dataset.y_true)

            # Predict from model
            model = handlers.substitute(handlers.seed(model, key), mcmc.get_samples())
            model_trace = handlers.trace(model).get_trace(X=jnp.array([self.dataset.x_true]).T, Y=None)
            predictions = model_trace["Y"]["value"]

            predictions = predictions[..., 0]
            ax.plot(self.dataset.x_true, jnp.mean(predictions, axis=0))
            percentiles = jnp.percentile(predictions, jnp.array([5.0, 95.0]), axis=0)
            ax.fill_between(
                self.dataset.x_true, percentiles[0, :], percentiles[1, :], color="lightblue"
            )
            ax.set_xlabel("x")
            ax.set_xlabel("y")
            if args.save:
                save_fig(args.save_dir, "posterior_predictive", f"{model.__name__}.png", fig)

    def plot_prior_predictive(self, rng_key, models, mcmcs, args):
        for model, mcmc in zip(models, mcmcs):
            rng_key, key = random.split(rng_key)
            fig = plt.figure()
            ax = fig.gca()
            ax.plot(self.dataset.X, self.dataset.Y, "ko", markersize=3)
            ax.plot(self.dataset.x_true, self.dataset.y_true)

            # Predict from model
            prior_predictive = Predictive(model, num_samples=100)
            predictions = prior_predictive(key, X=jnp.array([self.dataset.x_true]).T)["Y"]

            if predictions.ndim == 4:
                predictions = predictions[...,0,:,0]
            else:
                predictions = predictions[..., 0]
            ax.plot(self.dataset.x_true, jnp.mean(predictions, axis=0))
            percentiles = jnp.percentile(predictions, jnp.array([5.0, 95.0]), axis=0)
            ax.fill_between(
                self.dataset.x_true, percentiles[0, :], percentiles[1, :], color="lightblue"
            )
            ax.set_xlabel("x")
            ax.set_xlabel("y")
            if args.save:
                save_fig(args.save_dir, "prior_predictive", f"{model.__name__}.png", fig)

    def plot_duration(self, models, depths, durations, args):
        circulant_indices = [i for i, model in enumerate(models) if "Circ" in model.__name__]
        non_circulant_indices = [i for i, model in enumerate(models) if "Circ" not in model.__name__]

        fig = plt.figure()
        ax = fig.gca()
        ax.plot([depths[i] for i in circulant_indices], [durations[i] for i in circulant_indices], "^-", label="Circulant")
        ax.plot([depths[i] for i in non_circulant_indices], [durations[i] for i in non_circulant_indices], "^-", label="Regular")
        ax.set_xlabel("Depth")
        ax.set_ylabel("Duration (s)")
        ax.legend()

        if args.save:
            save_fig(args.save_dir, "", "duration.png", fig)

        fig = plt.figure()
        ax = fig.gca()
        total_samples = args.num_samples + args.num_warmup
        ax.plot([depths[i] for i in circulant_indices], [total_samples / durations[i] for i in circulant_indices], "^-", label="Circulant")
        ax.plot([depths[i] for i in non_circulant_indices], [total_samples / durations[i] for i in non_circulant_indices], "^-", label="Regular")
        ax.set_xlabel("Depth")
        ax.set_ylabel("Throughput (samples/s)")
        ax.legend()

        if args.save:
            save_fig(args.save_dir, "", "throughput.png", fig)

    def plot_summary(self, models, mcmcs, depths, args):
        r_hats = []
        n_effs = []
        for mcmc in mcmcs:
            summary = numpyro.diagnostics.summary(mcmc.get_samples(), group_by_chain=False)
            df = pd.DataFrame(summary)
            r_hat = df.loc["r_hat"]
            n_eff = df.loc["n_eff"]
            r_hats += [r_hat.mean().mean()]
            n_effs += [n_eff.mean().mean()]

        circulant_indices = [i for i, model in enumerate(models) if "Circ" in model.__name__]
        non_circulant_indices = [i for i, model in enumerate(models) if "Circ" not in model.__name__]

        fig = plt.figure()
        ax = fig.gca()
        ax.plot([depths[i] for i in circulant_indices], [r_hats[i] for i in circulant_indices], "^-", label="Circulant")
        ax.plot([depths[i] for i in non_circulant_indices], [r_hats[i] for i in non_circulant_indices], "^-", label="Regular")
        ax.plot(jnp.arange(0, max(depths)), jnp.ones(max(depths)), "--", color="gray", alpha=0.5)
        ax.set_xlabel("Depth")
        ax.set_ylabel("$\hat{r}$")
        ax.legend()
        
        if args.save:
            save_fig(args.save_dir, "", "r_hat.png", fig)

        fig = plt.figure()
        ax = fig.gca()
        ax.plot([depths[i] for i in circulant_indices], [n_effs[i] for i in circulant_indices], "^-", label="Circulant")
        ax.plot([depths[i] for i in non_circulant_indices], [n_effs[i] for i in non_circulant_indices], "^-", label="Regular")
        ax.set_xlabel("Depth")
        ax.set_ylabel("ESS")
        ax.legend()
        
        if args.save:
            save_fig(args.save_dir, "", "n_eff.png", fig)
