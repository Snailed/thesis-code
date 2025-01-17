from jax import random
import jax
from numpyro.infer import NUTS, MCMC, Predictive, log_likelihood, init_to_value, SVI, autoguide, Trace_ELBO
from abc import ABC
import matplotlib.pyplot as plt
from numpyro import handlers
import numpyro
import jax.numpy as jnp
import numpy as np
import os
from utils import save_txt, save_csv, save_arviz_posterior, save_arviz_p_value, colour_map, get_model_depth, save_fig, get_init_strategy, get_num_params
import pandas as pd
import arviz as az
import time
import json
from plotting import plot_comparative_boxplots, plot_comparative_bars, plot_comparative_violinplots, plot_seperate_violinplots, plot_seperate_bars, plot_line
import re
import seaborn as sns

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
                kernel = NUTS(model, init_strategy=get_init_strategy(args))
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
                        Predictive(model, posterior_samples=mcmc.get_samples())(key_, X=dataset.X, Y=dataset.Y, sigma=dataset.sigma)
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
        rng_key, ll_key, loo_key, waic_key, prior_predictive_key, posterior_predictive_key = random.split(rng_key, 6)
        self.plot_log_likelihood_per_param(ll_key, models, mcmcs, depths, args)
        self.plot_loo(loo_key, models, mcmcs, depths, args)
        self.plot_waic(waic_key, models, mcmcs, depths, args)
        self.plot_prior_predictive(prior_predictive_key, models, mcmcs, args)
        self.plot_posterior_predictive(posterior_predictive_key, models, mcmcs, args)
        self.plot_duration(models, depths, durations, args)
        self.plot_summary(models, mcmcs, depths, args)
        if args.show:
            plt.show()

    def get_map_estimate(self, rng_key, model, dataset, args):
        print("Computing MAP estimate...")
        guide = autoguide.AutoDelta(model)
        optimizer = numpyro.optim.Adam(0.001)
        svi = SVI(model, guide, optimizer, Trace_ELBO())
        svi_results = svi.run(rng_key, 20_000, X=dataset.X, Y=dataset.Y, sigma=dataset.sigma)
        params = svi_results.params

        return params, guide

    def mcmc(self, rng_key, model, dataset, args):
        init_strategy = get_init_strategy(args)
        if args.init_strategy == "init_to_map":
            rng_key, key_ = random.split(rng_key)
            map_params, map_guide = self.get_map_estimate(key_, model, dataset, args)
            def init_to_map(site=None):
                return init_to_value(site, map_params)
            init_strategy = init_to_map
        kernel = NUTS(model, init_strategy=init_strategy)
        mcmc = MCMC(kernel, num_chains=args.num_chains, num_samples=args.num_samples, num_warmup=args.num_warmup, progress_bar=False)
        time_before = time.time()
        mcmc.run(rng_key, dataset.X, dataset.Y, sigma=dataset.sigma, extra_fields=("i", "accept_prob", "mean_accept_prob", "adapt_state.step_size"))
        if args.save:
            save_txt(args.save_dir, "mcmc_stats", model.__name__, str(mcmc.get_extra_fields(group_by_chain=True)))
        time_after = time.time()
        delta = time_after - time_before
        return mcmc, delta

    def plot_loo(self, rng_key, models, mcmcs, depths, args):
        dicts = []
        for i, mcmc in enumerate(mcmcs):
            inference_data = az.from_numpyro(mcmc, num_chains=args.num_chains)
            for chain in range(args.num_chains):
                loo = az.loo(inference_data.sel(chains=[chain]), var_name="Y")
                dicts += [{
                            "depth": depths[i], 
                            "loo": loo["elpd_loo"], 
                            "width": re.search(r"(\d+)\_wide", models[i].__name__).group(1),
                            "weight tied": "WBNN" in models[i].__name__
                        }]
        fig, ax = plt.subplots()
        df = pd.DataFrame(dicts)
        plot_line(ax, df, "depth", "loo")

        if args.save:
            fig.tight_layout()
            save_fig(args.save_dir, "", "loo.png", fig)

    def plot_waic(self, rng_key, models, mcmcs, depths, args):
        dicts = []
        for i, mcmc in enumerate(mcmcs):
            inference_data = az.from_numpyro(mcmc, num_chains=args.num_chains)
            for chain in range(args.num_chains):
                waic = az.waic(inference_data.sel(chains=[chain]), var_name="Y")
                dicts += [{
                            "depth": depths[i], 
                            "waic": waic["elpd_waic"], 
                            "width": re.search(r"(\d+)\_wide", models[i].__name__).group(1),
                            "weight tied": "WBNN" in models[i].__name__
                        }]
        fig, ax = plt.subplots()
        df = pd.DataFrame(dicts)
        plot_line(ax, df, "depth", "waic")

        if args.save:
            fig.tight_layout()
            save_fig(args.save_dir, "", "waic.png", fig)
    
    def plot_log_likelihood_per_param(self, rng_key, models, mcmcs, depths, args):
        dicts = []
        for i, mcmc in enumerate(mcmcs):
            inference_data = az.from_numpyro(mcmc, num_chains=args.num_chains)
            ll = log_likelihood(handlers.seed(models[i], rng_key), mcmc.get_samples(group_by_chain=True), self.dataset.X, self.dataset.Y, sigma=self.dataset.sigma,batch_ndims=2)
            for chain in range(args.num_chains):
                dicts += [{
                            "depth": depths[i],
                            "# RVs": get_num_params(mcmc), 
                            "width": re.search(r"(\d+)\_wide", models[i].__name__).group(1),
                            "log likelihood": float(np.array(ll["Y"][chain].mean())), 
                            "weight tied": "WBNN" in models[i].__name__
                        }]
        df = pd.DataFrame(dicts)
        fig = plt.figure(figsize=(8,8))
        ax = fig.gca()
        plot_line(ax, df.copy(), "# RVs", "log likelihood")

        if args.save:
            fig.tight_layout()
            save_fig(args.save_dir, "", "log_likelihood_per_param.png", fig)

        fig, ax = plt.subplots()
        plot_line(ax, df, "depth", "log likelihood")

        if args.save:
            fig.tight_layout()
            save_fig(args.save_dir, "", "log_likelihood.png", fig)

    # def plot_log_likelihood(self, rng_key, models, mcmcs, depths, args):
    #     dicts = []
    #     for i, mcmc in enumerate(mcmcs):
    #         inference_data = az.from_numpyro(mcmc, num_chains=args.num_chains)
    #         ll = log_likelihood(handlers.seed(models[i], rng_key), mcmc.get_samples(group_by_chain=True), self.dataset.X, self.dataset.Y, self.dataset.sigma, batch_ndims=2)
    #         for chain in range(args.num_chains):
    #             dicts += [{
    #                         "depth": depths[i], 
    #                         "log likelihood": float(np.array(ll["Y"][chain].mean())), 
    #                         "weight tied": "WBNN" in models[i].__name__
    #                     }]
    #     df = pd.DataFrame(dicts)
    #     fig, axs = plot_seperate_bars(df, "depth", "log likelihood")

    #     if args.save:
    #         fig.tight_layout()
    #         save_fig(args.save_dir, "", "log_likelihood.png", fig)
    def _plot_predictive(self, ax, predictions):
        ax.plot(self.dataset.X, self.dataset.Y, "ko", markersize=3)
        ax.plot(self.dataset.x_true, self.dataset.y_true)


        #predictions = predictions[..., 0, :]
        ax.plot(self.dataset.x_true, jnp.mean(predictions, axis=0))
        percentiles = jnp.percentile(predictions, jnp.array([5.0, 95.0]), axis=0)
        ax.fill_between(
            self.dataset.x_true, percentiles[0, :], percentiles[1, :], color="lightblue"
        )
        ax.set_xlabel("x")
        ax.set_xlabel("y")

    def plot_posterior_predictive(self, rng_key, models, mcmcs, args):
        for model, mcmc in zip(models, mcmcs):
            rng_key, key = random.split(rng_key)
            posterior_samples = mcmc.get_samples(group_by_chain=True)
            # Predict from model
            model = handlers.substitute(handlers.seed(model, key), posterior_samples)
            model_trace = handlers.trace(model).get_trace(X=jnp.array([self.dataset.x_true]).T, Y=None)
            predictions = model_trace["Y"]["value"]
            predictions = predictions[..., 0]
            for chain in range(args.num_chains):
                predictions_chain = predictions[chain]
                fig = plt.figure()
                ax = fig.gca()
                self._plot_predictive(ax, predictions_chain)
                if args.save:
                    save_fig(args.save_dir, "posterior_predictive", f"{model.__name__}_chain_{chain}.png", fig)
            
            # Plot combined
            fig = plt.figure()
            ax = fig.gca()
            model = handlers.substitute(handlers.seed(model, key), mcmc.get_samples())
            model_trace = handlers.trace(model).get_trace(X=jnp.array([self.dataset.x_true]).T, Y=None)
            predictions = model_trace["Y"]["value"]
            predictions = predictions[..., 0]
            self._plot_predictive(ax, predictions)
            if args.save:
                save_fig(args.save_dir, "posterior_predictive", f"{model.__name__}_combined.png", fig)

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
        dicts = []
        for i, model in enumerate(models):
            for chain in range(args.num_chains):
                total_samples = args.num_samples + args.num_warmup
                dicts += [{
                            "depth": depths[i], 
                            "width": re.search(r"(\d+)\_wide", models[i].__name__).group(1),
                            "duration": durations[i], 
                            "throughput": total_samples/durations[i],
                            "weight tied": "WBNN" in models[i].__name__
                        }]
        df = pd.DataFrame(dicts)
        fig = plt.figure()
        ax = fig.gca()
        plot_comparative_bars(ax, df.copy(), "depth", "duration")

        if args.save:
            save_fig(args.save_dir, "", "duration.png", fig)

        fig = plt.figure()
        ax = fig.gca()

        plot_comparative_bars(ax, df, "depth", "throughput")
        if args.save:
            save_fig(args.save_dir, "", "throughput.png", fig)

    def plot_summary(self, models, mcmcs, depths, args):
        dicts = []
        for i, mcmc in enumerate(mcmcs):
            inference_data = az.from_numpyro(mcmc, num_chains=args.num_chains)
            rhat = np.array(az.rhat(inference_data).to_dataarray())
            for chain in range(args.num_chains):
                dicts += [{
                            "depth": depths[i], 
                            "width": re.search(r"(\d+)\_wide", models[i].__name__).group(1),
                            "Mean r-hat": rhat.mean(),  # R-hat needs multiple chains to make sense
                            "Mean ESS": np.array(az.ess(inference_data.sel(chains=[chain])).to_dataarray()).mean(),
                            "weight tied": "WBNN" in models[i].__name__
                        }]
        df = pd.DataFrame(dicts)
        fig = plt.figure()
        ax = fig.gca()
        plot_comparative_bars(ax, df.copy(), "depth", "Mean r-hat")

        if args.save:
            save_fig(args.save_dir, "", "rhat.png", fig)

        fig = plt.figure()
        ax = fig.gca()

        plot_comparative_bars(ax, df, "depth", "Mean ESS")
        if args.save:
            save_fig(args.save_dir, "", "ess.png", fig)

  #  def plot_summary(self, models, mcmcs, depths, args):
        
        # r_hats = []
        # n_effs = []
        # for mcmc in mcmcs:
        #     summary = numpyro.diagnostics.summary(mcmc.get_samples(), group_by_chain=False)
        #     df = pd.DataFrame(summary)
        #     r_hat = df.loc["r_hat"]
        #     n_eff = df.loc["n_eff"]
        #     r_hats += [r_hat.mean().mean()]
        #     n_effs += [n_eff.mean().mean()]

        # weight tied_indices = [i for i, model in enumerate(models) if "WBNN" in model.__name__]
        # non_weight tied_indices = [i for i, model in enumerate(models) if "WBNN" not in model.__name__]

        # fig = plt.figure()
        # ax = fig.gca()
        # ax.plot([depths[i] for i in weight tied_indices], [r_hats[i] for i in weight tied_indices], "^-", label="weight tied")
        # ax.plot([depths[i] for i in non_weight tied_indices], [r_hats[i] for i in non_weight tied_indices], "^-", label="Regular")
        # ax.plot(jnp.arange(0, max(depths)), jnp.ones(max(depths)), "--", color="gray", alpha=0.5)
        # ax.set_xlabel("Depth")
        # ax.set_ylabel("$\hat{r}$")
        # ax.legend()
        
        # if args.save:
        #     save_fig(args.save_dir, "", "r_hat.png", fig)

        # fig = plt.figure()
        # ax = fig.gca()
        # ax.plot([depths[i] for i in weight tied_indices], [n_effs[i] for i in weight tied_indices], "^-", label="weight tied")
        # ax.plot([depths[i] for i in non_weight tied_indices], [n_effs[i] for i in non_weight tied_indices], "^-", label="Regular")
        # ax.set_xlabel("Depth")
        # ax.set_ylabel("ESS")
        # ax.legend()
        # r_hats = []
        # n_effs = []
        # for mcmc in mcmcs:
        #     summary = numpyro.diagnostics.summary(mcmc.get_samples(), group_by_chain=False)
        #     df = pd.DataFrame(summary)
        #     r_hat = df.loc["r_hat"]
        #     n_eff = df.loc["n_eff"]
        #     r_hats += [r_hat.mean().mean()]
        #     n_effs += [n_eff.mean().mean()]

        # weight tied_indices = [i for i, model in enumerate(models) if "WBNN" in model.__name__]
        # non_weight tied_indices = [i for i, model in enumerate(models) if "WBNN" not in model.__name__]

        # fig = plt.figure()
        # ax = fig.gca()
        # ax.plot([depths[i] for i in weight tied_indices], [r_hats[i] for i in weight tied_indices], "^-", label="weight tied")
        # ax.plot([depths[i] for i in non_weight tied_indices], [r_hats[i] for i in non_weight tied_indices], "^-", label="Regular")
        # ax.plot(jnp.arange(0, max(depths)), jnp.ones(max(depths)), "--", color="gray", alpha=0.5)
        # ax.set_xlabel("Depth")
        # ax.set_ylabel("$\hat{r}$")
        # ax.legend()
        
        # if args.save:
        #     save_fig(args.save_dir, "", "r_hat.png", fig)

        # fig = plt.figure()
        # ax = fig.gca()
        # ax.plot([depths[i] for i in weight tied_indices], [n_effs[i] for i in weight tied_indices], "^-", label="weight tied")
        # ax.plot([depths[i] for i in non_weight tied_indices], [n_effs[i] for i in non_weight tied_indices], "^-", label="Regular")
        # ax.set_xlabel("Depth")
        # ax.set_ylabel("ESS")
        # ax.legend()
        
      #  if args.save:
      #      save_fig(args.save_dir, "", "n_eff.png", fig)

class SVI_Exp(Experiment):
    def run(self, rng_key, models, datasets, args):
        assert args.dataset == ["SineRegression"]
        self.dataset = datasets[0]
        svis = []
        guides = []
        depths = []
        durations = []
        for model in models:
            depths += [get_model_depth(model.__name__)]

        rng_key, *keys = random.split(rng_key, len(models) + 1)
        for model, key in zip(models, keys):
            svi, guide, duration = self.svi(key, model, self.dataset, args)
            svis += [svi]
            guides += [guide]
            durations += [duration]
        rng_key, ll_key, loo_key, waic_key, prior_predictive_key, posterior_predictive_key = random.split(rng_key, 6)
        self.plot_log_likelihood(ll_key, models, svis, guides, depths, args)
        self.plot_loss(ll_key, models, svis, guides, depths, args)
        self.plot_duration(models, depths, durations, args)
        if args.show:
            plt.show()
    
    def svi(self, rng_key, model, dataset, args):
        guide = autoguide.AutoDelta(model)
        optimizer = numpyro.optim.Adam(0.001)
        svi = SVI(model, guide, optimizer, Trace_ELBO())
        time_before = time.time()
        svi_results = svi.run(rng_key, 20_000, X=dataset.X, Y=dataset.Y, sigma=dataset.sigma)
        time_after = time.time()
        delta = time_after - time_before
        return svi_results, guide, delta
    
    def plot_log_likelihood(self, rng_key, models, svis, guides, depths, args):
        dicts = []
        keys = random.split(rng_key, len(models))
        for i, (svi, guide) in enumerate(zip(svis, guides)):
            posterior_samples = Predictive(guide, params=svi.params, num_samples=1000)(keys[i])
            ll = log_likelihood(handlers.seed(models[i], rng_key), posterior_samples, self.dataset.X, self.dataset.Y, sigma=self.dataset.sigma,batch_ndims=1)
            dicts += [{
                        "depth": depths[i],
                        "width": re.search(r"(\d+)\_wide", models[i].__name__).group(1),
                        "log likelihood": float(np.array(ll["Y"].mean())), 
                        "weight tied": "WBNN" in models[i].__name__
                    }]
        df = pd.DataFrame(dicts)
        fig, ax = plt.subplots()
        plot_line(ax, df, "depth", "log likelihood")

        if args.save:
            fig.tight_layout()
            save_fig(args.save_dir, "", "log_likelihood.png", fig)

    def plot_loss(self, rng_key, models, svis, guides, depths, args):
        dicts = []
        keys = random.split(rng_key, len(models))
        for i, (svi, guide) in enumerate(zip(svis, guides)):
            for step in range(0,len(svi.losses),10):
                dicts += [{
                            "depth": depths[i],
                            "width": re.search(r"(\d+)\_wide", models[i].__name__).group(1),
                            "step": step,
                            "loss": float(svi.losses[step]), 
                            "weight tied": "WBNN" in models[i].__name__
                        }]
        df = pd.DataFrame(dicts)
        fig, ax = plt.subplots()
        ax.set_yscale("log")
        df["weight tied"] = df["weight tied"].apply(lambda x: "WBNN" if x else "BNN")
        df["width and depth"] = "W=" + df["width"].astype(str) + ", D=" + df["depth"].astype(str)
        sns.lineplot(ax=ax, data=df, x="step", y="loss", hue="width and depth", style="weight tied", markers=False, estimator=None)

        if args.save:
            fig.tight_layout()
            save_fig(args.save_dir, "", "loss.png", fig)
    
    def plot_duration(self, models, depths, durations, args):
        dicts = []
        for i, model in enumerate(models):
            dicts += [{
                        "depth": depths[i], 
                        "width": re.search(r"(\d+)\_wide", models[i].__name__).group(1),
                        "duration": durations[i], 
                        "weight tied": "WBNN" in models[i].__name__
                    }]
        df = pd.DataFrame(dicts)
        fig = plt.figure()
        ax = fig.gca()
        plot_comparative_bars(ax, df.copy(), "depth", "duration")

        if args.save:
            save_fig(args.save_dir, "", "duration.png", fig)