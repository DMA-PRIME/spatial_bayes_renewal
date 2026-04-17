import numpy as np
import matplotlib.pyplot as plt
import numpyro.distributions as dist
from jax import random
import seaborn as sns
import jax.numpy as jnp

class Distributions:
    def __init__(self, sample_size, seed,visualize,fig_path):
        self.sample_size = sample_size
        self.key = random.PRNGKey(seed)
        self.fig_path = fig_path
        self.visualize=visualize


    def uniform(self, low=0.0, high=10.0):
        d = dist.Uniform(low, high)
        samples = d.sample(self.key, sample_shape=(self.sample_size,))
        if self.visualize==True:
            self.plot_dist(samples, f"Uniform Distribution ({low}, {high})")
        return samples

    def normal(self, loc=0.0, scale=1.0):
        d = dist.Normal(loc, scale)
        samples = d.sample(self.key, sample_shape=(self.sample_size,))
        if self.visualize == True:
            self.plot_dist(samples, f"Normal Distribution ({loc}, {scale})")
        return samples

    def lognormal(self, loc=0.0, scale=0.5,T=10):
        d = dist.LogNormal(loc, scale)
        samples = d.sample(self.key, sample_shape=(self.sample_size,))
        samples = jnp.floor(samples).astype(int)
        samples=[i for i in samples if i<=T]
        if self.visualize == True:
            self.plot_dist(samples, f"LogNormal Distribution ({loc}, {scale})")
        return samples

    def gamma(self, concentration=2.0, rate=1.0):
        d = dist.Gamma(concentration, rate)
        samples = d.sample(self.key, sample_shape=(self.sample_size,))
        if self.visualize == True:
            self.plot_dist(samples, f"Gamma Distribution ({concentration}, {rate})")
        return samples

    def truncated_normal(self, loc=0.0, scale=1.0, low=-1.0, high=1.0):
        d = dist.TruncatedNormal(loc, scale, low=low, high=high)
        samples = d.sample(self.key, sample_shape=(self.sample_size,))
        if self.visualize == True:
            self.plot_dist(samples, f"TruncatedNormal Distribution ({loc}, {scale}, {low}, {high})")
        return samples

    def negative_binomial(self, mu=2, alpha=2,T=10):
        d = dist.NegativeBinomial2(mu,alpha)
        samples = d.sample(self.key, sample_shape=(self.sample_size,))
        samples = jnp.floor(samples).astype(int)
        samples = [i for i in samples if i <= T]
        if self.visualize == True:
            self.plot_dist(samples, f"NegativeBinomial Distribution (mu={mu}, p={alpha})")
        return samples

    def compute_portion(self,samples):
        values, counts = np.unique(samples, return_counts=True)

        # Normalize to get probabilities
        probs = counts / counts.sum()
        return probs

    def viral_load_triangle(self,tau, V_peak, tau_peak, tau_shed):
        """
        Piecewise viral load shedding curve (log10[s_cont(tau)])

        Parameters
        ----------
        tau : array-like
            Time (days).
        V_peak : float
            Peak viral load (log10 scale).
        tau_peak : float
            Time of peak viral load.
        tau_shed : float
            Time when shedding ends.
        """
        tau = jnp.asarray(tau)

        # Case 1: ramp up
        case1 = (tau <= tau_peak) * (V_peak * tau / tau_peak)

        # Case 2: decline after peak
        case2 = ((tau > tau_peak) & (tau <= tau_shed)) * (
                V_peak * (1 - (tau - tau_peak) / (tau_shed - tau_peak))
        )

        # Case 3: after shedding
        case3 = (tau > tau_shed) * 0.0
        y_vals=case1 + case2 + case3
        if self.visualize == True:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(tau, y_vals, label="log10[s_cont(tau)]")
            ax.axvline(tau_peak, color="red", linestyle="--", label="tau_peak")
            ax.axvline(tau_shed, color="gray", linestyle="--", label="tau_shed")
            ax.set_xlabel("Days since infection (τ)")
            ax.set_ylabel("log10 viral load")
            plt.legend()
            plt.savefig(self.fig_path+f'viral_load_triangle.png')
        return y_vals


    def plot_dist(self, samples, title):
        fig, ax = plt.subplots(figsize=(4, 2.5))
        samples_np = np.array(samples)
        sns.histplot(
            samples_np,
            bins=10,
            stat='density',  # density=True equivalent for seaborn
            edgecolor='black',
            alpha=0.7,
            ax=ax
        )

        # KDE on the same axes
        sns.kdeplot(
            samples_np,
            ax=ax,
            linewidth=2
        )
        ax.set_xlim(left=0)
        ax.set_title(title)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        plt.tight_layout()
        plt.savefig(self.fig_path+f'{title}.png')


