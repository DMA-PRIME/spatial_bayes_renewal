# Vectorized Spatial Bayesian Renewal Forecaster
# NOTE: You must fill in gen_int_array, inf_hosp_array, and sheddin before calling run_mcmc

import numpy as np
import polars as pl
from pathlib import Path
import jax.numpy as jnp
import jax.random as jr
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from jax import vmap

import matplotlib.pyplot as plt

class SpatialForecaster:
    """
    Spatial Bayesian Renewal Model for forecasting infectious disease dynamics
    across multiple connected spatial regions.
    
    Models disease transmission with spatial spillover effects between regions,
    incorporates wastewater surveillance data, and handles hierarchical parameters.
    """
    
    def __init__(
        self,
        df_data,
        spatial_net,
        region_list,
        region_obs_list,
        cols_concern,
        n_forecast_points,
        data_path,
        Renewal_infection_case='Basic',
        num_samples=200,
        num_warmup=100,
        num_chains=1,
        progress_bar=True,
        print_summary=True,
    ):
        self.cols_concern = cols_concern
        self.spatial_net = spatial_net
        self.region_list = region_list
        self.region_obs_list = region_obs_list
        self.region_ww_obs_list=region_obs_list
        self.region_pop=None

        self.n_forecast_points = int(n_forecast_points)
        self.data_path = data_path

        self.hosp_obs = None
        self.ww_obs = None

        self.gen_int_array = None
        self.inf_hosp_array = None
        self.sheddin = None
        self.n_previous_points = 10

        for col in cols_concern:
            if col == 'hosp_obs':
                blk = []
                for region in self.region_obs_list:
                    df = df_data.filter(pl.col('county') == region)
                    blk.append(df[col].to_numpy().astype(np.float32).ravel())
                self.hosp_obs = np.vstack(blk)
                self.N_obs_regions, self.T = self.hosp_obs.shape

            if col == 'ww_obs':
                blk = []
                for region in self.region_ww_obs_list:
                    df = df_data.filter(pl.col('county') == region)
                    blk.append(df[col].to_numpy().astype(np.float32).ravel())
                self.ww_obs = np.vstack(blk)
                self.N_obs_regions, self.T = self.ww_obs.shape

        self.N_regions = len(region_list)
        self.num_samples = num_samples
        self.num_warmup = num_warmup
        self.num_chains = num_chains
        self.progress_bar = progress_bar
        self.print_summary = print_summary

        self.R_t = None
        self.I_t = None

        self.set_I0 = 1000.0
        self.set_R0 = 1.2
        self.set_P_hosp = 0.1
        self.set_G = 1.0
        self.delay_ww=1

        # Build adjacency matrix for spatial vectorization
        self.adj_matrix = self._build_adjacency_matrix(spatial_net, region_list)

    def update_training_data(self, df_data_train):
        """Update internal arrays when new training data arrive."""
        self.df_data_train = df_data_train

        for col in self.cols_concern:
            if col == 'hosp_obs':
                blk = []
                for region in self.region_obs_list:
                    df = df_data_train[df_data_train['county'] == region]
                    blk.append(df[col].to_numpy().astype(np.float32).ravel())
                self.hosp_obs = np.vstack(blk)
                self.N_obs_regions, self.T = self.hosp_obs.shape

            if col == 'ww_obs':
                blk = []
                for region in self.region_ww_obs_list:
                    df = df_data_train[df_data_train['county'] == region]
                    blk.append(df[col].to_numpy().astype(np.float32).ravel())
                self.ww_obs = np.vstack(blk)
                self.N_obs_regions, self.T = self.ww_obs.shape


    def _build_adjacency_matrix(self, spatial_net, region_list):
        N = len(region_list)
        M = np.zeros((N, N), dtype=np.float32)
        idx = {r: i for i, r in enumerate(region_list)}
        for i, ri in enumerate(region_list):
            for rj in spatial_net.predecessors(ri):
                j = idx[rj]
                # edge rj -> ri, so weight from j -> i
                M[i, j] = spatial_net[rj][ri]["weight"]
        return jnp.array(M)

    def R_t_func_old(self, T):
        R0 = numpyro.sample("R0", dist.TruncatedNormal(loc=self.set_R0, scale=0.1, low=0.1))

        eps = numpyro.sample("rw_steps", dist.Normal(loc=jnp.zeros(T), scale=0.005))
        logR0 = jnp.log(R0)
        logR_rest = logR0 + jnp.cumsum(eps[:-1])
        logR = jnp.concatenate([jnp.array([logR0]), logR_rest])
        R_t = jnp.clip(jnp.exp(logR), 0.01, 2.0)
        numpyro.deterministic("R_t", R_t)
        return R_t

    def R_t_func(self, T):
        """
        R_t per time and region: shape (T, N_regions).

        - Baseline: log R_t is a random walk shared across regions.
        - For regions with wastewater (region_obs_list), log R_t is adjusted
          by that region's wastewater covariate.
        - For regions without wastewater, R_t stays as the baseline RW.
        """

        N = self.N_regions

        # ----- Baseline random walk on log R_t -----
        R0 = numpyro.sample(
            "R0",
            dist.TruncatedNormal(loc=self.set_R0, scale=0.1, low=0.1)
        )

        # increments for times 1..T-1 (vector of length T-1)
        eps = numpyro.sample(
            "rw_steps",
            dist.Normal(loc=0.0, scale=0.005).expand([T - 1])
        )

        # baseline log R_t over time (shared across regions)
        # log R_base_0 = log(R0)
        # log R_base_t = log R_base_{t-1} + eps_{t-1}
        logR_base_0 = jnp.log(R0)
        logR_base_t = logR_base_0 + jnp.concatenate(
            [jnp.zeros(1), jnp.cumsum(eps)]
        )  # shape (T,)

        # broadcast baseline across regions: (T, N)
        logR_all = jnp.repeat(logR_base_t[:, None], N, axis=1)

        # ----- Wastewater effect per region (Option 1:  trend) -----

        if self.ww_obs is not None and len(self.region_ww_obs_list) > 0:
            # one global beta_ww (shared across regions)
            beta_ww = numpyro.sample("beta_ww", dist.Normal(0.01, 0.05))
            beta_ww = jnp.clip(beta_ww, 0, 0.5)

            # assume self.ww_obs and self.ww_quality are lists/arrays
            # indexed in the same order as region_obs_list
            ww_idx = 0  # index into ww_obs / ww_quality

            for region_idx, region in enumerate(self.region_list):
                if region in self.region_ww_obs_list:
                    # region-specific lag with unique name

                    ww = jnp.asarray(self.ww_obs[ww_idx], dtype=jnp.float32)

                    # --- build a simple quality measure from smoothed WW slope (all JAX) ---
                    ww_diff = jnp.diff(ww, prepend=ww[0])
                    q = (ww_diff > 0.0).astype(jnp.float32)

                    # ensure length T (truncate or assume already correct)
                    ww = ww[:T]
                    q = q[:T]

                    # Covariate = first difference, gated by quality > 0.5
                    ww_delta = jnp.concatenate(
                        [jnp.array([0.0], dtype=jnp.float32),
                         jnp.diff(ww)]
                    )
                    ww_cov = ww_delta * q  # shape (T,)

                    # Apply temporal delay: guard against delay_ww=0 (ww_cov[:-0] == ww_cov[:0] is empty)
                    if self.delay_ww > 0:
                        ww_cov = jnp.concatenate(
                            [jnp.zeros(self.delay_ww, dtype=jnp.float32),
                             ww_cov[:-self.delay_ww]]
                        )

                    ww_cov_full = jnp.concatenate(
                        [jnp.zeros(self.n_previous_points, dtype=jnp.float32),
                         ww_cov]
                    )

                    # Pad to length T with zeros if needed (e.g., during prediction with extended horizon)
                    if ww_cov_full.shape[0] < T:
                        ww_cov_full = jnp.concatenate(
                            [ww_cov_full, jnp.zeros(T - ww_cov_full.shape[0], dtype=jnp.float32)]
                        )

                    # add wastewater drift to log R for this region
                    logR_all = logR_all.at[:, region_idx].add(beta_ww * ww_cov_full)

                # if region not in region_obs_list, logR_all stays as baseline

        # ----- Final R_t -----
        R_t = jnp.clip(jnp.exp(logR_all), 0.01, 3.0)  # (T, N)

        numpyro.deterministic("R_t", R_t)
        return R_t

    def latent_infection_func(self, R, I0, T, N_regions):
        """
        Renewal Model (vectorized over regions):

        I_t,i = R_t * sum_{tau < t} I_{t-1-tau, i} * g_tau
                + spatial spillover
        """
        eta_spatial = numpyro.sample("eta_spatial", dist.Normal(loc=0.001, scale=0.005))


        # I[t, i] = infections at time t in region i
        I = jnp.zeros((T, N_regions))
        I = I.at[0, :].set(I0 / self.region_pop)

        A = self.adj_matrix  # shape (N_regions, N_regions): A[i,j] = weight from j -> i

        for t in range(1, T):
            # spatial component: eta_spatial * (A @ I[t-1])
            neighbor_inf = eta_spatial * (A @ I[t - 1])  # (N_regions,)
            I_t = R[t] * I[t - 1] + neighbor_inf  # (N_regions,)
            I_t = jnp.clip(I_t, 0.0, 1.0)
            I = I.at[t, :].set(I_t)

        I=I*self.region_pop
        numpyro.deterministic("I_t", I)
        return I

    def Hosptial_admission_model(self, T, I, hosp_obs_v=None):
        """
            Hierarchical P_hosp:
              mu_log_P_hosp ~ Normal(log(0.2), 0.5)
              sigma_log_P_hosp ~ HalfNormal(0.5)
              log_P_hosp[i] ~ Normal(mu_log_P_hosp, sigma_log_P_hosp)
              P_hosp[i] = exp(log_P_hosp[i])
            """

        d = self.inf_hosp_array

        # ---- hyperpriors ----
        mu_log_P_hosp = numpyro.sample(
            "mu_log_P_hosp",
            dist.Normal(jnp.log(self.set_P_hosp), 0.5)
        )
        sigma_log_P_hosp = numpyro.sample(
            "sigma_log_P_hosp",
            dist.HalfNormal(0.5)
        )

        # ---- region-level parameters ----
        log_P_hosp = numpyro.sample(
            "log_P_hosp",
            dist.Normal(mu_log_P_hosp, sigma_log_P_hosp).expand([self.N_regions])
        )
        P_hosp = jnp.clip(jnp.exp(log_P_hosp), 0.0, 1.0)

        # ---- convolution + selection of observed regions (vectorized) ----


        def conv_region(x):
            return jnp.convolve(x, d, mode="full")

        conv_all = vmap(conv_region, in_axes=1, out_axes=1)(I)

        start = self.n_previous_points
        conv_trim = conv_all[start:T, :]  # (T, N_regions)

        H_full = (P_hosp[None, :] * conv_trim).T  # (N_regions, T)
        numpyro.deterministic("H_full", H_full)

        idx_obs = [self.region_list.index(r) for r in self.region_obs_list]
        H = H_full[idx_obs, :]  # (N_obs_regions, T)

        numpyro.deterministic("H_t", H)

        k_hosp = numpyro.sample(
            "k_hosp",
            dist.LogNormal(jnp.log(0.5), jnp.log(10.0))
        )

        numpyro.sample("hosp_obs",
                       dist.NegativeBinomial2(mean=H, concentration=k_hosp),
                       obs=hosp_obs_v)



    def waste_water_model(self, T, I, ww_obs_v=None):
        """
            Hierarchical G (WW scaling):
              mu_log_G ~ Normal(log(100), 1.0)
              sigma_log_G ~ HalfNormal(1.0)
              log_G[i] ~ Normal(mu_log_G, sigma_log_G)
              G[i] = exp(log_G[i])
            """

        s = self.sheddin

        # ---- hyperpriors ----
        mu_log_G = numpyro.sample(
            "mu_log_G",
            dist.Normal(jnp.log(self.set_G), 1.0)
        )
        sigma_log_G = numpyro.sample(
            "sigma_log_G",
            dist.HalfNormal(1.0)
        )

        # ---- region-level parameters ----
        log_G = numpyro.sample(
            "log_G",
            dist.Normal(mu_log_G, sigma_log_G).expand([self.N_regions])
        )
        G = jnp.exp(log_G)  # WW scale can be > 0 without an upper bound

        # ---- convolution + selection of observed regions (vectorized) ----

        def conv_region(x):
            return jnp.convolve(x, s, mode="full")

        conv_all = vmap(conv_region, in_axes=1, out_axes=1)(I)

        start = self.n_previous_points
        conv_trim = conv_all[start:T, :]  # (T, N_regions)

        W_full = (G[None, :] * conv_trim).T  # (N_regions, T)

        idx_obs = [self.region_list.index(r) for r in self.region_obs_list]
        W = W_full[idx_obs, :]  # (N_obs_regions, T)

        numpyro.deterministic("W_t", W)

        k_ww = numpyro.sample(
            "k_ww",
            dist.LogNormal(jnp.log(0.5), jnp.log(10.0))
        )
        numpyro.sample("ww_obs", dist.Normal(W, k_ww), obs=ww_obs_v)

    def model_test(self, T, hosp_obs_v=None, ww_obs_v=None):
        R = self.R_t_func(T)
        I0 = numpyro.sample("I0", dist.LogNormal(jnp.log(self.set_I0), jnp.log(1.5)))

        I = self.latent_infection_func(R, I0, T, self.N_regions)

        if "hosp_obs" in self.cols_concern:
            self.Hosptial_admission_model(T, I, hosp_obs_v)

        if "ww_obs" in self.cols_concern:
            self.waste_water_model(T, I, ww_obs_v)


    def run_mcmc(self, init_params=None):
        rng = jr.key(np.random.randint(0, 2**31 - 1))
        rng, sub = random.split(rng)

        kernel = NUTS(self.model_test)
        mcmc = MCMC(kernel,
                    num_warmup=self.num_warmup,
                    num_samples=self.num_samples,
                    num_chains=self.num_chains,
                    progress_bar=self.progress_bar,
                    chain_method="parallel")

        T_train = self.T + self.n_previous_points

        if init_params is not None:
            mcmc.run(rng, T=T_train, hosp_obs_v=self.hosp_obs, ww_obs_v=self.ww_obs, init_params=init_params)
        else:
            mcmc.run(rng, T=T_train, hosp_obs_v=self.hosp_obs, ww_obs_v=self.ww_obs)

        if self.print_summary:
            mcmc.print_summary()

        samples = mcmc.get_samples()
        pred = Predictive(self.model_test, samples)(sub, T=self.T + self.n_forecast_points + self.n_previous_points)

        self.posterior_samples = samples
        self.post_pred = pred
        return samples, pred
