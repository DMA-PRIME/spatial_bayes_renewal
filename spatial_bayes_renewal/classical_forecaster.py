import numpy as np
import polars as pl
from pathlib import Path
import jax.numpy as jnp
import jax.random as jr
import numpyro
import jax.random as random
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
import matplotlib.pyplot as plt
from jax.nn import sigmoid
import math
from jax import lax


class ClassicalForecaster:
    """
    Classical (non-spatial) Bayesian Renewal Model for forecasting infectious disease dynamics.
    
    This model captures disease transmission in a single or aggregated region with various
    infection dynamics options (Basic, Feedback, Logistic_S).
    """
    
    def __init__(self, df_data_train,
                 cols_concern,  # columns for observed data
                 n_forecast_points,
                 data_path,
                 pop,
                 Renewal_infection_case='Basic',  ##Adjust_S, Feedback,Logistic_S
                 num_samples=100,
                 num_warmup=100,
                 num_chains=2,
                 progress_bar=True,
                 print_summary=True,
                 ):
        self.cols_concern = cols_concern

        self.Renewal_infection_case = Renewal_infection_case
        self.n_forecast_points = int(n_forecast_points)
        self.data_path = data_path

        self.hosp_obs = None
        self.ed_obs = None
        self.ww_obs = None

        self.sheddin = None
        self.gen_int_array = None
        self.inf_hosp_array = None
        self.feedback_pmf = None
        self.pop = pop
        self.n_previous_points = 10
        self.delay_ww=0

        for col in cols_concern:
            if col == 'hosp_obs':
                self.hosp_obs = (df_data_train[col].to_numpy()).astype(np.float32).ravel()
                self.T = len(self.hosp_obs)
            if col == 'ed_obs':
                self.ed_obs = (df_data_train[col].to_numpy()).astype(np.float32).ravel()
                self.T = len(self.ed_obs)
            if col == 'ww_obs':
                self.ww_obs = (df_data_train[col].to_numpy()).astype(np.float32).ravel()
                self.T = len(self.ww_obs)

        # rng + mcmc cfg
        self.num_samples = int(num_samples)
        self.num_warmup = int(num_warmup)
        self.num_chains = int(num_chains)
        self.progress_bar = bool(progress_bar)
        self.print_summary=bool(print_summary)

        ##save temporary results
        self.R_t_input=None
        self.R_t = None
        self.I_t = None

        ###set prior set
        self.set_I0=100
        self.set_R0=1.2
        self.set_P_hosp=0.2
        self.set_G=100
        # define mean vector for each time step (can be zeros or dynamic)
        self.set_eps = jnp.zeros(self.T +self.n_previous_points)  # or some other function of time

    def update_training_data(self, df_data_train):
        """Update internal arrays when new training data arrive."""
        self.df_data_train = df_data_train

        for col in self.cols_concern:
            if col == 'hosp_obs':
                self.hosp_obs = df_data_train[col].to_numpy().astype(np.float32).ravel()
                self.T = len(self.hosp_obs)
            if col == 'ed_obs':
                self.ed_obs = df_data_train[col].to_numpy().astype(np.float32).ravel()
                self.T = len(self.ed_obs)
            if col == 'ww_obs':
                self.ww_obs = df_data_train[col].to_numpy().astype(np.float32).ravel()
                self.T = len(self.ww_obs)

    def R_t_func(self, T):
        R0 = numpyro.sample("R0", dist.TruncatedNormal(loc=self.set_R0, scale=0.1, low=0.0))
        eps = numpyro.sample("rw_steps", dist.Normal(loc=self.set_eps, scale=0.005))

        beta_ww = numpyro.sample("beta_ww", dist.Normal(0.01, 0.05))
        beta_ww = jnp.clip(beta_ww, 0, 0.5)

        R_t = jnp.zeros((T,))
        R_t = R_t.at[0].set(R0)
   
        if self.ww_obs is not None:
            ww = jnp.asarray(self.ww_obs, dtype=jnp.float32)

            # --- build a simple quality measure from smoothed WW slope (all JAX) ---
            ww_diff = jnp.diff(ww, prepend=ww[0])
            q = (ww_diff > 0.0).astype(jnp.float32)

            # ensure length T (truncate or assume already correct)
            ww = ww[:T]
            q = q[:T]

            ww_delta = jnp.concatenate(
                [jnp.array([0.0], dtype=jnp.float32),
                 jnp.diff(ww)]
            )
            ww_cov= ww_delta * q  # shape (T,)

            ww_cov = jnp.concatenate(
                [jnp.zeros(self.delay_ww, dtype=jnp.float32),
                 ww_cov[:-self.delay_ww]]
            )

            ww_cov = jnp.concatenate(
                [jnp.zeros(self.n_previous_points, dtype=jnp.float32),
                 ww_cov]
            )

        else:
            ww_cov = jnp.zeros(T, dtype=jnp.float32)

        for t in range(1, T):
            drift =  beta_ww*ww_cov[t]
            R_t_here = jnp.clip(jnp.exp(jnp.log(R_t[t - 1])) + eps[t - 1]+ drift , 0.01, 3.0)
            R_t = R_t.at[t].set(R_t_here)

        numpyro.deterministic("R_t", R_t)
        return R_t



    def latent_infection_func(self, R, I0, T):
        I = jnp.zeros((T,))
        I = I.at[0].set(I0)

        if self.Renewal_infection_case == 'Basic':
            '''
            Renewal Model:
            I(t)=R(t) \times \sum_{\tau\lt t} I(\tau) g(t-\tau)
            '''
            for t in range(1, T):
                infectivity = 0
                for tau in range(len(self.gen_int_array)):
                    if tau < t:
                        infectivity = infectivity + I[t - tau - 1] * self.gen_int_array[tau]
                I = I.at[t].set(R[t] * infectivity)


        if self.Renewal_infection_case == 'Feedback':
            '''
            Renewal Model:
            I(t)=R(t) \times \sum_{\tau\lt t} I(t-\tau) g(-\tau) ;
            R(t)=R^u(t)exp(-\gamma(t)\sum_{\tau}  I(I-\tau) f(\tau)) ;
            '''

            gamma = numpyro.sample("infection_feedback_strength", dist.Normal(0.01, 0.01))
            gamma = jnp.clip(gamma, a_min=0.0001, a_max=1)
            R_t_adj = jnp.zeros((T,))
            for t in range(1, T):
                feedback_term = 0
                for tau in range(len(self.feedback_pmf)):
                    if tau < t:
                        feedback_term = feedback_term + I[t - tau - 1] * self.feedback_pmf[tau]

                R_t_here = jnp.clip(R[t] * jnp.exp(-gamma * feedback_term), a_min=0.01, a_max=3)
                R_t_adj = R_t_adj.at[t].set(R_t_here)

                infectivity = 0
                for tau in range(len(self.gen_int_array)):
                    if tau < t:
                        infectivity = infectivity + I[t - tau - 1] * self.gen_int_array[tau]
                I = I.at[t].set(R_t_adj[t] * infectivity)

        if self.Renewal_infection_case == "Logistic_S":

            I_raw = jnp.zeros((T,))
            I_raw = I_raw.at[0].set(I0)
            for t in range(1, T):
                infectivity = 0
                for tau in range(len(self.gen_int_array)):
                    if tau < t:
                        infectivity = infectivity + I_raw[t - tau - 1] * self.gen_int_array[tau]

                I_raw = I_raw.at[t].set(R[t] * infectivity)

            I = jnp.zeros((T,))
            for t in range(0, T):
                if t == 0:
                    infectivity = (self.pop) * (1 - jnp.exp(-I_raw[t] / self.pop))
                else:
                    infectivity = (self.pop - I_raw[t - 1]) * (1 - jnp.exp(-I_raw[t] / self.pop))
                I = I.at[t].set(I_raw[t])

        numpyro.deterministic("I_t", I)
        return I

    def Hosptial_admission_model(self, T, I, hosp_obs_v=None,weight=None):
        '''
        Hospital Admission Model:
        h(t)~NegativeBinomial (mean=H(t), concentration=k) ;
        H(t)=P_{hosp}(t) \sum_\tau d(\tau) I(t-\tau) ;
        log[P_hosp(t)] \sim Normal(\mu=log(0.005),\sigma=log(1,1)) ;
        log[k] \sim  Normal(\mu=log(1),\sigma=log(10)) ;

        '''
        ###latent hosptial admission
        P_hosp = numpyro.sample("P_hosp", dist.LogNormal(loc=jnp.log(self.set_P_hosp), scale=jnp.log(1.1)))
        P_hosp = jnp.clip(P_hosp, a_min=0, a_max=1)

        H = P_hosp * jnp.convolve(I, self.inf_hosp_array, mode="full")
        H = H[self.n_previous_points:-len(self.inf_hosp_array) + 1]

        numpyro.deterministic("H_t", H)

        k_hosp = numpyro.sample("k_hosp", dist.LogNormal(loc=jnp.log(0.5), scale=jnp.log(10)))
        if weight is not None and hosp_obs_v is not None:
            log_lik = dist.NegativeBinomial2(mean=H, concentration=k_hosp).log_prob(hosp_obs_v)
            numpyro.factor("hosp_downweighted", weight * jnp.sum(log_lik))
        else:
            numpyro.sample("hosp_obs", dist.NegativeBinomial2(mean=H, concentration=k_hosp), obs=hosp_obs_v)


    def waste_water_model(self, T, I, ww_obs_v=None,weight=None):
        '''
        Waste water Model:
        log(c_t) ~ Normal (C(t),sigma_c) ;
        C(t)=G \sum_\tau s(\tau) I(t-\tau) ;
        '''
        G = numpyro.sample("G", dist.LogNormal(jnp.log(self.set_G), 0.2))

        W = G * jnp.convolve(I, self.sheddin, mode="full")
        W = jnp.log(W) / jnp.log(2)
        W = W[self.n_previous_points:-len(self.sheddin) + 1]
        numpyro.deterministic("W_t", W)

        k_ww = numpyro.sample("k_ww", dist.LogNormal(loc=jnp.log(0.5), scale=jnp.log(10)))

        if weight is not None and ww_obs_v is not None:
            q_raw = numpyro.sample("q_raw_ww", dist.Normal(0.5, 0.5).expand([W.shape[0]]))
            q_t = jnp.clip(q_raw,0,1)

            log_lik = dist.Normal(loc=W, scale=k_ww).log_prob(ww_obs_v)
            numpyro.factor("ww_downweighted", weight * jnp.sum(q_t*log_lik))
        else:
            numpyro.sample("ww_obs", dist.Normal(loc=W, scale=k_ww), obs=ww_obs_v)




    def model_test(self, T, R_input=None,hosp_obs_v=None, ww_obs_v=None):
        if R_input==None:
            self.R_t  = self.R_t_func(T)
        else:
            self.R_t = R_input
            numpyro.sample("R_t", dist.Normal(loc=R_input, scale=0.1))

        I0 = numpyro.sample("I0", dist.LogNormal(loc=jnp.log(self.set_I0), scale=jnp.log(1.75)))
        self.I_t = self.latent_infection_func(self.R_t, I0, T)


        if 'ww_obs' in self.cols_concern:
            self.waste_water_model(T, self.I_t, ww_obs_v,weight=0.5)

        if 'hosp_obs' in  self.cols_concern:
            self.Hosptial_admission_model(T, self.I_t, hosp_obs_v,weight=0.8)


    def run_mcmc(self,init_params=None):

        rand_int = np.random.randint(np.iinfo(np.int32).min, np.iinfo(np.int32).max)
        rng_key = jr.key(rand_int)
        rng_key, subkey = random.split(rng_key)

        kernel = NUTS(self.model_test, target_accept_prob=0.95, max_tree_depth=8)
        mcmc = MCMC(kernel,
                    num_warmup=self.num_warmup,
                    num_samples=self.num_samples,
                    num_chains=self.num_chains,
                    progress_bar=self.progress_bar,
                    chain_method="parallel")

        if init_params is not None:
            print("Using warm-start initialization...")
            mcmc.run(rng_key,
                     T=self.T + self.n_previous_points,
                     R_input=self.R_t_input,
                     hosp_obs_v=self.hosp_obs,
                     ww_obs_v=self.ww_obs,
                     init_params=init_params)
        else:
            mcmc.run(rng_key,
                     T=self.T + self.n_previous_points,
                     R_input=self.R_t_input,
                     hosp_obs_v=self.hosp_obs,
                     ww_obs_v=self.ww_obs)


        if self.print_summary==True:
            mcmc.print_summary()
        posterior_samples = mcmc.get_samples()

        predictive_post = Predictive(self.model_test, posterior_samples)
        post_pred = predictive_post(subkey, T=self.T + self.n_forecast_points + self.n_previous_points)

        self.posterior_samples = posterior_samples
        self.post_pred = post_pred

        return self.posterior_samples, self.post_pred

    def plot_results(self, df_data_all, strx, strx_label, fig_path, show_forecast, show_log):
        print('cols_concern', self.cols_concern)
        pred_mean = np.mean(self.post_pred[strx], axis=0)
        pred_median = np.percentile(self.post_pred[strx], 50, axis=0)
        pred_hpd = np.percentile(self.post_pred[strx], [2.5, 97.5], axis=0)

        fig, ax = plt.subplots(figsize=(6, 4))

        time = np.arange(len(df_data_all))

        ax.scatter(time, df_data_all[strx], label="Data", color='black')
        if show_forecast == True:
            ax.plot(time, pred_mean, label="Predict Mean", color='C0')
            ax.fill_between(time, pred_hpd[0], pred_hpd[1], color="C0", alpha=0.3,
                            label="Predict (95% credible interval)")

        ax.axvline(x=int(self.T), color='k', linestyle='--', label='Test/Training split')
        ax.legend()
        ax.set_xlabel("Time", fontsize=10)
        ax.set_ylabel(strx_label, fontsize=10)

        if show_log == True:
            ax.set_yscale('log')
        ax.set_title(strx_label)

        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path)

    def plot_posterior_results(self, strx, strx_label, fig_path):
        pred_mean = np.mean(self.posterior_samples[strx], axis=0)
        pred_median = np.percentile(self.posterior_samples[strx], 50, axis=0)
        pred_hpd = np.percentile(self.posterior_samples[strx], [2.5, 97.5], axis=0)
        fig, ax = plt.subplots(figsize=(6, 4))

        time = np.arange(self.T)
        n_cut = self.n_previous_points
        ax.plot(time, pred_median[n_cut:], label="Posterior Mean", color='C0')
        ax.fill_between(time, pred_hpd[0][n_cut:], pred_hpd[1][n_cut:], color="C0", alpha=0.3,
                        label="Posterior (95% credible interval)")
        ax.set_xlabel("Time", fontsize=10)
        ax.set_ylabel(strx_label, fontsize=10)
        ax.set_title(strx_label)
        ax.set_xlim(7, len(time))
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path)

    def sum_every_n_elements(self,input_list, n):
        """
        Calculates the sum of every n elements in a list.

        Args:
            input_list: The list of numbers to process.
            n: The interval at which to sum the elements.

        Returns:
            A list containing the sums of each n-element sublist.
        """
        if n <= 0:
            raise ValueError("Interval 'n' must be a positive integer.")

        return jnp.array([jnp.sum(input_list[i:i+n]) for i in range(0, len(input_list), n)])
