"""
USAGE EXAMPLES - Both Models

This file demonstrates how to use both ClassicalForecaster and SpatialForecaster
from the spatial_bayes_renewal package.
"""

# ============================================================================
# EXAMPLE 1: CLASSICAL FORECASTER (Single Region / Aggregated)
# ============================================================================

import numpy as np
import polars as pl
from spatial_bayes_renewal import ClassicalForecaster

# Load your data
df_data = pl.read_csv("your_data.csv")

# Create forecaster instance
classical_forecaster = ClassicalForecaster(
    df_data_train=df_data,
    cols_concern=['hosp_obs', 'ww_obs'],  # Which observations to use
    n_forecast_points=14,  # Forecast 2 weeks ahead
    data_path="/path/to/data",
    pop=1_000_000,  # Population size
    Renewal_infection_case='Basic',  # Options: 'Basic', 'Feedback', 'Logistic_S'
    num_samples=200,
    num_warmup=100,
    num_chains=2,
    progress_bar=True,
    print_summary=True
)

# Set required distribution parameters
classical_forecaster.gen_int_array = generation_interval  # shape (T,)
classical_forecaster.inf_hosp_array = infection_to_hosp   # shape (D,)
classical_forecaster.sheddin = viral_shedding              # shape (S,)

# Optionally adjust default priors
classical_forecaster.set_I0 = 100      # Initial infections
classical_forecaster.set_R0 = 1.2      # Basic reproduction number
classical_forecaster.set_P_hosp = 0.2  # Proportion hospitalized
classical_forecaster.set_G = 100       # WW scaling factor

# Run MCMC inference
samples, predictions = classical_forecaster.run_mcmc()

# Access results
R_t_samples = samples['R_t']
I_t_samples = samples['I_t']
H_t_predictions = predictions['H_t']  # Hospital predictions

# Plot results
classical_forecaster.plot_results(
    df_data_all=df_data,
    strx='hosp_obs',
    strx_label='Hospital Admissions',
    fig_path='classical_forecast.png',
    show_forecast=True,
    show_log=False
)


# ============================================================================
# EXAMPLE 2: SPATIAL FORECASTER (Multiple Regions)
# ============================================================================

import networkx as nx
from spatial_bayes_renewal import SpatialForecaster

# Load multi-region data
df_spatial_data = pl.read_csv("regional_data.csv")

# Create spatial network (directed graph of regions)
G = nx.DiGraph()
# Add edges with weights representing spillover strength
G.add_edge('SC_central', 'SC_lowcountry', weight=0.3)
G.add_edge('SC_lowcountry', 'SC_coastal', weight=0.2)
G.add_edge('SC_central', 'SC_upstate', weight=0.15)
# ... add more edges based on your geography

# Create forecaster instance
spatial_forecaster = SpatialForecaster(
    df_data=df_spatial_data,
    spatial_net=G,
    region_list=['SC_central', 'SC_lowcountry', 'SC_coastal', 'SC_upstate'],
    region_obs_list=['SC_central', 'SC_lowcountry', 'SC_coastal', 'SC_upstate'],
    cols_concern=['hosp_obs', 'ww_obs'],
    n_forecast_points=14,
    data_path="/path/to/data",
    num_samples=200,
    num_warmup=100,
    num_chains=1,  # Spatial model uses fewer chains
    progress_bar=True,
    print_summary=True
)

# Set required parameters
spatial_forecaster.gen_int_array = generation_interval
spatial_forecaster.inf_hosp_array = infection_to_hosp
spatial_forecaster.sheddin = viral_shedding

# IMPORTANT: Set regional populations
spatial_forecaster.region_pop = np.array([
    1_000_000,  # SC_central population
    800_000,    # SC_lowcountry population
    600_000,    # SC_coastal population
    900_000,    # SC_upstate population
])

# Adjust default priors if needed
spatial_forecaster.set_I0 = 1000
spatial_forecaster.set_R0 = 1.2
spatial_forecaster.set_P_hosp = 0.1
spatial_forecaster.set_G = 1.0

# Run MCMC inference
samples, predictions = spatial_forecaster.run_mcmc()

# Access spatial results
R_t_by_region = samples['R_t']  # shape: (samples, time, regions)
infection_by_region = samples['I_t']  # shape: (samples, time, regions)
spatial_spillover = samples['eta_spatial']  # shape: (samples,)

# Get predictions per region
H_t_pred = predictions['H_t']  # Hospital admissions
W_t_pred = predictions['W_t']  # Wastewater


# ============================================================================
# EXAMPLE 3: SWITCHING BETWEEN MODELS
# ============================================================================

def choose_model(num_regions, use_spatial_network=False):
    """
    Helper function to select appropriate model based on data.
    """
    if num_regions == 1 or not use_spatial_network:
        return 'classical'
    else:
        return 'spatial'

# Determine model based on data
num_regions = len(df_data['county'].unique())
model_type = choose_model(num_regions, use_spatial_network=True)

if model_type == 'classical':
    print("Using ClassicalForecaster for single region")
    # ... initialize and run ClassicalForecaster
else:
    print("Using SpatialForecaster for multiple regions")
    # ... initialize and run SpatialForecaster


# ============================================================================
# EXAMPLE 4: UPDATING WITH NEW DATA
# ============================================================================

# Update training data and re-run
new_df_data = pl.read_csv("updated_data.csv")
classical_forecaster.update_training_data(new_df_data)
samples, predictions = classical_forecaster.run_mcmc()

# For spatial model
spatial_forecaster.update_training_data(new_df_data)
samples, predictions = spatial_forecaster.run_mcmc()


# ============================================================================
# EXAMPLE 5: WARM-START INITIALIZATION
# ============================================================================

# Use previous inference results to initialize new runs
print("First run...")
samples_1, pred_1 = forecaster.run_mcmc()

print("Warm-start with previous samples...")
# Update data
forecaster.update_training_data(new_data)

# Run with initialization from previous posterior
samples_2, pred_2 = forecaster.run_mcmc(init_params=samples_1)


# ============================================================================
# EXAMPLE 6: POSTERIOR ANALYSIS
# ============================================================================

import matplotlib.pyplot as plt

# Extract and analyze posterior samples
def analyze_reproduction_number(samples):
    """Analyze time-varying reproduction number estimates."""
    R_t = samples['R_t']
    
    # Compute credible intervals
    R_median = np.median(R_t, axis=0)
    R_hpd_lower = np.percentile(R_t, 2.5, axis=0)
    R_hpd_upper = np.percentile(R_t, 97.5, axis=0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    time = np.arange(len(R_median))
    
    ax.plot(time, R_median, 'b-', label='Median R(t)', linewidth=2)
    ax.fill_between(time, R_hpd_lower, R_hpd_upper, 
                     alpha=0.3, label='95% Credible Interval')
    ax.axhline(y=1.0, color='r', linestyle='--', label='R=1 (threshold)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Reproduction Number R(t)')
    ax.legend()
    ax.set_title('Estimated Time-Varying Reproduction Number')
    plt.tight_layout()
    plt.savefig('reproduction_number.png', dpi=300)
    
    return R_median, R_hpd_lower, R_hpd_upper

# Run analysis
R_median, R_lower, R_upper = analyze_reproduction_number(samples)


# ============================================================================
# EXAMPLE 7: MODEL COMPARISON (Classical vs Spatial)
# ============================================================================

def compare_models(df_data, spatial_net=None):
    """
    Compare classical and spatial models on same data.
    """
    
    # Run classical model
    classical_fc = ClassicalForecaster(...)
    samples_classical, pred_classical = classical_fc.run_mcmc()
    
    # Run spatial model if network provided
    if spatial_net is not None:
        spatial_fc = SpatialForecaster(...)
        samples_spatial, pred_spatial = spatial_fc.run_mcmc()
        
        # Compare predictions
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Classical
        axes[0].plot(np.mean(pred_classical['H_t'], axis=0))
        axes[0].set_title('Classical Model Forecast')
        
        # Spatial (aggregate)
        spatial_agg = np.sum(pred_spatial['H_t'], axis=(0, 2))  # Sum regions
        axes[1].plot(np.mean(spatial_agg, axis=0))
        axes[1].set_title('Spatial Model Forecast')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300)
        
        return samples_classical, samples_spatial
    
    return samples_classical, None


# ============================================================================
# REQUIREMENTS TO SET BEFORE RUNNING
# ============================================================================

"""
Before calling run_mcmc(), ensure you've set:

1. gen_int_array
   - Generation interval distribution
   - Shape: (num_days,)
   - Values should sum to 1
   Example:
   from scipy.stats import lognorm
   gen_int_array = lognorm.pdf(np.arange(30), s=0.83, scale=np.exp(1.63))
   gen_int_array /= gen_int_array.sum()

2. inf_hosp_array
   - Infection to hospitalization delay distribution
   - Shape: (num_delay_days,)
   Example:
   inf_hosp_array = lognorm.pdf(np.arange(14), s=0.5, scale=np.exp(2.0))
   inf_hosp_array /= inf_hosp_array.sum()

3. sheddin (or shedding)
   - Viral shedding kinetics
   - Shape: (num_shedding_days,)
   Example:
   sheddin = lognorm.pdf(np.arange(30), s=0.5, scale=np.exp(1.0))
   sheddin /= sheddin.sum()

4. region_pop (spatial model only)
   - Array of population sizes per region
   - Shape: (num_regions,)
   Example:
   spatial_forecaster.region_pop = np.array([pop1, pop2, pop3])
"""
