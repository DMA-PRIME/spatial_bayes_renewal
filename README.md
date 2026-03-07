# Spatial Bayesian Renewal Models for Infectious Disease Forecasting 

Comprehensive probabilistic framework for forecasting infectious disease dynamics using Bayesian renewal equations with wastewater surveillance data.

This package provides two complementary model variants:
- **ClassicalForecaster**: Non-spatial model for single or aggregated regions
- **SpatialForecaster**: Spatial model for multiple connected regions with human mobility/commuting

## Installation

### Install from source

```bash
# Clone or download
cd spatial_bayes_renewal
pip install -e .
```

### Install from PyPI (when available)

```bash
pip install spatial-bayes-renewal
```

## Quick Start

### Classical Model (Single Region)

```python
from spatial_bayes_renewal import ClassicalForecaster

# Initialize with single-region data
forecaster = ClassicalForecaster(
    df_data_train=your_data,
    cols_concern=['hosp_obs', 'ww_obs'],
    n_forecast_points=14,
    data_path='/path/to/data',
    pop=1000000,  # population size
    Renewal_infection_case='Basic',  # 'Basic', 'Feedback', or 'Logistic_S'
    num_samples=200,
    num_warmup=100,
    num_chains=2
)

# Set required parameters
forecaster.gen_int_array = generation_interval  # shape (T,)
forecaster.inf_hosp_array = inf_to_hosp_delay   # shape (D,)
forecaster.sheddin = viral_shedding_kinetics     # shape (S,)

# Run inference
samples, predictions = forecaster.run_mcmc()
```

### Spatial Model (Multiple Regions)

```python
from spatial_bayes_renewal import SpatialForecaster
import networkx as nx

# Create spatial network (directed graph of regions)
G = nx.DiGraph()
G.add_edge('region1', 'region2', weight=0.5)
G.add_edge('region2', 'region3', weight=0.3)
# ... add more edges

# Initialize with multi-region data
forecaster = SpatialForecaster(
    df_data=your_data,  # columns: 'county', 'hosp_obs', 'ww_obs'
    spatial_net=G,
    region_list=['region1', 'region2', 'region3'],
    region_obs_list=['region1', 'region2', 'region3'],  # regions with observations
    cols_concern=['hosp_obs', 'ww_obs'],
    n_forecast_points=14,
    data_path='/path/to/data',
    num_samples=200,
    num_warmup=100,
    num_chains=1
)

# Set required parameters
forecaster.gen_int_array = generation_interval
forecaster.inf_hosp_array = inf_to_hosp_delay
forecaster.sheddin = viral_shedding_kinetics
forecaster.region_pop = np.array([pop1, pop2, pop3])  # population per region

# Run inference
samples, predictions = forecaster.run_mcmc()
```

## Model Details

### Classical Renewal Model

The classical model implements the renewal equation:

$$I(t) = R(t) \sum_{\tau=0}^{t-1} I(\tau) g(t-\tau)$$

where:
- $I(t)$ = infections at time $t$
- $R(t)$ = time-varying reproduction number
- $g(\tau)$ = generation interval distribution

**Infection Case Options:**
- **Basic**: Simple renewal equation
- **Feedback**: Includes negative feedback based on prior infections
- **Logistic_S**: Accounts for susceptible depletion in finite populations

### Spatial Renewal Model

Extends classical model with spatial spillover:

$$I_{t,i} = R_{t,i} \sum_{\tau < t} I_{t-\tau,i} g(\tau) + \eta_{spatial} \sum_{j} A_{ij} I_{t-1,j}$$

where:
- $A_{ij}$ = spatial adjacency matrix (connectivity weight from region $j$ to $i$)
- $\eta_{spatial}$ = spatial spillover rate parameter
- Separate $R_t$ evolves per region with region-specific wastewater effects

### Observation Models

**Hospital Admissions:**
$$h(t) \sim \text{NegativeBinomial2}(\mu=P_{hosp} * (I \otimes d), k)$$

**Wastewater:**
$$w(t) \sim \text{Normal}(\mu=G * (I \otimes s), \sigma=k_{ww})$$

where:
- $d$ = infection-to-hospitalization delay distribution
- $s$ = viral shedding kinetics
- $\otimes$ = convolution operator
- $P_{hosp}$, $G$ = hierarchical parameters with hyperpriors

## Data Requirements

### Classical Model Input
```python
df_data_train: DataFrame with columns
  - 'hosp_obs': Hospital admissions (1D array)
  - 'ww_obs': Wastewater concentration (1D array)
  - 'ed_obs': (optional) Emergency department visits
```

### Spatial Model Input
```python
df_data: DataFrame with columns
  - 'county': Region identifier
  - 'hosp_obs': Hospital admissions per region
  - 'ww_obs': Wastewater concentration per region
  - 'date': (optional) Temporal index
```

## Key Parameters

### Both Models
- `cols_concern`: List of observation types to include
- `n_forecast_points`: Number of time steps to forecast ahead
- `num_samples`: MCMC samples to draw (default: 200)
- `num_warmup`: MCMC burn-in iterations (default: 100)
- `num_chains`: Parallel chains to run (default: 2 for classical, 1 for spatial)

### Classical-specific
- `pop`: Total population size
- `Renewal_infection_case`: Infection dynamics model type

### Spatial-specific
- `spatial_net`: NetworkX graph of region connectivity
- `region_list`: All regions in model
- `region_obs_list`: Subset with observations (creates hierarchical structure)
- `region_pop`: Population per region (array)

## Model Comparison

| Feature | Classical | Spatial |
|---------|-----------|---------|
| Regions | Single/aggregate | Multiple |
| Spatial Effects | No | Yes (adjacency matrix) |
| Wastewater | Yes | Yes (per region) |
| Scalability | Very high | High (linear in regions) |
| Inference Speed | Fast | Medium |
| Complexity | Simple | Moderate |

**Choose Classical when:**
- Analyzing a single city/region
- Need fast inference with limited data
- Spatial effects are negligible

**Choose Spatial when:**
- Multiple regions with connections
- Need to account for cross-region transmission
- Have region-level wastewater data
- Building a regional forecasting system

## Output

`run_mcmc()` returns:
```python
samples, predictions = forecaster.run_mcmc()

# samples: Dictionary of posterior samples
#   Keys: 'R0', 'R_t', 'I_t', 'P_hosp', 'G', 'k_hosp', 'k_ww', etc.
#   Values: Arrays of shape (num_chains * num_samples, ...)

# predictions: Posterior predictive samples for forecast period
#   Keys: 'I_t', 'H_t', 'W_t', 'R_t', etc.
#   Values: Arrays of shape (num_samples, forecast_len, ...)
```

## Usage Examples

### Plot Hospital Admissions Forecast

```python
# Classical model
forecaster.plot_results(
    df_data_all=your_data,
    strx='hosp_obs',
    strx_label='Hospital Admissions',
    fig_path='forecast.png',
    show_forecast=True,
    show_log=False
)
```

### Extract Posterior Summary

```python
import numpy as np

# Get posterior R(t) estimates
R_post = samples['R_t']
R_median = np.median(R_post, axis=0)
R_hpd = np.percentile(R_post, [2.5, 97.5], axis=0)

print(f"R(t) = {R_median} [{R_hpd[0]}, {R_hpd[1]}]")
```

### Warm-start with Previous Results

```python
# Use previous posterior samples as initialization
samples, pred = forecaster.run_mcmc(init_params=previous_samples)
```

## Requirements

- Python >= 3.9
- numpy
- polars
- jax >= 0.3.0
- jaxlib >= 0.3.0
- numpyro >= 0.10.0
- matplotlib

## Contributing

Contributions welcome! Areas for enhancement:
- Additional infection dynamics models
- GPU acceleration
- Approximate Bayesian computation (ABC) variants
- More flexible spatial models
- Joint inference across diseases

## Citation

If you use this package, please cite:

```bibtex
@software{luzhong2024bayesian,
  title={Bayesian Renewal Models for Infectious Disease Forecasting},
  author={LuZhong},
  year={2024},
  url={https://github.com/...}
}
```

## References

- Gostic et al. (2020). "Potential for large outbreaks of coronavirus disease 2019." *The Lancet Infectious Diseases*.
- Viboud & Vall-Spasos (2021). "Ranking the effectiveness of worldwide COVID-19 government interventions."
- Bhatt et al. (2008). "The Global Distribution and Burden of Dengue." *Nature*.

## License

MIT License - see LICENSE file for details

## Support

For issues, questions, or feature requests, please open an issue on GitHub or contact the development team.
