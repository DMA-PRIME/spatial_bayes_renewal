# Bayesian Renewal Models for Infectious Disease Forecasting

A probabilistic framework for forecasting infectious disease dynamics using Bayesian renewal equations with wastewater surveillance data.

This package provides two complementary model variants:
- **ClassicalForecaster**: Non-spatial model for single or aggregated regions
- **SpatialForecaster**: Spatial model for multiple connected regions with spillover effects

## Features
### Spatial Model
- **Multi-region Dynamics**: Models disease transmission across connected spatial regions
- **Mobility Network**: Captures human movement between adjacent regions 
- **Wastewater Data**: Incorporates WW signals per region to modulate transmission
- **Vectorized Operations**: Efficient JAX-based computation across regions simultaneously

## Installation

### Install from source

```bash
# Clone or download
cd spatial_bayes_renewal
pip install -e .
```

## Quick Start

### Spatial Model (Multiple Regions)

```python
from spatial_bayes_renewal import ClassicalForecaster, SpatialForecaster

# Initialize with multiple-region data
forecaster_spatial = SpatialForecaster(
    df_data=df_train,
    spatial_net=net_mobility,   ### mobility network
    region_list=region_list,    ### region list
    region_obs_list=region_list,  # Use regions for hospital observations
    region_ww_obs_list=region_ww,   # Use  regions for ww observations
    cols_concern=['hosp_obs','ww_obs'], 
    n_forecast_points=n_test,  # 4 weeks ahead
    data_path='.',
    Rt_mode='Local',
    num_samples=200,  # Reduced for speed
    num_warmup=100,
    num_chains=2,  # Use at least 2 chains so R-hat can be checked
    progress_bar=True,
    print_summary=True  # Print MCMC diagnostics (R-hat, n_eff, divergences)
)

# Set required parameters
forecaster_spatial.gen_int_array = generation_interval  
forecaster_spatial.inf_hosp_array = inf_to_hosp_delay
forecaster_spatial.region_pop=region_pop

# Run inference
samples_spatial, predictions_spatial = forecaster_spatial.run_mcmc()
```

## Model Details

### Classical Bayesian Renewal Model

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

### Spatial Bayesian Renewal Model

Extends the classical model with a mobility network:

$$I_{t,i} = R_{t,i} \sum_{\tau < t} I_{t-\tau,i} g(\tau) + \eta_{spatial} \sum_{j} A_{ij} I_{t-1,j}$$

where:
- $A_{ij}$ = spatial adjacency matrix (connectivity weight from region $j$ to $i$)
- $\eta_{spatial}$ = spatial spillover rate parameter
- Separate $R_t$ evolves per region with region-specific wastewater effects

### Observation Models

**Hospital Admissions:**
$$h(t) \sim \text{NegativeBinomial2}(\mu=P_{hosp} * (I \otimes d), k)$$


where:
- $d$ = infection-to-hospitalization delay distribution
- $\otimes$ = convolution operator
- $P_{hosp}$= hierarchical parameters with hyperpriors

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
- With sufficient data
- Spatial effects are negligible

**Choose Spatial when:**
- Analyzing multiple regions
- Need to account for cross-region transmission
- Have insufficient hospital data
- Building a regional forecasting system

## Requirements

- Python >= 3.9
- numpy
- polars
- jax >= 0.3.0
- jaxlib >= 0.3.0
- numpyro >= 0.10.0
- matplotlib


## Citation

If you use this package, please cite:

```bibtex
@article{luzhong2024bayesian,
  title={A spatial wastewater-informed framework for early prediction of hospital demand for respiratory diseases
},
  author={Lu Zhong, Amanda Bleichrodt, Aakash Pandey, Deborah Kunkel, Lior Rennert
},
  year={2026},
  url={...},
}
```

## License

MIT License - see LICENSE file for details

## Support

For issues, questions, or feature requests, please open an issue on GitHub or contact the development team.
