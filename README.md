# Bayesian Renewal Models for Infectious Disease Forecasting

A probabilistic framework for forecasting infectious disease dynamics using Bayesian renewal equations with wastewater surveillance data.

This package provides two complementary model variants:
- **ClassicalForecaster**: Non-spatial model for single or aggregated regions
- **SpatialForecaster**: Spatial model for multiple connected regions with spillover effects

## Features
### Spatial Model
- **Multi-region Dynamics**: Models disease transmission across connected spatial regions
- **Spatial Spillover**: Captures disease movement between adjacent regions via adjacency matrices (e.g., NetworkX graphs)
- **Wastewater Covariate**: Incorporates WW signals per region to modulate transmission
- **Vectorized Operations**: Efficient JAX-based computation across regions simultaneously

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
forecaster.gen_int_array = generation_interval  
forecaster.inf_hosp_array = inf_to_hosp_delay 

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
@software{luzhong2024bayesian,
  title={Bayesian Renewal Models for Infectious Disease Forecasting},
  author={LuZhong},
  year={2024},
  url={https://github.com/...}
}
```

## References

- CDC Pyrenew
--Semi Bayesian model

## License

MIT License - see LICENSE file for details

## Support

For issues, questions, or feature requests, please open an issue on GitHub or contact the development team.
