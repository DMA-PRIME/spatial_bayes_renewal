# Quick Installation & Reference Guide

## Package Location
```
/Users/luzhong/Library/CloudStorage/Box-Box/BoxPHI-PHMR Projects/LuZhong/Projects/spatial_bayes_renewal
```

## Installation

### Option 1: Development Mode (Recommended)
```bash
cd "/Users/luzhong/Library/CloudStorage/Box-Box/BoxPHI-PHMR Projects/LuZhong/Projects/spatial_bayes_renewal"
pip install -e .
```

### Option 2: Standard Install
```bash
cd "/Users/luzhong/Library/CloudStorage/Box-Box/BoxPHI-PHMR Projects/LuZhong/Projects/spatial_bayes_renewal"
pip install .
```

## After Installation - Usage

### Import and Use Classical Model
```python
from spatial_bayes_renewal import ClassicalForecaster

forecaster = ClassicalForecaster(
    df_data_train=your_data,
    cols_concern=['hosp_obs', 'ww_obs'],
    n_forecast_points=14,
    pop=1_000_000,
    num_samples=200,
    num_warmup=100
)

# Set distributions before running
forecaster.gen_int_array = gen_interval
forecaster.inf_hosp_array = hosp_delay
forecaster.sheddin = shedding_dist

samples, predictions = forecaster.run_mcmc()
```

### Import and Use Spatial Model
```python
from spatial_bayes_renewal import SpatialForecaster
import networkx as nx

# Create network
G = nx.DiGraph()
G.add_edge('region1', 'region2', weight=0.5)

forecaster = SpatialForecaster(
    df_data=your_data,
    spatial_net=G,
    region_list=['region1', 'region2'],
    region_obs_list=['region1', 'region2'],
    cols_concern=['hosp_obs', 'ww_obs'],
    n_forecast_points=14
)

# Set population and distributions
forecaster.region_pop = np.array([pop1, pop2])
forecaster.gen_int_array = gen_interval
forecaster.inf_hosp_array = hosp_delay
forecaster.sheddin = shedding_dist

samples, predictions = forecaster.run_mcmc()
```

## Package Contents

### Core Modules
- **ClassicalForecaster** - Non-spatial, single-region model
  - File: `spatial_bayes_renewal/classical_forecaster.py`
  - Infection cases: Basic, Feedback, Logistic_S
  - Best for: Single city/region analysis

- **SpatialForecaster** - Multi-region model with spatial effects
  - File: `spatial_bayes_renewal/spatial_forecaster.py`
  - Models spillover via adjacency matrices
  - Best for: Regional networks, state-level studies

### Documentation Files
- `README.md` - Comprehensive documentation
- `PACKAGE_STRUCTURE.txt` - Package structure overview
- `USAGE_EXAMPLES.py` - Detailed usage examples
- `setup.py` - Installation configuration
- `requirements.txt` - Dependencies
- `LICENSE` - MIT License

## Quick Reference: Model Selection

| Scenario | Use |
|----------|-----|
| Single city/region | ClassicalForecaster |
| Multiple regions (independent) | ClassicalForecaster |
| Multiple connected regions | SpatialForecaster |
| Need spatial spillover effects | SpatialForecaster |
| Want fast inference | ClassicalForecaster |
| Regional network analysis | SpatialForecaster |

## Key Methods

### Both Models
```python
forecaster.run_mcmc(init_params=None)  # Run MCMC inference
forecaster.update_training_data(df)     # Update with new data
forecaster.plot_results(...)            # Plot forecasts
```

### Classical Only
```python
forecaster.plot_posterior_results(...)  # Plot posterior distributions
forecaster.sum_every_n_elements(...)    # Aggregate by time period
```

## Output Structure

### samples (Dictionary)
- `R0` - Basic reproduction number
- `R_t` - Time-varying reproduction (shape: samples × time)
- `I_t` - Infections (shape: samples × time [× regions])
- `P_hosp` - Hospitalization proportion
- `G` - Wastewater scaling factor
- `k_hosp`, `k_ww` - Dispersion parameters

### predictions (Dictionary)
- `I_t` - Predicted infections
- `H_t` - Predicted hospitalizations
- `W_t` - Predicted wastewater concentrations
- `R_t` - Predicted reproduction numbers

## Required Inputs

### Always Set (Both Models)
```python
forecaster.gen_int_array      # Generation interval PDF (length ~14-30)
forecaster.inf_hosp_array     # Infection-to-hosp delay PDF (length ~7-14)
forecaster.sheddin            # Viral shedding kinetics PDF (length ~20-30)
```

### Spatial Model Only
```python
forecaster.region_pop         # Array of regional populations
```

## Example Workflow

```python
import numpy as np
import polars as pl
from spatial_bayes_renewal import SpatialForecaster
import networkx as nx
from scipy.stats import lognorm

# 1. Load data
df = pl.read_csv("your_data.csv")

# 2. Create network
G = nx.DiGraph()
G.add_edge('region1', 'region2', weight=0.3)
G.add_edge('region2', 'region3', weight=0.2)

# 3. Create distributions
gen_interval = lognorm.pdf(np.arange(30), s=0.83, scale=np.exp(1.63))
gen_interval /= gen_interval.sum()

hosp_delay = lognorm.pdf(np.arange(14), s=0.5, scale=np.exp(2.0))
hosp_delay /= hosp_delay.sum()

shedding = lognorm.pdf(np.arange(30), s=0.5, scale=np.exp(1.0))
shedding /= shedding.sum()

# 4. Initialize forecaster
fc = SpatialForecaster(
    df_data=df,
    spatial_net=G,
    region_list=['region1', 'region2', 'region3'],
    region_obs_list=['region1', 'region2', 'region3'],
    cols_concern=['hosp_obs', 'ww_obs'],
    n_forecast_points=14,
    num_samples=200,
    num_warmup=100
)

# 5. Set parameters
fc.region_pop = np.array([1_000_000, 800_000, 600_000])
fc.gen_int_array = gen_interval
fc.inf_hosp_array = hosp_delay
fc.sheddin = shedding

# 6. Run inference
samples, predictions = fc.run_mcmc()

# 7. Analyze
print(f"Sample shape: {samples['R_t'].shape}")
print(f"Predictions shape: {predictions['H_t'].shape}")
```

## Dependencies

- Python ≥ 3.9
- numpy ≥ 1.21.0
- polars ≥ 0.18.0
- jax ≥ 0.3.0
- jaxlib ≥ 0.3.0
- numpyro ≥ 0.10.0
- matplotlib ≥ 3.5.0

Install with:
```bash
pip install -e .
```

## Troubleshooting

### ModuleNotFoundError: No module named 'spatial_bayes_renewal'
→ Make sure you've installed the package: `pip install -e .`

### Missing 'gen_int_array' error
→ Set required distributions before calling `run_mcmc()`:
```python
forecaster.gen_int_array = your_generation_interval
```

### Shape mismatch errors
→ Verify distribution arrays have correct shape and sum to 1:
```python
assert gen_int_array.sum() ≈ 1.0
assert inf_hosp_array.sum() ≈ 1.0
assert sheddin.sum() ≈ 1.0
```

## More Information

- See `README.md` for detailed documentation
- See `USAGE_EXAMPLES.py` for code examples
- See `PACKAGE_STRUCTURE.txt` for file organization
