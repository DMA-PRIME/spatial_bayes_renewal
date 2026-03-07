"""
Bayesian Renewal Models for Infectious Disease Forecasting

A comprehensive probabilistic framework for forecasting infectious disease dynamics
using Bayesian renewal equations with wastewater surveillance data.

Two model variants:
- ClassicalForecaster: Non-spatial model for single/aggregated regions
- SpatialForecaster: Spatial model for multiple connected regions with spillover effects
"""

from .classical_forecaster import ClassicalForecaster
from .spatial_forecaster import SpatialForecaster

__version__ = "0.1.0"
__author__ = "LuZhong"

__all__ = [
    "ClassicalForecaster",
    "SpatialForecaster",
]
