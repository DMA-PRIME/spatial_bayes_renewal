import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error

try:
    from dtaidistance import dtw as _dtw
    _HAS_DTW = True
except ImportError:
    _HAS_DTW = False


def measure_accuracy(true_v, forecast):
    """Compute and print three accuracy metrics for the forecast period.

    Parameters
    ----------
    true_v   : array-like  Observed (ground-truth) values.
    forecast : array-like  Predicted (forecast) values.

    Returns
    -------
    pa       : float  Mean proportional accuracy  min/max per pair.
    rmse     : float  Root mean squared error.
    dtw_dist : float  DTW distance (NaN if dtaidistance not installed).
    """
    true_v   = np.asarray(true_v,   dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    # Proportional Accuracy: min(true, pred) / max(true, pred)
    pa_list = [
        min(t, f) / (max(t, f) + 1e-9)
        for t, f in zip(true_v, forecast)
    ]
    pa = float(np.mean(pa_list))

    # RMSE
    rmse = float(np.sqrt(mean_squared_error(true_v, forecast)))

    # DTW distance
    if _HAS_DTW:
        dtw_dist = float(_dtw.distance(true_v.tolist(), forecast.tolist()))
    else:
        dtw_dist = float('nan')
        print("  [warning] dtaidistance not installed — DTW set to NaN. "
              "Install with: pip install dtaidistance")


    return pa, rmse, dtw_dist