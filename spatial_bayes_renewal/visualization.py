"""Visualization helpers for Bayesian renewal forecasting examples.

These functions package the notebook plotting logic into reusable Python
functions so the same figures can be generated from scripts or notebooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


DEFAULT_COLORS = ["darkred", "steelblue", "mediumseagreen"]


def _finalize_figure(fig, save_path: str | Path | None = None, show: bool = True):
    """Apply tight layout, optionally save, and optionally show the figure."""
    fig.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if not show:
        plt.close(fig)
    return fig


def plot_temporal_data(
    dates: Sequence,
    hosp_obs: Sequence[float],
    ww_obs: Sequence[float],
    split_idx: int,
    save_path: str | Path | None = None,
    show: bool = True,
):
    """Plot  hospital and wastewater observations."""
    dates = pd.to_datetime(dates)
    hosp_obs = np.asarray(hosp_obs)
    ww_obs = np.asarray(ww_obs)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(
        dates,
        hosp_obs,
        marker="o",
        linestyle="-",
        color="darkred",
        linewidth=2,
        markersize=4,
        label="Hospital Admissions",
    )
    axes[0].axvline(
        x=dates[split_idx],
        color="gray",
        linestyle="--",
        alpha=0.5,
        label="Training/Test Split",
    )
    axes[0].set_xlabel("Date", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Daily Hospital Admissions", fontsize=12, fontweight="bold")
    axes[0].set_title("Synthetic Data: Single Region (Classical Model)", fontsize=14, fontweight="bold")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        dates,
        ww_obs,
        marker="s",
        linestyle="-",
        color="steelblue",
        linewidth=2,
        markersize=4,
        label="Wastewater Concentration",
    )
    axes[1].axvline(
        x=dates[split_idx],
        color="gray",
        linestyle="--",
        alpha=0.5,
        label="Training/Test Split",
    )
    axes[1].set_xlabel("Date", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("WW Concentration (copies/mL)", fontsize=12, fontweight="bold")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.3)
    axes[1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)

    return _finalize_figure(fig, save_path=save_path, show=show)


def plot_forecast(
    dates,
    hosp_obs,
    h_center,
    h_lower,
    h_upper,
    n_train,
    n_test,
    wastewater=None,              # NEW
    wastewater_label="Wastewater Concentration",
    save_path=None,
    show=True,
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    dates = pd.to_datetime(dates)
    hosp_obs = np.asarray(hosp_obs)
    h_center = np.asarray(h_center)
    h_lower = np.asarray(h_lower)
    h_upper = np.asarray(h_upper)

    fig, ax = plt.subplots(figsize=(15, 7))

    # --- LEFT AXIS (hospitalizations) ---
    ax.scatter(
        dates[:n_train],
        hosp_obs[:n_train],
        s=80,
        color="darkred",
        alpha=0.7,
        label="Observed (Training)",
        zorder=5,
    )

    ax.scatter(
        dates[n_train : n_train + n_test],
        hosp_obs[n_train : n_train + n_test],
        s=80,
        color="orange",
        alpha=0.7,
        label="Observed (Test)",
        zorder=5,
    )

    ax.plot(dates, h_center, color="steelblue", linewidth=2.5, label="Posterior Median", zorder=4)

    ax.fill_between(dates, h_lower, h_upper, color="steelblue", alpha=0.25, label="95% CI", zorder=2)
    
    ax.set_ylabel("Hospital Admissions", fontsize=13, fontweight="bold")

    # --- RIGHT AXIS (wastewater) ---
    if wastewater is not None:
        wastewater = np.asarray(wastewater)

        ax2 = ax.twinx()

        ax2.plot(
            dates,
            wastewater,
            color="green",
            linewidth=2,
            linestyle="--",
            label=wastewater_label,
        )

        ax2.set_ylabel(wastewater_label, fontsize=13, fontweight="bold", color="green")
        ax2.tick_params(axis="y", labelcolor="green")

    # --- shared formatting ---
    ax.axvline(
        x=dates[n_train],
        color="gray",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label="Train/Test Split",
    )

    ax.axvspan(dates[n_train], dates[-1], alpha=0.08, color="yellow")

    ax.set_xlabel("Date", fontsize=13, fontweight="bold")

    ax.set_title(
        "Bayesian Forecast with Wastewater Signal",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    ax.grid(alpha=0.3, linestyle=":")

    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # --- combine legends ---
    lines, labels = ax.get_legend_handles_labels()
    if wastewater is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines += lines2
        labels += labels2

    ax.legend(lines, labels, loc="upper left", fontsize=10, framealpha=0.95)

    return _finalize_figure(fig, save_path=save_path, show=show)


def plot_reproduction_number(
    dates: Sequence,
    r_center: Sequence[float],
    r_lower: Sequence[float],
    r_upper: Sequence[float],
    n_train: int,
    save_path: str | Path | None = None,
    show: bool = True,
):
    """Plot time-varying reproduction number estimates with uncertainty bands."""
    dates = pd.to_datetime(dates)
    r_center = np.asarray(r_center)
    r_lower = np.asarray(r_lower)
    r_upper = np.asarray(r_upper)


    n_points = len(r_center)
    dates_used = dates[:n_points]

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(dates_used, r_center, color="darkred", linewidth=2.5, label="Posterior Median", zorder=4)
    #ax.fill_between(dates_used, r_lower, r_upper, color="darkred", alpha=0.25, label="95% Credible Interval", zorder=2)
 
    if len(dates_used) > 0:
        split_idx = min(max(n_train, 0), len(dates_used) - 1)
        ax.axvline(
            x=dates_used[split_idx],
            color="gray",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="Training/Forecast Split",
        )
        #ax.axvspan(dates_used[split_idx], dates_used[-1], alpha=0.08, color="yellow")

    ax.set_xlabel("Date", fontsize=13, fontweight="bold")
    ax.set_ylabel("Reproduction Number R(t)", fontsize=13, fontweight="bold")
    ax.set_title(
        "Classical Bayesian Renewal Model: Time-Varying Reproduction Number",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax.grid(alpha=0.3, linestyle=":")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    return _finalize_figure(fig, save_path=save_path, show=show)

'''
def plot_reproduction_number(
    forecast_dates: Sequence,
    r_median: Sequence[float],
    r_lower: Sequence[float],
    r_upper: Sequence[float],
    n_train: int,
    r_q25: Sequence[float] | None = None,
    r_q75: Sequence[float] | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
):
    """Plot the time-varying reproduction number with uncertainty bands."""
    forecast_dates = pd.to_datetime(forecast_dates)
    r_median = np.asarray(r_median)
    r_lower = np.asarray(r_lower)
    r_upper = np.asarray(r_upper)

    if r_q25 is None:
        r_q25 = r_lower
    if r_q75 is None:
        r_q75 = r_upper

    n_points = len(r_median)
    dates_used = forecast_dates[:n_points]

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(dates_used, r_median, color="darkred", linewidth=2.5, label="Posterior Median", zorder=4)
    ax.fill_between(dates_used, r_lower, r_upper, color="darkred", alpha=0.25, label="95% Credible Interval", zorder=2)
    ax.fill_between(dates_used, r_q25, r_q75, color="darkred", alpha=0.4, label="50% Credible Interval", zorder=3)
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=2, label="R=1 (Epidemic Threshold)", zorder=3)
    split_idx = min(n_train, len(dates_used) - 1)
    ax.axvline(x=dates_used[split_idx], color="gray", linestyle="--", linewidth=2, alpha=0.7, label="Training/Forecast Split")
    ax.axvspan(dates_used[split_idx], dates_used[-1], alpha=0.08, color="yellow")
    ax.set_xlabel("Date", fontsize=13, fontweight="bold")
    ax.set_ylabel("Reproduction Number R(t)", fontsize=13, fontweight="bold")
    ax.set_title("Classical Bayesian Renewal Model: Time-Varying Reproduction Number", fontsize=14, fontweight="bold", pad=20)
    ax.set_ylim([0.5, 2.5])
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax.grid(alpha=0.3, linestyle=":")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    return _finalize_figure(fig, save_path=save_path, show=show)


def plot_synthetic_spatial_data(
    df_spatial,
    regions: Sequence[str],
    split_idx: int = 39,
    save_path: str | Path | None = None,
    show: bool = True,
):
    """Plot synthetic multi-region hospital and wastewater observations."""
    fig, axes = plt.subplots(len(regions), 1, figsize=(15, 10))
    if len(regions) == 1:
        axes = [axes]

    for idx, region in enumerate(regions):
        region_data = _filter_region(df_spatial, region)
        dates_r = pd.to_datetime(region_data["date"].to_numpy())
        hosp_r = region_data["hosp_obs"].to_numpy()
        ww_r = region_data["ww_obs"].to_numpy()

        ax = axes[idx]
        ax2 = ax.twinx()
        line1 = ax.plot(dates_r, hosp_r, marker="o", linestyle="-", color="darkred", linewidth=2, markersize=4, label="Hospital Admissions")
        line2 = ax2.plot(dates_r, ww_r, marker="s", linestyle="-", color="steelblue", linewidth=2, markersize=4, label="Wastewater")
        ax.axvline(x=dates_r[split_idx], color="gray", linestyle="--", alpha=0.5)
        ax.fill_between(dates_r[:split_idx + 1], 0, 300, alpha=0.1, color="blue")
        ax.fill_between(dates_r[split_idx:], 0, 300, alpha=0.1, color="orange")
        ax.set_ylabel("Daily Hospital Admissions", fontsize=11, fontweight="bold")
        ax2.set_ylabel("WW Concentration (copies/mL)", fontsize=11, fontweight="bold")
        ax.set_title(f"Region: {region}", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        if idx == 0:
            lines = line1 + line2
            labels = [line.get_label() for line in lines]
            ax.legend(lines, labels, loc="upper left", fontsize=10)

    axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha="right")
    fig.suptitle("Synthetic Multi-Region Data (Spatial Model)", fontsize=14, fontweight="bold", y=0.995)

    return _finalize_figure(fig, save_path=save_path, show=show)


# Avoid a hard runtime dependency on polars import unless the function is called.
def _filter_region(df_spatial, region: str):
    if hasattr(df_spatial, "filter"):
        try:
            import polars as pl  # local import

            return df_spatial.filter(pl.col("county") == region)
        except Exception:
            pass
    if isinstance(df_spatial, pd.DataFrame):
        return df_spatial.loc[df_spatial["county"] == region]
    raise TypeError("df_spatial must be a Polars or pandas DataFrame")


def plot_spatial_network(
    graph: nx.DiGraph,
    save_path: str | Path | None = None,
    show: bool = True,
    seed: int = 42,
):
    """Visualize the directed spatial spillover network."""
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.title("Regional Spillover Network", fontsize=14, fontweight="bold", pad=20)

    pos = nx.spring_layout(graph, k=2, iterations=50, seed=seed)
    nx.draw_networkx_nodes(graph, pos, node_color="lightblue", node_size=3000, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=11, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        edge_color="gray",
        arrows=True,
        arrowsize=20,
        arrowstyle="->",
        width=2,
        connectionstyle="arc3,rad=0.1",
    )

    edge_labels = nx.get_edge_attributes(graph, "weight")
    for (source, target), weight in edge_labels.items():
        x = (pos[source][0] + pos[target][0]) / 2
        y = (pos[source][1] + pos[target][1]) / 2
        ax.text(
            x,
            y,
            f"{weight:.1f}",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    ax.axis("off")
    return _finalize_figure(fig, save_path=save_path, show=show)


def plot_spatial_forecasts(
    forecast_dates: Sequence,
    df_spatial,
    h_pred_spatial: np.ndarray,
    regions: Sequence[str],
    n_train: int,
    n_prev: int,
    test_horizon: int = 20,
    colors: Sequence[str] | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
):
    """Plot spatial hospital forecasts for all regions in separate panels."""
    forecast_dates = pd.to_datetime(forecast_dates)
    colors = list(colors or DEFAULT_COLORS)
    if len(colors) < len(regions):
        repeats = (len(regions) // len(colors)) + 1
        colors = (colors * repeats)[: len(regions)]

    fig, axes = plt.subplots(len(regions), 1, figsize=(15, 11))
    if len(regions) == 1:
        axes = [axes]

    for region_idx, region in enumerate(regions):
        region_obs_data = _filter_region(df_spatial, region)
        hosp_obs_r = region_obs_data["hosp_obs"].to_numpy()
        h_pred_r = h_pred_spatial[:, region_idx, :]

        h_median_r = np.median(h_pred_r, axis=0)[n_prev : n_prev + n_train + 28]
        h_lower_r = np.percentile(h_pred_r, 2.5, axis=0)[n_prev : n_prev + n_train + 28]
        h_upper_r = np.percentile(h_pred_r, 97.5, axis=0)[n_prev : n_prev + n_train + 28]
        h_q25_r = np.percentile(h_pred_r, 25, axis=0)[n_prev : n_prev + n_train + 28]
        h_q75_r = np.percentile(h_pred_r, 75, axis=0)[n_prev : n_prev + n_train + 28]

        ax = axes[region_idx]
        color = colors[region_idx]
        ax.scatter(forecast_dates[:n_train], hosp_obs_r[:n_train], s=70, color=color, alpha=0.7, label="Observed (Training)", zorder=5)
        ax.scatter(forecast_dates[n_train:], hosp_obs_r[n_train : n_train + test_horizon], s=70, color="orange", alpha=0.7, label="Observed (Test)", zorder=5)
        ax.plot(forecast_dates, h_median_r, color=color, linewidth=2.5, label="Posterior Median", zorder=4)
        ax.fill_between(forecast_dates, h_lower_r, h_upper_r, color=color, alpha=0.25, label="95% Credible Interval", zorder=2)
        ax.fill_between(forecast_dates, h_q25_r, h_q75_r, color=color, alpha=0.4, label="50% Credible Interval", zorder=3)
        ax.axvline(x=forecast_dates[n_train], color="gray", linestyle="--", linewidth=2, alpha=0.7)
        ax.axvspan(forecast_dates[n_train], forecast_dates[-1], alpha=0.08, color="yellow")
        ax.set_ylabel("Daily Hospital Admissions", fontsize=11, fontweight="bold")
        ax.set_title(f"Region: {region}", fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.3, linestyle=":")
        if region_idx == len(regions) - 1:
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            ax.set_xlabel("Date", fontsize=11, fontweight="bold")
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    fig.suptitle("Spatial Bayesian Renewal Model: 4-Week Hospital Admissions Forecast (All Regions)", fontsize=14, fontweight="bold", y=0.995)
    return _finalize_figure(fig, save_path=save_path, show=show)


def plot_model_comparison(
    forecast_dates: Sequence,
    hosp_obs: Sequence[float],
    h_median: Sequence[float],
    h_lower: Sequence[float],
    h_upper: Sequence[float],
    h_spatial_median_agg: Sequence[float],
    h_spatial_lower_agg: Sequence[float],
    h_spatial_upper_agg: Sequence[float],
    n_train: int,
    test_horizon: int = 20,
    save_path: str | Path | None = None,
    show: bool = True,
):
    """Compare classical and aggregated spatial hospital forecasts."""
    forecast_dates = pd.to_datetime(forecast_dates)
    hosp_obs = np.asarray(hosp_obs)

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.scatter(
        forecast_dates[:n_train],
        hosp_obs[:n_train],
        s=100,
        color="black",
        alpha=0.7,
        label="Observed (Training)",
        edgecolors="white",
        linewidth=1.5,
        zorder=6,
    )
    ax.scatter(
        forecast_dates[n_train:],
        hosp_obs[n_train : n_train + test_horizon],
        s=100,
        color="orange",
        alpha=0.7,
        label="Observed (Test)",
        edgecolors="white",
        linewidth=1.5,
        zorder=6,
    )
    ax.plot(forecast_dates, h_median, color="steelblue", linewidth=2.5, label="Classical Model - Posterior Median", zorder=4, linestyle="-")
    ax.fill_between(forecast_dates, h_lower, h_upper, color="steelblue", alpha=0.15, label="Classical - 95% CI", zorder=2)
    ax.plot(forecast_dates, h_spatial_median_agg, color="darkred", linewidth=2.5, label="Spatial Model (Aggregated) - Posterior Median", zorder=5, linestyle="--")
    ax.fill_between(forecast_dates, h_spatial_lower_agg, h_spatial_upper_agg, color="darkred", alpha=0.15, label="Spatial - 95% CI", zorder=3)
    ax.axvline(x=forecast_dates[n_train], color="gray", linestyle=":", linewidth=2.5, alpha=0.7, label="Training/Forecast Split", zorder=2)
    ax.axvspan(forecast_dates[n_train], forecast_dates[-1], alpha=0.06, color="yellow", zorder=1)
    ax.text(
        forecast_dates[min(n_train + 14, len(forecast_dates) - 1)],
        ax.get_ylim()[1] * 0.92,
        "4-Week Forecast Period",
        fontsize=12,
        fontweight="bold",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.7),
    )
    ax.set_xlabel("Date", fontsize=13, fontweight="bold")
    ax.set_ylabel("Daily Hospital Admissions", fontsize=13, fontweight="bold")
    ax.set_title("Model Comparison: Classical vs Spatial Bayesian Renewal (4-Week Forecast)", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.96)
    ax.grid(alpha=0.3, linestyle=":")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    return _finalize_figure(fig, save_path=save_path, show=show)
'''

__all__ = [
    "plot_temporal_data",
    "plot_forecast",
    "plot_reproduction_number",
]