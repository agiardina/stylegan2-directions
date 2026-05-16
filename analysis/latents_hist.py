#!/usr/bin/env python
# coding: utf-8
"""
Plot latent and direction value distributions for the paper analysis.

Run from project root:
python3 analysis/latents_hist.py

Required inputs:
- out/latentspaces.csv
- directions/{blueeyes,eyebrown,nose,lips,chin}.npy
- generators-with-stylegan2/latent_directions/{age,gender}.npy

External dependency for age/gender directions:
git clone https://github.com/a312863063/generators-with-stylegan2.git

Outputs:
- out/analysis/latents_hist_panels_abcd.png
- out/analysis/latents_hist_panels_abcd.tif
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator


OUT_DIR = Path("out/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _clipped_fd_bins(values: np.ndarray, min_bins: int, max_bins: int) -> int:
    """Freedman-Diaconis bins clipped to a publication-friendly range."""
    n_bins = len(np.histogram_bin_edges(values, bins="fd")) - 1
    return int(np.clip(n_bins, min_bins, max_bins))


def _remove_outliers_iqr(values: np.ndarray, factor: float = 20.0):
    flat_values = np.asarray(values).ravel()
    q1 = np.percentile(flat_values, 25)
    q3 = np.percentile(flat_values, 75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    mask = (flat_values >= lower_bound) & (flat_values <= upper_bound)
    bounds = {
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower": float(lower_bound),
        "upper": float(upper_bound),
    }
    return flat_values[mask], mask, bounds


def _shape_statistics(values: np.ndarray):
    series = pd.Series(np.asarray(values).ravel())
    skewness = float(series.skew())
    kurtosis_fisher = float(series.kurt())
    return {
        "skewness": skewness,
        "kurtosis_fisher": kurtosis_fisher,
    }


def _summary_statistics(values: np.ndarray):
    flat_values = np.asarray(values).ravel()
    return {
        "mean": float(np.mean(flat_values)),
        "std": float(np.std(flat_values)),
        "min": float(np.min(flat_values)),
        "max": float(np.max(flat_values)),
    }


latent_df = pd.read_csv("out/latentspaces.csv")
latent_vectors = latent_df.drop(columns=["id"], errors="ignore").to_numpy()
data_flat = latent_vectors.ravel()

quantile_5_w = np.percentile(data_flat, 5)
quantile_95_w = np.percentile(data_flat, 95)
print(f"Quantile for photos 5-95%: {quantile_5_w}, {quantile_95_w}")

clean_data, latent_mask, latent_bounds = _remove_outliers_iqr(data_flat, factor=20.0)
num_outliers = int(np.sum(~latent_mask))

latent_values_for_std = latent_vectors.astype(float, copy=True)
latent_values_for_std[~latent_mask.reshape(latent_vectors.shape)] = np.nan
latent_std_values = np.nanstd(latent_values_for_std, axis=0)
latent_std_values = latent_std_values[np.isfinite(latent_std_values)]
latent_std_summary = _summary_statistics(latent_std_values)

print(f"Q1: {latent_bounds['q1']:.2f}, Q3: {latent_bounds['q3']:.2f}")
print(f"IQR: {latent_bounds['iqr']:.2f}")
print(f"Lower outlier bound: {latent_bounds['lower']:.2f}")
print(f"Upper outlier bound: {latent_bounds['upper']:.2f}")
print(f"Number of outliers removed: {num_outliers}")

print(f"Original data ({len(data_flat)}) -> Clean data ({len(clean_data)})")

latent_shape = _shape_statistics(clean_data)
print(f"Skewness (clean latent values): {latent_shape['skewness']:.6f}")
print(f"Kurtosis Fisher (clean latent values): {latent_shape['kurtosis_fisher']:.6f}")
print(
    "Latent variable std (across subjects): "
    f"mean={latent_std_summary['mean']:.6f}, std={latent_std_summary['std']:.6f}, "
    f"min={latent_std_summary['min']:.6f}, max={latent_std_summary['max']:.6f}"
)

# Optional clean dataframe for quick inspection
df_clean = pd.DataFrame(clean_data, columns=['valore_pulito'])
print("Clean dataframe (first rows):")
print(df_clean.head())

v = {}
v["age"] = np.load("generators-with-stylegan2/latent_directions/age.npy")
v["age"] = v["age"] / np.linalg.norm(v["age"])
v["age"] = np.expand_dims(v["age"], axis=0)
v["gender"] = np.load("generators-with-stylegan2/latent_directions/gender.npy")
v["gender"] = v["gender"] / np.linalg.norm(v["gender"])
v["gender"] = np.expand_dims(v["gender"], axis=0)
v["blueeyes"] = np.load("directions/blueeyes.npy")
v["eyebrow"] = np.load("directions/eyebrown.npy")
v["nose"] = np.load("directions/nose.npy")
v["lips"] = np.load("directions/lips.npy")
v["chin"] = np.load("directions/chin.npy")

direction_values = np.concatenate([np.asarray(values).ravel() for values in v.values()])
quantile_5_d = np.percentile(direction_values, 5)
quantile_95_d = np.percentile(direction_values, 95)
print(f"Quantile for directions 5-95%: {quantile_5_d}, {quantile_95_d}")

clean_direction_values, direction_mask, direction_bounds = _remove_outliers_iqr(direction_values, factor=20.0)
direction_outliers = int(np.sum(~direction_mask))

print("Direction values outlier filtering:")
print(f"Q1: {direction_bounds['q1']:.6f}, Q3: {direction_bounds['q3']:.6f}")
print(f"IQR: {direction_bounds['iqr']:.6f}")
print(f"Lower outlier bound: {direction_bounds['lower']:.6f}")
print(f"Upper outlier bound: {direction_bounds['upper']:.6f}")
print(f"Number of outliers removed: {direction_outliers}")
print(
    f"Original direction data ({len(direction_values)}) -> "
    f"Clean direction data ({len(clean_direction_values)})"
)

direction_shape = _shape_statistics(clean_direction_values)
print(f"Skewness (clean direction values): {direction_shape['skewness']:.6f}")
print(f"Kurtosis Fisher (clean direction values): {direction_shape['kurtosis_fisher']:.6f}")

direction_tensor = np.concatenate(
    [np.asarray(values).reshape(1, 18, 512) for values in v.values()],
    axis=0,
)
direction_tensor = direction_tensor.astype(float, copy=False)
direction_tensor[~direction_mask.reshape(direction_tensor.shape)] = np.nan
direction_std_values = np.nanstd(direction_tensor, axis=0).ravel()
direction_std_values = direction_std_values[np.isfinite(direction_std_values)]
direction_std_summary = _summary_statistics(direction_std_values)
print(
    "Latent variable std (across directions): "
    f"mean={direction_std_summary['mean']:.6f}, std={direction_std_summary['std']:.6f}, "
    f"min={direction_std_summary['min']:.6f}, max={direction_std_summary['max']:.6f}"
)


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)

fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

colors = {
    "latent": "#0072B2",      # Okabe-Ito blue
    "direction": "#E69F00",   # Okabe-Ito orange
}

bins_latent = _clipped_fd_bins(clean_data, min_bins=50, max_bins=140)
bins_direction = _clipped_fd_bins(clean_direction_values, min_bins=30, max_bins=100)
bins_latent_std = _clipped_fd_bins(latent_std_values, min_bins=40, max_bins=140)
bins_direction_std = _clipped_fd_bins(direction_std_values, min_bins=30, max_bins=100)

axes[0, 0].hist(
    clean_data,
    bins=bins_latent,
    range=(-50, 50),
    color=colors["latent"],
    edgecolor="white",
    linewidth=0.5,
    alpha=0.9,
)
axes[0, 0].set_xlim(-50, 50)
axes[0, 0].set_title("Faces' latent vectors (59 subjects)")
axes[0, 0].set_xlabel("Value")
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].xaxis.set_major_locator(MultipleLocator(10))
axes[0, 0].xaxis.set_minor_locator(MultipleLocator(5))
axes[0, 0].yaxis.set_major_locator(MaxNLocator(nbins=9))
axes[0, 0].tick_params(axis="x", which="minor", length=3)

axes[0, 1].hist(
    clean_direction_values,
    bins=bins_direction,
    color=colors["direction"],
    edgecolor="white",
    linewidth=0.5,
    alpha=0.9,
)
axes[0, 1].set_title("Direction vectors (7 directions)")
axes[0, 1].set_xlabel("Value")
axes[0, 1].set_ylabel("Frequency")
axes[0, 1].xaxis.set_major_locator(MaxNLocator(nbins=9))
axes[0, 1].yaxis.set_major_locator(MaxNLocator(nbins=9))

axes[1, 0].hist(
    latent_std_values,
    bins=bins_latent_std,
    color=colors["latent"],
    edgecolor="white",
    linewidth=0.5,
    alpha=0.9,
)
axes[1, 0].set_title("Std. dev. per latent variable (59 subjects)")
axes[1, 0].set_xlabel("Standard deviation")
axes[1, 0].set_ylabel("Frequency")
axes[1, 0].xaxis.set_major_locator(MaxNLocator(nbins=8))
axes[1, 0].yaxis.set_major_locator(MaxNLocator(nbins=7))

axes[1, 1].hist(
    direction_std_values,
    bins=bins_direction_std,
    color=colors["direction"],
    edgecolor="white",
    linewidth=0.5,
    alpha=0.9,
)
axes[1, 1].set_title("Std. dev. per latent variable (7 directions)")
axes[1, 1].set_xlabel("Standard deviation")
axes[1, 1].set_ylabel("Frequency")
axes[1, 1].xaxis.set_major_locator(MaxNLocator(nbins=7))
axes[1, 1].yaxis.set_major_locator(MaxNLocator(nbins=7))
axes[1, 1].ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))

stats_box = {
    "facecolor": "white",
    "edgecolor": "#B0B0B0",
    "boxstyle": "round,pad=0.25",
    "alpha": 0.9,
}

axes[0, 0].text(
    0.98,
    0.95,
    (
        f"Skewness: {latent_shape['skewness']:.3f}\n"
        f"Kurtosis (Fisher): {latent_shape['kurtosis_fisher']:.3f}"
    ),
    transform=axes[0, 0].transAxes,
    ha="right",
    va="top",
    fontsize=9,
    bbox=stats_box,
)

axes[0, 1].text(
    0.98,
    0.95,
    (
        f"Skewness: {direction_shape['skewness']:.3f}\n"
        f"Kurtosis (Fisher): {direction_shape['kurtosis_fisher']:.3f}"
    ),
    transform=axes[0, 1].transAxes,
    ha="right",
    va="top",
    fontsize=9,
    bbox=stats_box,
)

for axis, label in zip(axes.ravel(), ("A", "B", "C", "D")):
    axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.text(
        -0.14,
        1.06,
        label,
        transform=axis.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="left",
    )

png_path = OUT_DIR / "latents_hist_panels_abcd.png"
tiff_path = OUT_DIR / "latents_hist_panels_abcd.tif"
fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(tiff_path, dpi=300, bbox_inches="tight")

plt.show()

print(f"Saved panel figure: {png_path}")
print(f"Saved panel figure: {tiff_path}")
