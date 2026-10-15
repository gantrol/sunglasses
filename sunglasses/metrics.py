"""Image diagnostics and offline matched baselines; no flash-safety classifier."""

import numpy as np


def matched_gains(source_y: np.ndarray, filtered_y: np.ndarray, threshold: float) -> dict[str, float]:
    """Choose constant gains using the entire clip: deliberately non-causal controls."""
    peak = float(source_y.max())
    total = float(source_y.sum(dtype=np.float64))
    gains = {
        "dim_peak": float(filtered_y.max()) / peak if peak else 1.0,
        "dim_mean": float(filtered_y.sum(dtype=np.float64)) / total if total else 1.0,
    }
    # Exact discrete distribution, not histogram bins or repeated full-clip scans.
    values, counts = np.unique(source_y, return_counts=True)
    values = values.astype(np.float64)
    tail_mass = np.r_[np.cumsum((values * counts)[::-1])[::-1], 0.0]
    tail_count = np.r_[np.cumsum(counts[::-1])[::-1], 0]
    target = float(np.maximum(filtered_y.astype(np.float64) - threshold, 0).mean())
    low, high = 0.0, 1.0
    for _ in range(48):
        gain = (low + high) / 2
        start = np.searchsorted(values, threshold / gain, side="right")
        exposure = (gain * tail_mass[start] - threshold * tail_count[start]) / source_y.size
        if exposure <= target:
            low = gain
        else:
            high = gain
    gains["dim_exposure"] = low
    return gains


def measure(y: np.ndarray, source_y: np.ndarray, knee: float, threshold: float, dt: float) -> dict:
    dark = (source_y > 0) & (source_y <= knee)
    dark_sum = float(source_y[dark].sum(dtype=np.float64))
    delta = np.abs(np.diff(y, axis=0))
    frame_mean = y.mean(axis=(1, 2), dtype=np.float64)
    return {
        "peak_y": float(y.max()),
        "mean_y": float(y.mean(dtype=np.float64)),
        "p95_y": float(np.quantile(y, .95)),
        "bright_area_fraction": float((y > threshold).mean()),
        "excess_y_area_seconds": float(np.maximum(y - threshold, 0).mean(axis=(1, 2), dtype=np.float64).sum() * dt),
        "dark_luminance_retention": float(y[dark].sum(dtype=np.float64)) / dark_sum if dark_sum else None,
        "mean_frame_delta": float(np.abs(np.diff(frame_mean)).mean()) if delta.size else 0.0,
        "mean_pixel_delta": float(delta.mean(dtype=np.float64)) if delta.size else 0.0,
        "max_pixel_delta": float(delta.max()) if delta.size else 0.0,
    }
