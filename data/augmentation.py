import numpy as np
from tsaug import Crop, Reverse, TimeWarp


def augment_workout(hr_series, speed_series, seed=None):
    """
    Augment a single heart rate and speed workout time series.

    Returns one physiologically feasible variant using the given seed.

    Parameters:
    - hr_series: 1D np.ndarray of length T (heart rate in bpm)
    - speed_series: 1D np.ndarray of length T (speed in original units)
    - crop_size: int or None, subsequence length L <= T; if None, full length T
    - seed: int or None, random seed for reproducibility

    Returns:
    - hr_aug: 1D np.ndarray of length L
    - speed_aug: 1D np.ndarray of length L
    """
    # Validate inputs
    hr = np.asarray(hr_series)
    sp = np.asarray(speed_series)
    if hr.ndim != 1 or sp.ndim != 1:
        raise ValueError("hr_series and speed_series must be 1D arrays of length T")
    if hr.shape[0] != sp.shape[0]:
        raise ValueError("hr_series and speed_series must have the same length")
    T = hr.shape[0]

    # Prepare data for tsaug: shape (1, T, 2)
    X = np.stack((hr, sp), axis=1)[None, ...]

    # Build pipeline for a single variant
    pipe = (
        TimeWarp(n_speed_change=3, max_speed_ratio=(1.5, 3), seed=seed)
        + Crop(size=round(0.8 * T), seed=seed) @ 0.5
        # + Drift(max_drift=(0.1, 0.5), n_drift_points=[1,2,3], normalize=True, per_channel=True, seed=seed)
        # + Quantize(n_levels=(90, 100), how="uniform", per_channel=True, seed=seed) @ 0.2
        + Reverse(seed=seed) @ 0.05
    )

    # Generate one augmented sample
    X_aug = pipe.augment(X)  # shape (1, L, 2)
    hr_aug = X_aug[0, :, 0]
    speed_aug = X_aug[0, :, 1]
    return hr_aug, speed_aug
