"""
Legacy module for C-style processing functions.

This module contains the original C-style processing functions for heart rate,
speed, and gain calculations used in VO2max and FTP cycling algorithms.

This is a legacy implementation that should be used for reference or
compatibility with existing systems.
"""

from .c_process_array import (
    c_process_features_array,
    c_clear_all,
    c_add_to_histogram,
    c_applySOSFilt,
    c_applyButterworth,
    c_applyFilter,
    c_computePercentile,
    c_compute_mean_and_std_cycling,
    c_compute_mean_and_std_ftp,
    c_compute_features,
    c_interpolate,
    c_process_window,
    c_count_size,
    Stats,
)

__all__ = [
    "c_process_features_array",
    "c_clear_all", 
    "c_add_to_histogram",
    "c_applySOSFilt",
    "c_applyButterworth",
    "c_applyFilter",
    "c_computePercentile",
    "c_compute_mean_and_std_cycling",
    "c_compute_mean_and_std_ftp",
    "c_compute_features",
    "c_interpolate",
    "c_process_window",
    "c_count_size",
    "Stats",
]
