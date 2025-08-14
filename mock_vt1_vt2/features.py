"""
Online feature summarization pipeline using provided C-like module.
Follows cookie-cutter data science structure: reads interim data and writes engineered features.
"""

import logging
from decimal import ROUND_HALF_UP, Decimal
from typing import Optional, List, Tuple
import ast

import numpy as np
import pandas as pd

from .config import DATA_INTERIM, INTERIM_DATA_FILE

logger = logging.getLogger(__name__)

# =========================
# Core module (as provided)
# =========================

# Run for every file/session
all_data = {}
gain_hist = {}
gain_pwm_hist = {}

OFFSET = 51  # HR offset (51-200)
hr_const = 0
speed_const = 0
gain_const = 0
gain_pwm_const = 0

last_valid_hr = 0
last_valid_speed = 0
last_valid_pwm = 0

# Filtering
hrStateSOSfilt = [[0, 0], [0, 0]]
isStateEmpty = 1


def c_clear_all():
    global hr_const
    global speed_const
    global gain_const
    global gain_pwm_const
    global last_valid_hr
    global last_valid_speed
    global last_valid_pwm
    global hrStateSOSfilt
    global isStateEmpty
    hr_const = 0
    speed_const = 0
    gain_const = 0
    gain_pwm_const = 0
    all_data.clear()
    gain_hist.clear()
    gain_pwm_hist.clear()

    last_valid_hr = 0
    last_valid_speed = 0
    last_valid_pwm = 0

    hrStateSOSfilt = [[0, 0], [0, 0]]
    isStateEmpty = 1
    return


class Stats:
    def __init__(
        self,
        idx,
        count,
        hr_sum=None,
        hr_sum2=None,
        speed_sum=None,
        speed_sum2=None,
        gain_sum=None,
        gain_sum2=None,
    ):
        self.idx = idx
        self.count = count
        self.hr_sum = hr_sum
        self.hr_sum2 = hr_sum2
        self.speed_sum = speed_sum
        self.speed_sum2 = speed_sum2
        self.gain_sum = gain_sum
        self.gain_sum2 = gain_sum2

    def __str__(self):
        return f"ct: {self.count}  hr_s: {self.hr_sum}  hr_s2: {self.hr_sum2}  speed_s: {self.speed_sum}  speed_s2: {self.speed_sum2}  gain_s: {self.gain_sum}  gain_s2: {self.gain_sum2}"


def c_add_to_histogram(hr, speed, gain, algorithm, is_powermeter_available, verbose):
    global hr_const
    global speed_const
    global gain_const
    global gain_pwm_const

    if hr_const == 0:
        hr_const = hr
    if speed_const == 0:
        speed_const = speed
    if gain_const == 0:
        gain_const = gain

    count = 1
    hr_sum = hr - hr_const
    hr_sum2 = (hr - hr_const) * (hr - hr_const)
    speed_sum = speed - speed_const
    speed_sum2 = (speed - speed_const) * (speed - speed_const)
    gain_sum = gain - gain_const
    gain_sum2 = (gain - gain_const) * (gain - gain_const)

    # MAIN STRUCTURE
    hr_idx = int(Decimal(hr).to_integral_value(rounding=ROUND_HALF_UP)) - OFFSET
    # NOTE: Default round method from python and numpy relies on rounding .5 up and down for odd and even numbers. At the C code, every .5 number is rounded up. To replicate the C behaviour it's needed to use the Decimal library, and set the rounding method to be the same as the C code.

    if verbose:
        print(f"      * Added hr_idx: {hr_idx}, hr: {hr}, speed: {speed}, gain: {gain}")
    idx = hr_idx
    if idx in all_data.keys():  # Increment current data
        all_data.update(
            {
                idx: Stats(
                    idx,
                    all_data.get(idx).count + 1,
                    all_data.get(idx).hr_sum + hr_sum,
                    all_data.get(idx).hr_sum2 + hr_sum2,
                    all_data.get(idx).speed_sum + speed_sum,
                    all_data.get(idx).speed_sum2 + speed_sum2,
                    all_data.get(idx).gain_sum + gain_sum,
                    all_data.get(idx).gain_sum2 + gain_sum2,
                )
            }
        )
    else:  # Create new key
        all_data.update(
            {
                idx: Stats(
                    idx,
                    count,
                    hr_sum,
                    hr_sum2,
                    speed_sum,
                    speed_sum2,
                    gain_sum,
                    gain_sum2,
                )
            }
        )

    if algorithm == "vo2max-cycling":
        # GAIN STRUCTURE
        # Round gain values to the closest .5 value and double it to select the array index.
        # E.g. gain 15.43 rounds to 15.5 and turned to be gainIdx 31
        gain_idx = int(2 * np.round(2 * gain) / 2)
        idx = gain_idx
        if idx in gain_hist.keys():  # Increment current data
            gain_hist.update({idx: Stats(idx, gain_hist.get(idx).count + 1)})
        else:  # Create new key
            gain_hist.update({idx: Stats(idx, count)})
    elif algorithm == "ftp-cycling":
        # GAIN STRUCTURE
        # Round gain values to the closest .5 value and double it to select the array index.
        # E.g. gain 15.43 rounds to 15.5 and turned to be gainIdx 31
        gain_idx = int(2 * np.round(2 * gain) / 2)
        # Round gain values to the closest .1 value to select the array index.
        # For instance, 0.9099 of Power Meter gain becomes "9.099" and its index is "9"
        gain_pwm_idx = int(
            Decimal(gain / 0.1).to_integral_value(rounding=ROUND_HALF_UP)
        )

        if is_powermeter_available:
            idx = gain_pwm_idx
            if idx in gain_pwm_hist.keys():  # Increment current data
                gain_pwm_hist.update(
                    {idx: Stats(idx, gain_pwm_hist.get(idx).count + 1)}
                )
            else:  # Create new key
                gain_pwm_hist.update({idx: Stats(idx, count)})
        else:
            idx = gain_idx
            if idx in gain_hist.keys():  # Increment current data
                gain_hist.update({idx: Stats(idx, gain_hist.get(idx).count + 1)})
            else:  # Create new key
                gain_hist.update({idx: Stats(idx, count)})
    else:
        raise ValueError("algorithm must be either 'vo2max-cycling' or 'ftp-cycling'")

    return


def c_applySOSFilt(data: float):
    sos_num = [
        [0.000000439093, 0.000000878186, 0.000000439093],
        [1.000000000000, 2.000000000000, 1.000000000000],
    ]
    sos_den = [
        [1.000000000000, -1.905141444098, 0.907755957334],
        [1.000000000000, -1.958043178328, 0.960730291048],
    ]
    x_cur = data

    for s in range(0, 2):
        x_new = sos_num[s][0] * x_cur + hrStateSOSfilt[s][0]
        hrStateSOSfilt[s][0] = (
            sos_num[s][1] * x_cur - sos_den[s][1] * x_new + hrStateSOSfilt[s][1]
        )
        hrStateSOSfilt[s][1] = sos_num[s][2] * x_cur - sos_den[s][2] * x_new
        x_cur = x_new
    y = x_cur
    return y


def c_applyButterworth(
    data: pd.Series, dataSize: int, outputData: pd.Series, isStateEmpty: int
):
    y = 0.0
    for i in range(0, dataSize):
        y = c_applySOSFilt(data[i])
        if isStateEmpty == 1:
            continue
        else:
            if y >= 0 + OFFSET and y <= 150 + OFFSET:
                outputData[i] = y
    return outputData


def c_applyFilter(data: pd.Series):
    global isStateEmpty
    inputDataSize = 30
    data = data.reset_index(drop=True)
    data_slice = c_applyButterworth(data, inputDataSize, data, isStateEmpty)
    isStateEmpty = 0
    return data_slice


def c_isDecimalNumber(number: float):
    if number == int(number):
        return 0
    return 1


def c_computePercentile(mode: int, data: dict, dataSize: int, percentile: float = 0.75):
    pIdx = (percentile * (dataSize - 1)) + 1
    pMeanFlag = 0
    pDecimalPart = 0.0
    if c_isDecimalNumber(pIdx):
        pMeanFlag = 1
        pDecimalPart = pIdx - np.floor(pIdx)
        pIdx = np.floor(pIdx)

    pInterval = dataSize - int(np.floor(pIdx))
    pThreshold = pInterval + 1
    lastStoredIdx = 0
    prevPThreshold = 0

    for key in sorted(list(data.keys()), reverse=True):
        if pMeanFlag:
            pThreshold = pThreshold - data.get(key).count
            if pThreshold <= 0:
                if prevPThreshold < (pInterval):
                    if mode == 0:
                        return key + OFFSET
                    elif mode == 1:
                        return float(key / 2.0)
                    else:
                        return float(key * 0.1)
                if mode == 0:
                    return (key + OFFSET) + pDecimalPart * float(
                        ((lastStoredIdx + OFFSET) - (key + OFFSET))
                    )
                elif mode == 1:
                    return (float(key / 2.0)) + pDecimalPart * float(
                        ((float(lastStoredIdx / 2.0)) - (float(key / 2.0)))
                    )
                else:
                    return (key * 0.1) + pDecimalPart * float(
                        (lastStoredIdx * 0.1) - (key * 0.1)
                    )
            prevPThreshold = prevPThreshold + data.get(key).count
            lastStoredIdx = key

        else:
            pThreshold = pThreshold - data.get(key).count
            if pThreshold <= 0:
                if mode == 0:
                    return key + OFFSET
                elif mode == 1:
                    return float(key / 2)
                else:
                    return float(key * 0.1)
    return 0.0


def c_compute_mean_and_std_cycling(percentile: float, data: dict):
    threshold = int(np.ceil(percentile))
    count = 0
    sum_hr = 0
    sum2_hr = 0
    sum_speed = 0
    sum2_speed = 0
    sum_gain = 0
    sum2_gain = 0

    for key in sorted(list(data.keys()), reverse=True):
        if (key + OFFSET) < threshold:
            break
        sum_hr += data.get(key).hr_sum
        sum2_hr += data.get(key).hr_sum2
        sum_speed += data.get(key).speed_sum
        sum2_speed += data.get(key).speed_sum2
        sum_gain += data.get(key).gain_sum
        sum2_gain += data.get(key).gain_sum2
        count += data.get(key).count

    hr_mean = hr_const + (sum_hr / count)
    speed_mean = speed_const + (sum_speed / count)
    gain_mean = gain_const + (sum_gain / count)

    hr_std = np.sqrt((sum2_hr - (sum_hr * sum_hr) / count) / (count - 1))
    speed_std = np.sqrt((sum2_speed - (sum_speed * sum_speed) / count) / (count - 1))
    gain_std = np.sqrt((sum2_gain - (sum_gain * sum_gain) / count) / (count - 1))

    return hr_mean, hr_std, speed_mean, speed_std, gain_mean, gain_std


def c_compute_mean_and_std_ftp(percentile: float, data: dict):
    threshold = int(np.ceil(percentile))
    count = 0
    sum_hr = 0
    sum2_hr = 0
    sum_speed = 0
    sum2_speed = 0
    sum_gain = 0
    sum2_gain = 0

    for key in sorted(list(data.keys()), reverse=True):
        if (key + OFFSET) < threshold:
            break
        sum_hr += data.get(key).hr_sum
        sum2_hr += data.get(key).hr_sum2
        sum_speed += data.get(key).speed_sum
        sum2_speed += data.get(key).speed_sum2
        sum_gain += data.get(key).gain_sum
        sum2_gain += data.get(key).gain_sum2
        count += data.get(key).count

    hr_mean = hr_const + (sum_hr / count)
    speed_mean = speed_const + (sum_speed / count)
    gain_mean = gain_const + (sum_gain / count)

    hr_std = np.sqrt(abs((sum2_hr / count) - ((sum_hr / count) * (sum_hr / count))))
    speed_std = np.sqrt(
        abs((sum2_speed / count) - ((sum_speed / count) * (sum_speed / count)))
    )
    gain_std = np.sqrt(
        abs((sum2_gain / count) - ((sum_gain / count) * (sum_gain / count)))
    )

    return hr_mean, hr_std, speed_mean, speed_std, gain_mean, gain_std


def c_compute_features(
    algorithm: str, is_powermeter_available: bool, verbose: bool = False
):
    if algorithm == "vo2max-cycling":
        size = c_count_size(all_data)
        percentile_hr = c_computePercentile(0, all_data, size, 0.75)
        size = c_count_size(gain_hist)
        percentile_gain = c_computePercentile(1, gain_hist, size, 0.75)
        hr_mean, hr_std, speed_mean, speed_std, gain_mean, gain_std = (
            c_compute_mean_and_std_cycling(percentile_hr, all_data)
        )
    elif algorithm == "ftp-cycling":
        size = c_count_size(all_data)
        percentile_hr = c_computePercentile(0, all_data, size, 0.75)
        if is_powermeter_available:
            size = c_count_size(gain_pwm_hist)
            percentile_gain = c_computePercentile(2, gain_pwm_hist, size, 0.75)
        else:
            size = c_count_size(gain_hist)
            percentile_gain = c_computePercentile(1, gain_hist, size, 0.75)
        hr_mean, hr_std, speed_mean, speed_std, gain_mean, gain_std = (
            c_compute_mean_and_std_ftp(percentile_hr, all_data)
        )
    else:
        raise ValueError("algorithm must be either 'vo2max-cycling' or 'ftp-cycling'")
    return (
        hr_mean,
        hr_std,
        speed_mean,
        speed_std,
        gain_mean,
        gain_std,
        percentile_hr,
        percentile_gain,
    )


def c_interpolate(input: pd.Series, last_valid_value: float):
    # Return if no need to interpolate
    if input.isna().sum() == 0:
        return input.reset_index(drop=True), input.reset_index(drop=True)[
            input.size - 1
        ]
    series = input.reset_index(drop=True)
    series = series.interpolate()
    if last_valid_value == 0:
        first_value = series[series.first_valid_index()]
        series = series.fillna(first_value)
    else:
        nan_count = series.isna().sum()
        fillFactor = (series[series.first_valid_index()] - last_valid_value) / (
            series.first_valid_index() + 1
        )
        for i in range(0, series.first_valid_index()):
            series[i] = last_valid_value + ((i + 1) * fillFactor)
    # Update last_valid_value
    last_valid_value = series[series.size - 1]
    return series, last_valid_value


def c_process_window(
    hr: pd.Series, speed: pd.Series, powermeter: pd.Series, algorithm: str
):
    global last_valid_hr
    global last_valid_speed

    hr_int, last_valid_hr = c_interpolate(hr, last_valid_hr)
    speed_int, last_valid_speed = c_interpolate(speed, last_valid_speed)

    # Apply butterworth at HR signal
    if algorithm == "vo2max-cycling":
        hr_int = c_applyFilter(hr_int)

    # Powermeter replacements
    powermeter = powermeter.reset_index(drop=True)
    for i in range(0, len(powermeter)):
        if pd.isna(powermeter[i]):
            powermeter[i] = -1
        if powermeter[i] < 50 and powermeter[i] != -1:
            powermeter[i] = 50

    return hr_int, speed_int, powermeter


def c_count_size(data: dict):
    size = 0
    for key in data.keys():
        size = size + data.get(key).count
    return size


def c_process_features_array(
    hr: pd.Series,
    speed: pd.Series,
    powermeter: pd.Series = [],
    algorithm: str = "vo2max-cycling",
    verbose: bool = False,
):
    if verbose:
        print(f">>> Processing {algorithm} series")
        print("    - Cleaning global state")
    c_clear_all()
    if len(hr) != len(speed):
        raise ValueError("series must have the same length")

    # Fill powermeter (-1 as error/invalid)
    is_powermeter_available = 0
    if len(powermeter) == 0:
        powermeter = pd.Series(np.full(len(hr), -1))
        if verbose:
            print(
                f"      * is_powermeter_available: {is_powermeter_available}, empty column. Filling with '-1'"
            )
    else:
        powermeter = powermeter.fillna(-1)
        is_powermeter_available = 1
        if verbose:
            print(
                f"      * is_powermeter_available: {is_powermeter_available}. Filling gaps with '-1'"
            )
    if algorithm == "vo2max-cycling":
        hr[hr < 51] = np.nan
        hr[hr > 200] = np.nan
        speed[speed < 3] = np.nan
        speed[speed > 210.4] = np.nan
        speed[speed == 65535] = np.nan
    elif algorithm == "ftp-cycling":
        if len(hr) != len(powermeter):
            raise ValueError("series must have the same length")
            # TODO: Check for valid FTP data (-1, 0, %valid, etc)
        hr[hr < 51] = np.nan
        hr[hr > 200] = np.nan
        speed[speed < 3] = np.nan
        speed[speed > 210.4] = np.nan
        speed[speed == 65535] = np.nan
    else:
        raise ValueError("algorithm must be either 'vo2max-cycling' or 'ftp-cycling'")

    # Process
    windows = len(hr) // 30
    if windows == 0:
        raise ValueError("not enought data to compute features")

    full_hr = pd.Series()
    full_speed = pd.Series()
    full_pwm = pd.Series()
    full_gain = pd.Series()
    treated_gain = pd.Series()  #

    if verbose:
        print("    - Processing data slices")

    for w in range(windows):
        # Data slices
        hr_slice = hr[(w * 30) + 0 : (w * 30) + 30]
        speed_slice = speed[(w * 30) + 0 : (w * 30) + 30]
        powermeter_slice = powermeter[(w * 30) + 0 : (w * 30) + 30]

        hr_slice = pd.Series(hr_slice)
        speed_slice = pd.Series(speed_slice)
        powermeter_slice = pd.Series(powermeter_slice)

        # Check invalid thresholds
        if hr_slice.isna().sum() > 10 or speed_slice.isna().sum() > 10:
            if verbose:
                print(
                    f"      * w[{w}] SKIPPED! no_hr: {hr_slice.isna().sum()} no_speed: {speed_slice.isna().sum()}"
                )
            continue

        # Check negative slope HR window
        gain_slice = hr_slice / speed_slice
        if full_gain.empty:
            full_gain = gain_slice
        else:
            full_gain = pd.concat([full_gain, gain_slice], ignore_index=True)

        if (hr_slice.values[-1] - hr_slice.values[0] > 0) and (
            speed_slice.values[-1] - speed_slice.values[0] > 0
        ):
            if treated_gain.empty:
                treated_gain = gain_slice
            else:
                treated_gain = pd.concat([treated_gain, gain_slice], ignore_index=True)

        # Process data
        hri, speedi, powermeteri = c_process_window(
            hr_slice, speed_slice, powermeter_slice, algorithm
        )

        if full_hr.empty:
            full_hr = hri
        else:
            full_hr = pd.concat([full_hr, hri], ignore_index=True)

        if full_speed.empty:
            full_speed = speedi
        else:
            full_speed = pd.concat([full_speed, speedi], ignore_index=True)

        if full_pwm.empty:
            full_pwm = powermeteri
        else:
            full_pwm = pd.concat([full_pwm, powermeteri], ignore_index=True)

        if verbose:
            print(f"      * w[{w}] Processed succesfully")

    if verbose:
        print("    - Filling data structures")
    # Features
    hr_mean = 0
    hr_std = 0
    speed_mean = 0
    speed_std = 0
    gain_mean = 0
    gain_std = 0
    percentile_hr = 0
    percentile_gain = 0

    iter = 0
    if algorithm == "vo2max-cycling":
        for i in range(32, len(full_hr)):
            speed_median = full_speed[i - 32 : i - 2].median()
            gain_buffer = pd.Series(
                [
                    full_hr[i - 32] / full_speed[i - 32 : i - 2].median(),
                    full_hr[i - 31] / full_speed[i - 31 : i - 1].median(),
                    full_hr[i - 30] / full_speed[i - 30 : i - 0].median(),
                ]
            )
            gain_median = gain_buffer.median()
            c_add_to_histogram(
                full_hr[i - 32],
                speed_median,
                gain_median,
                algorithm,
                is_powermeter_available,
                verbose,
            )
            iter += 1
            if iter == 30:
                iter = 0
                (
                    hr_mean,
                    hr_std,
                    speed_mean,
                    speed_std,
                    gain_mean,
                    gain_std,
                    percentile_hr,
                    percentile_gain,
                ) = c_compute_features(
                    algorithm, is_powermeter_available, verbose=verbose
                )
                if verbose:
                    print("    - Computing features")
                    print(f"      * percHR:         {percentile_hr}")
                    print(f"      * PercGain:       {percentile_gain}")
                    print(f"      * HR Mean:        {hr_mean}")
                    print(f"      * HR Std.Dev:     {hr_std}")
                    print(f"      * Speed Mean:     {speed_mean}")
                    print(f"      * Speed Std.Dev:  {speed_std}")
                    print(f"      * Gain Mean:      {gain_mean}")
                    print(f"      * Gain Std.Dev:   {gain_std}")
                    print("")

    elif algorithm == "ftp-cycling":
        gain_buffer = []
        for i in range(30, len(full_hr)):
            hr_filt = full_hr[i - 30 : (i - 30) + 30].mean()
            speed_filt = full_speed[i - 30 : (i - 30) + 30].mean()
            pwm_filt = full_pwm[i - 30 : (i - 30) + 30].mean()

            gain_selected = hr_filt / speed_filt
            if is_powermeter_available:
                gain_selected = hr_filt / pwm_filt
            c_add_to_histogram(
                hr_filt,
                speed_filt,
                gain_selected,
                algorithm,
                is_powermeter_available,
                verbose,
            )

            iter += 1
            if iter == 30:
                iter = 0
                (
                    hr_mean,
                    hr_std,
                    speed_mean,
                    speed_std,
                    gain_mean,
                    gain_std,
                    percentile_hr,
                    percentile_gain,
                ) = c_compute_features(
                    algorithm, is_powermeter_available, verbose=verbose
                )
                if verbose:
                    print("    - Computing features")
                    print(f"      * percHR:         {percentile_hr}")
                    print(f"      * PercGain:       {percentile_gain}")
                    print(f"      * HR Mean:        {hr_mean}")
                    print(f"      * HR Std.Dev:     {hr_std}")
                    print(f"      * Speed Mean:     {speed_mean}")
                    print(f"      * Speed Std.Dev:  {speed_std}")
                    print(f"      * Gain Mean:      {gain_mean}")
                    print(f"      * Gain Std.Dev:   {gain_std}")
                    print("")
    else:
        raise ValueError("algorithm must be either 'vo2max-cycling' or 'ftp-cycling'")

    return (
        hr_mean,
        hr_std,
        speed_mean,
        speed_std,
        gain_mean,
        gain_std,
        percentile_hr,
        percentile_gain,
    )

# ==============================
# Thin wrapper for pipeline use
# ==============================

def c_process_array(
    hr: pd.Series,
    speed: pd.Series,
    powermeter: Optional[pd.Series] = None,
    algorithm: str = "vo2max-cycling",
    verbose: bool = False,
) -> Tuple[float, float, float, float, float, float, float, float]:
    """Wrapper that delegates to c_process_features_array to match requested name."""
    pwm_series = powermeter if powermeter is not None else []
    return c_process_features_array(hr, speed, pwm_series, algorithm, verbose)


# ============================================
# Cookie-cutter friendly feature engineering
# ============================================

def _parse_sequence(obj) -> List[float]:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, (pd.Series, np.ndarray)):
        return list(obj)
    if isinstance(obj, str):
        try:
            parsed = ast.literal_eval(obj)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        # Fallback: split by comma
        try:
            return [float(x) for x in obj.strip("[]").split(",") if x.strip() != ""]
        except Exception:
            return []
    return []


def compute_session_features(
    hr_seq: List[float],
    speed_seq: List[float],
    powermeter_seq: Optional[List[float]] = None,
    prefer_algorithm: Optional[str] = None,
    verbose: bool = False,
) -> dict:
    """Compute summary features for one session using the provided online algorithm.

    Returns a dict with keys: hr_mean, hr_std, speed_mean, speed_std, gain_mean, gain_std,
    percentile_hr, percentile_gain, algorithm.
    """
    hr_series = pd.Series(hr_seq)
    speed_series = pd.Series(speed_seq)
    pwm_series = pd.Series(powermeter_seq) if powermeter_seq is not None else None

    # Decide algorithm
    if prefer_algorithm is not None:
        algorithm = prefer_algorithm
    else:
        has_pwm = (
            pwm_series is not None
            and len(pwm_series) > 0
            and pwm_series.replace(-1, np.nan).notna().sum() > 0
        )
        algorithm = "ftp-cycling" if has_pwm else "vo2max-cycling"

    # Run computation
    try:
        (
            hr_mean,
            hr_std,
            speed_mean,
            speed_std,
            gain_mean,
            gain_std,
            percentile_hr,
            percentile_gain,
        ) = c_process_array(hr_series, speed_series, pwm_series, algorithm=algorithm, verbose=verbose)
    except ValueError as e:
        # Not enough data or invalid inputs
        logger.warning(f"Session skipped: {e}")
        return {
            "hr_mean": np.nan,
            "hr_std": np.nan,
            "speed_mean": np.nan,
            "speed_std": np.nan,
            "gain_mean": np.nan,
            "gain_std": np.nan,
            "percentile_hr": np.nan,
            "percentile_gain": np.nan,
            "algorithm": algorithm,
        }

    return {
        "hr_mean": hr_mean,
        "hr_std": hr_std,
        "speed_mean": speed_mean,
        "speed_std": speed_std,
        "gain_mean": gain_mean,
        "gain_std": gain_std,
        "percentile_hr": percentile_hr,
        "percentile_gain": percentile_gain,
        "algorithm": algorithm,
    }


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def create_features_from_interim_df(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Create engineered features for each session row in the interim dataframe."""
    hr_col = _find_column(df, ["hr", "heart_rate", "heartRate"])
    speed_col = _find_column(df, ["speed", "velocity"])
    pwm_col = _find_column(df, ["powermeter", "power", "pwm", "power_meter"])

    if hr_col is None or speed_col is None:
        raise ValueError("Input dataframe must contain 'hr' and 'speed' (or equivalent) columns")

    features_rows: List[dict] = []

    for idx, row in df.iterrows():
        hr_list = _parse_sequence(row[hr_col])
        speed_list = _parse_sequence(row[speed_col])
        pwm_list = _parse_sequence(row[pwm_col]) if pwm_col is not None else None

        # Clear internal state per session
        c_clear_all()

        feats = compute_session_features(hr_list, speed_list, pwm_list, verbose=verbose)
        feats["session_index"] = idx
        features_rows.append(feats)

    return pd.DataFrame(features_rows)


def save_features(features_df: pd.DataFrame, filename: str = "engineered_features.csv") -> None:
    output_path = DATA_INTERIM / filename
    features_df.to_csv(output_path, index=False)
    logger.info(f"Saved engineered features to {output_path}")


def main():
    logger.info("Starting online feature summarization pipeline...")

    try:
        interim_df = pd.read_csv(INTERIM_DATA_FILE)
        logger.info(f"Loaded interim data from {INTERIM_DATA_FILE} with shape {interim_df.shape}")
    except FileNotFoundError:
        logger.error("Interim data not found. Run dataset.py first.")
        return

    features_df = create_features_from_interim_df(interim_df, verbose=False)
    save_features(features_df)

    logger.info("Feature engineering pipeline completed!")


if __name__ == "__main__":
    main()
