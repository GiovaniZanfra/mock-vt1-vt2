# src/features/stages.py
import sys

# Ajuste de path para importação de módulo específico
sys.path.append(
    "/home/g-brandao/workspace/crf-sandbox/experiments/cycling/ftp/deploy/cpy_features"
)

import json
from pathlib import Path

import pandas as pd

# alternatively, if installed as package:
# from cycling.ftp.deploy.cpy_features.cycling.cycling import c_process_features_array
from fitness.features.anthropometric_features import (
    calculate_anthropometric_ftp,
    calculate_relative_body_fat,
    calculate_vo2max_myers,
)
from fitness.features.hr_features import calculate_age_hr_max, calculate_hr_mean_hr_std
from fitness.features.speed_features import (
    calculate_kinetic_energy,
    calculate_momentum,
    calculate_speed_mean_speed_std,
)
from near_bia_features.nearBIA import near_bia_extractor
from process.c_process_array import (
    c_process_features_array,
)


def _parse_series(value) -> pd.Series:
    """
    Converte valor de célula em pd.Series de floats.
    Aceita:
      - JSON string with NaN: '[nan, 1.0, ...]'
      - Python list repr: '[1.0, 2.0, 3.0]'
      - pandas Series repr: '0    1.0\n1    2.0\n...'
      - Lista Python
    """
    # lista pura
    if isinstance(value, list):
        return pd.Series(value, dtype=float)
    # pandas Series
    if isinstance(value, pd.Series):
        return value.astype(float)
    # string possível
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return pd.Series([], dtype=float)
        # tenta JSON com suporte a NaN via parse_constant
        try:
            parsed = json.loads(s, parse_constant=lambda x: float("nan"))
            return pd.Series(parsed, dtype=float)
        except Exception:
            pass
        # tenta literal eval mas substitui nan por 'nan'
        try:
            # replace bare nan to float('nan') in expression
            expr = s.replace("nan", 'float("nan")')
            parsed = eval(expr)
            return pd.Series(parsed, dtype=float)
        except Exception:
            pass
        # fallback repr de Series
        lines = s.splitlines()
        parsed = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            val = parts[-1]
            if val.lower() in ("nan", "none", "null", "na"):
                parsed.append(float("nan"))
            else:
                try:
                    parsed.append(float(val))
                except ValueError:
                    continue
        return pd.Series(parsed, dtype=float)
    # nulo ou não esperado
    return pd.Series([], dtype=float)


class ReadRawStage:
    def __init__(self, path: Path):
        self.path = path

    def run(self, df: pd.DataFrame = None) -> pd.DataFrame:
        return pd.read_csv(self.path)


class TimeSeriesFeatureStage:
    """Aplica c_process_features_array para hr/speed/gain."""

    def __init__(self, algorithm: str = "ftp-cycling", verbose: bool = False):
        self.algorithm = algorithm
        self.verbose = verbose

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        records = []
        for _, row in df.iterrows():
            hr_series = _parse_series(row.get("hr", []))
            speed_series = _parse_series(row.get("speed", []))
            (
                hr_mean,
                hr_std,
                speed_mean,
                speed_std,
                gain_mean,
                gain_std,
                percentile_hr,
                percentile_gain,
            ) = c_process_features_array(
                hr_series,
                speed_series,
                algorithm=self.algorithm,
                verbose=self.verbose,
            )

            records.append(
                {
                    "idx": row["idx"],
                    "variant_id": row["variant_id"],
                    "hr_mean": hr_mean,
                    "hr_std": hr_std,
                    "speed_mean": speed_mean,
                    "speed_std": speed_std,
                    "gain_mean": gain_mean,
                    "gain_std": gain_std,
                    "percentile_hr": percentile_hr,
                    "percentile_gain": percentile_gain,
                }
            )
        features_df = pd.DataFrame(records)
        print(
            f"COLUMNS AT TIMESERIESFEATUREUSAGE{df.merge(features_df, on=('idx', 'variant_id'), how='left').columns.tolist()}"
        )
        return df.merge(features_df, on=("idx", "variant_id"), how="left")


class AnthroFeatureStage:
    """Calcula ftp_anthropometric, vo2max_myers, relative_body_fat de forma vetorizada segura."""

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # calcula linha a linha para evitar ambiguidade de arrays
        df["ftp_anthropometric"] = df.apply(
            lambda row: calculate_anthropometric_ftp(
                float(row["weight"]), int(row["age"]), row["gender"]
            ),
            axis=1,
        )
        df["vo2max_myers"] = df.apply(
            lambda row: calculate_vo2max_myers(
                int(row["age"]), row["gender"], float(row["weight"])
            ),
            axis=1,
        )
        df["relative_body_fat"] = df.apply(
            lambda row: calculate_relative_body_fat(
                float(row["weight"]), float(row["height"])
            ),
            axis=1,
        )
        print(f"COLUMNS AT AnthroFeatureStage{df.columns.tolist()}")
        return df


class HRFeatureStage:
    """Age‐based HR max e comparação mean/std de HR."""

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["age_hr_max"] = df.apply(
            lambda row: calculate_age_hr_max(int(row["age"]))["hr_max"], axis=1
        )
        df["hr_mean_hr_std"] = df.apply(
            lambda row: calculate_hr_mean_hr_std(row["hr_mean"], hr_std=row["hr_std"]),
            axis=1,
        )
        print(f"COLUMNS AT HRFeatureStage{df.columns.tolist()}")
        return df


class RelativeMetricsStage:
    """Cria colunas relativas: ganho vs peso, BMI, body fat, inversos etc."""

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["gain_weight"] = df["gain_mean"] / df["weight"]
        df["gain_bmi"] = df["gain_mean"] / df["bmi"]
        df["gain_relative_bf"] = df["gain_mean"] / df["relative_body_fat"]
        df["gain_inverse"] = 1.0 / df["gain_mean"]
        df["gain_inverse_relative"] = (
            df["relative_body_fat"] * df["weight"] / df["gain_mean"]
        )
        print(f"COLUMNS AT RelativeMetricsStage{df.columns.tolist()}")
        return df


class CalculatedBiaStage:
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for index, row in df.iterrows():
            age = row["age"]
            gender = row["gender"]
            weight = row["weight"]
            height = row["height"]
            bia_results = near_bia_extractor(gender, age, height, weight)
            for k, v in bia_results.items():
                df.at[index, k] = v
        print(f"COLUMNS AT CalculatedBiaStage{df.columns.tolist()}")
        return df


class MergeEcgStage:
    def __init__(self, ecg_df, ecg_cols):
        self.ecg_df = ecg_df
        self.ecg_cols = ecg_cols

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Explicitly include 'sid' in the selection after groupby and last()
        ecg_df = (
            self.ecg_df.groupby("sid").last()[self.ecg_cols].reset_index()
        )  # Add 'sid' back
        df["date"] = df["idx"].apply(
            lambda x: pd.to_datetime(x.split("-")[-1]).strftime("%d/%m/%Y")
        )
        df["sid"] = df["sid"].astype(str)
        return df.merge(ecg_df, on=["sid"], how="left")


class SpeedMomentumStage:
    """Momentum, energia cinética e std of speed."""

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["linear_momentum"] = df.apply(
            lambda row: calculate_momentum(row["speed_mean"], row["weight"]), axis=1
        )
        df["kinetic_energy"] = df.apply(
            lambda row: calculate_kinetic_energy(row["speed_mean"], row["weight"]),
            axis=1,
        )
        df["speed_mean_speed_std"] = df.apply(
            lambda row: calculate_speed_mean_speed_std(
                row["speed_mean"], row["speed_std"]
            ),
            axis=1,
        )
        print(f"COLUMNS AT SpeedMomentumStage{df.columns.tolist()}")
        return df


class SaveFeaturesStage:
    def __init__(self, path: Path):
        self.path = path

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df.to_csv(self.path, index=False)
        return df
