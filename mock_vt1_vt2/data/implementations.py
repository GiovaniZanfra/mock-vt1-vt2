"""IMPLEMENTATION.py FILE"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd
from data.config import DataLoadConfig
from fitness.data import DataLoader
from fitness.data.parser import InterfaceSession, MetadataDB
from fitness.features.anthropometric_features import calculate_bmi
from fitness.utils.BMI import get_BMI_classes

GENDER_MAP = {"Male": 1.0, "Female": 0.0}


class AILabSessionLoader:
    def __init__(self, cfg: DataLoadConfig):
        self.cfg = cfg
        meta = MetadataDB(cfg.db_path)
        self.loader = DataLoader(cfg.db_path, metadata_db=meta, bound_protocol=False)

    def load_sessions(self) -> Iterable[InterfaceSession]:
        self.loader.load_sessions(pc_subset=self.cfg.protocol_list, pc_mode="keep")
        for idx, session in self.loader.loaded_sessions.items():
            yield session


@dataclass
class BasicSignalCollector:
    label: str
    device: str

    def collect(
        self, session: InterfaceSession, sid: str, idx: str
    ) -> Optional[pd.Series]:
        model_data = session.get_data("model")
        device_data = model_data.get(self.device)
        if device_data is None:
            return None
        return device_data.get(self.label)


@dataclass
class AnthropometricsCollector:
    demographic_df: pd.DataFrame
    label: str = "anthro"

    def collect(self, session: InterfaceSession, sid: str, idx: str) -> Dict[str, Any]:
        df = self.demographic_df
        user = df[df.idx.str.contains(idx)].iloc[0]
        weight, height, age = user.weight, user.height, user.age
        gender = GENDER_MAP[user.gender]
        bmi = calculate_bmi(weight, height)
        bmi_class = get_BMI_classes(
            pd.DataFrame({"weight": [weight], "height": [height]})
        )[0]
        return {
            "age": age,
            "weight": weight,
            "height": height,
            "gender": gender,
            "bmi": bmi,
            "bmi_class": bmi_class,
        }


@dataclass
class FTPCollector:
    db_path: Path
    power_label: str

    def collect(
        self, sid: str, session: InterfaceSession, extra: Dict[str, Any]
    ) -> Dict[str, Any]:
        max_df = pd.read_parquet(self.db_path / "metadata" / "maximal_bk.parquet")
        ftp_df = pd.read_parquet(self.db_path / "metadata" / "ftp.parquet")
        if sid not in ftp_df.sid.values:
            return {}
        ftp = ftp_df.loc[ftp_df.sid == sid, "ftp_label"].item()
        pmax = max_df.loc[max_df.sid == sid, "peak_power"].item()
        out = {"ftp": ftp, "ftp_per_kg": ftp / extra["weight"], "pmax": pmax}
        raw = session.get_data("raw")["timeseries"]
        if self.power_label in raw.columns:
            out["power_meter"] = True
            out["power"] = raw[self.power_label]
        else:
            out["power_meter"] = False
            out["power"] = pd.Series([], dtype=float)
        return out


@dataclass
class CyclingATCollector:
    db_path: Path

    def collect(
        self, sid: str, session: InterfaceSession, extra: Dict[str, Any]
    ) -> Dict[str, Any]:
        df = pd.read_parquet(self.db_path / "metadata" / "maximal_bk.parquet")
        df = df.groupby("sid").last().reset_index()
        row = df[df.sid == sid]
        return (
            {"at": row["at"].item(), "ant": row["ant"].item()} if not row.empty else {}
        )


@dataclass
class ATCollector:
    db_path: Path

    def collect(
        self, sid: str, session: InterfaceSession, extra: Dict[str, Any]
    ) -> Dict[str, Any]:
        df = pd.read_parquet(self.db_path / "metadata" / "maximal.parquet")
        df = df.groupby("sid").last().reset_index()
        row = df[df.sid == sid]
        return (
            {"at": row["at"].item(), "ant": row["ant"].item()} if not row.empty else {}
        )


@dataclass
class HRMaxCollector:
    db_path: Path

    def collect(
        self, sid: str, session: InterfaceSession, extra: Dict[str, Any]
    ) -> Dict[str, Any]:
        df = pd.read_parquet(self.db_path / "metadata" / "maximal_bk.parquet")
        df = df.groupby("sid").last().reset_index()
        row = df[df.sid == sid]
        return {"hr_max": row.hr_max.item()} if not row.empty else {}
