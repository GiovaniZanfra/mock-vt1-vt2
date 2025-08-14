"""DATASET.py FILE"""

from pathlib import Path

import pandas as pd
import yaml
from data.config import DataLoadConfig
from data.implementations import (
    AILabSessionLoader,
    AnthropometricsCollector,
    ATCollector,
    BasicSignalCollector,
    CyclingATCollector,
    FTPCollector,
    HRMaxCollector,
)
from data.pipeline import DataLoadPipeline

cfg: DataLoadConfig = DataLoadConfig.from_yaml(Path("conf/project.yaml"))
gen = yaml.safe_load(open("conf/project.yaml"))["general"]
ecg_path = gen["ecg_path"]
ecg_cols = gen["columns"]["ecg"]
ecg_df = pd.read_csv(ecg_path)
# session loader
loader = AILabSessionLoader(cfg)
# prepare demographic df
demo_df = pd.read_parquet(cfg.db_path / "metadata" / "demographic.parquet")
# signal collectors include anthro as collector
signals = [
    AnthropometricsCollector(demo_df),
    BasicSignalCollector(cfg.labels["hr"], cfg.device),
    BasicSignalCollector(cfg.labels["speed"], cfg.device),
]
# target collector based on config
if cfg.exercise == "cycling":
    if cfg.target == "ftp":
        target = FTPCollector(cfg.db_path, cfg.labels["power"])
    elif cfg.target in {"at", "ant"}:
        target = CyclingATCollector(cfg.db_path)
    elif cfg.target == "hrmax":
        target = HRMaxCollector(cfg.db_path)
    else:
        raise ValueError(f"Unknown target: {cfg.target}")
elif (cfg.exercise == "running") or (cfg.exercise == "walking"):
    if cfg.target in {"at", "ant"}:
        target = ATCollector(cfg.db_path)
else:
    raise NotImplementedError("Exercise not implemented: " + cfg.exercise)
pipeline = DataLoadPipeline(cfg, loader, signals, target)
pipeline.run()
