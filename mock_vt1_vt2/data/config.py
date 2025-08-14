"""CONFIG.py FILE"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import yaml


# ----- Configuration -----
@dataclass(frozen=True)
class DataLoadConfig:
    db_path: Path
    protocol_list: List[str]
    device: str
    labels: Dict[str, str]
    exercise: str
    target: str
    output_path: Path
    droplist: bool
    augmentation: bool
    augmentation_n: int
    augmentation_seed: int

    @staticmethod
    def from_yaml(path: Path) -> "DataLoadConfig":
        cfg = yaml.safe_load(path.read_text())
        dl = cfg["data_load"]
        gen = cfg["general"]
        return DataLoadConfig(
            db_path=Path(dl["db_path"]),
            protocol_list=dl["protocol_list"],
            device=dl["device"],
            labels={
                "hr": dl["labels"]["hr"],
                "speed": dl["labels"]["speed"],
                "power": dl["labels"]["power"],
            },
            exercise=dl["exercise"],
            target=gen.get("target_column", ""),
            output_path=Path(dl["output_path"]),
            droplist=dl["droplist"],
            augmentation=dl["augmentation"],
            augmentation_n=dl["augmentation_n"],
            augmentation_seed=dl["augmentation_seed"],
        )
