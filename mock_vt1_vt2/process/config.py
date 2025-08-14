from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProcessingConfig:
    input_path: Path
    output_path: Path

    @staticmethod
    def from_dict(d):
        return ProcessingConfig(
            input_path=Path(d["input"]), output_path=Path(d["output"])
        )


@dataclass(frozen=True)
class FeaturesConfig:
    input_path: Path
    output_path: Path

    @staticmethod
    def from_dict(d):
        return FeaturesConfig(
            input_path=Path(d["input"]), output_path=Path(d["output"])
        )
