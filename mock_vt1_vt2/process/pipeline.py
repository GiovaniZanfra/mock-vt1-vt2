# src/features/pipeline.py
from typing import List

import pandas as pd
from process.protocols import Stage


class Pipeline:
    def __init__(self, stages: List[Stage]):
        self.stages = stages

    def run(self) -> pd.DataFrame:
        df = None
        for stage in self.stages:
            df = stage.run(df)
        return df
