# protocols.py
from typing import Protocol

import pandas as pd


class Stage(Protocol):
    def run(self, df: pd.DataFrame) -> pd.DataFrame: ...
