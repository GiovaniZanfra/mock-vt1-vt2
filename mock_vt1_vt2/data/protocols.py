"""PROTOCOLS.py FILE"""

from typing import Dict, Iterable, Optional, Protocol

import pandas as pd
from fitness.data.parser import InterfaceSession


class SessionLoader(Protocol):
    def load_sessions(self) -> Iterable[InterfaceSession]: ...


class SignalCollector(Protocol):
    def collect(self, session: object, sid: str, idx: str) -> Optional[pd.Series]: ...


class TargetCollector(Protocol):
    def collect(self, sid: str, session: object, extra: dict) -> Dict[str, object]: ...
