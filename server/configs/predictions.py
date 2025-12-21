from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PredictionsConfig:
    model_name: str
    task: Literal["track", "predict"]
    tracker: Literal["bytetrack", "botsort"]
    conf: float
    retina_masks: bool
