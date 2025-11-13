from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Tracker(ABC):
    @abstractmethod
    def update(
        self, boxes: list[list[float]], scores: list[float], frame: Any = None
    ) -> list[int]: ...
