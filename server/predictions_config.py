from dataclasses import dataclass


@dataclass(frozen=True)
class PredictionsConfig:
    model_name: str
    task: str
    tracker: str
    conf: float
    retina_masks: bool
