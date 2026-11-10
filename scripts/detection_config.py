"""Experimental settings, not clinical or validated decision boundaries."""

from dataclasses import asdict, dataclass
from math import isfinite


@dataclass(frozen=True)
class DetectionConfig:
    ear_threshold: float = 0.12
    mar_threshold: float = 0.5
    closure_seconds: float = 1.5
    mouth_seconds: float = 2.0
    focus_seconds: float = 2.0
    max_gap_seconds: float = 0.5
    max_people: int = 2
    provenance: str = "experimental_user_configuration"

    def __post_init__(self):
        for name in (
            "ear_threshold",
            "mar_threshold",
            "closure_seconds",
            "mouth_seconds",
            "focus_seconds",
            "max_gap_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.ear_threshold >= 1 or self.mar_threshold >= 2:
            raise ValueError("Ratio threshold outside supported range")
        if type(self.max_people) is not int or not 1 <= self.max_people <= 6:
            raise ValueError("max_people must be an integer between 1 and 6")
        if self.provenance != "experimental_user_configuration":
            raise ValueError("Unsupported threshold provenance")

    def to_dict(self):
        return asdict(self)
