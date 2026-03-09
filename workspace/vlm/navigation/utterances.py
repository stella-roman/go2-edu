from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def round_cm(x: Optional[float], *, step: int = 5) -> Optional[int]:
    if x is None:
        return None
    step = max(1, int(step))
    return int(round(float(x) / step) * step)


@dataclass(frozen=True)
class FindUtterance:
    kind: str  # "near_warn" | "stop"
    direction: str  # "left" | "right" | "center" | "unknown"

    def to_text_en(self) -> str:
        if self.kind == "stop":
            return "Stop. Do not proceed further."
        # near_warn
        if self.direction in {"left", "right"}:
            return f"Object immediately ahead. Reach to your {self.direction}."
        if self.direction == "center":
            return "Object immediately ahead. Reach forward."
        return "Object immediately ahead. Reach carefully."


@dataclass(frozen=True)
class GuideUtterance:
    kind: str  # "door" | "obstacle"
    distance_cm: Optional[int]
    obstacle_type: str = "Obstacle"
    obstacle_side: str = "center"  # "left"|"center"|"right"
    change_course_to: str = "left"  # "left"|"right"

    def to_text_en(self) -> str:
        x = self.distance_cm
        if self.kind == "door":
            if x is None:
                return "There is a door ahead."
            return f"In {x} centimeters, there is a door."
        # obstacle
        dist_part = "ahead" if x is None else f"{x} centimeters ahead"
        obs = self.obstacle_type or "Obstacle"
        side = self.obstacle_side or "center"
        change = self.change_course_to or "left"
        return f"{obs} detected {dist_part} on the {side}. Change course to the {change}."

