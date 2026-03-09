from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class NavMode(str, Enum):
    idle = "idle"
    find = "find"
    guide = "guide"


@dataclass
class NavState:
    mode: NavMode = NavMode.idle
    find_target: Optional[str] = None

