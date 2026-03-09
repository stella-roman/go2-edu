from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Session:
    max_turns: int = 20
    history: List[Dict[str, str]] = field(default_factory=list)

    def add_user(self, text: str) -> None:
        self.history.append({"role": "user", "content": text})
        self._trim()

    def add_assistant(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})
        self._trim()

    def _trim(self) -> None:
        # Keep last N turns (= 2*N messages)
        if self.max_turns <= 0:
            return
        max_msgs = self.max_turns * 2
        if len(self.history) > max_msgs:
            self.history = self.history[-max_msgs:]

