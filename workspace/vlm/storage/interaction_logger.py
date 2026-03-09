from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class InteractionLogger:
    path: Path

    def log_turn(
        self,
        *,
        user_text: str,
        assistant_text: str,
        used_frame_ids: List[str],
        provider: str,
        model: str,
        usage: Optional[Dict[str, Any]] = None,
        inference_time: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        record: Dict[str, Any] = {
            "ts": time.time(),
            "user": user_text,
            "assistant": assistant_text,
            "used_frame_ids": used_frame_ids,
            "provider": provider,
            "model": model,
        }
        if usage:
            record["usage"] = usage
            
        if inference_time is not None:
            record["inference_time"] = inference_time

        if extra:
            record["extra"] = extra
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

