from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol


@dataclass(frozen=True)
class ImageInput:
    mime_type: str
    data_base64: str
    detail: str = "low"


@dataclass(frozen=True)
class VLMResult:
    text: str
    usage: Optional[Dict[str, Any]] = None
    inference_time: Optional[float] = None


class VLMProvider(Protocol):
    name: str
    model: str

    def generate(
        self,
        *,
        system_prompt: str,
        history: List[Dict[str, str]],
        user_text: str,
        images: List[ImageInput],
    ) -> VLMResult: ...

