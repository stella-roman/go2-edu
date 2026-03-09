from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .base import ImageInput, VLMResult


@dataclass
class OpenAICompatibleProvider:
    name: str
    model: str
    api_key: str
    base_url: str = ""

    def __post_init__(self) -> None:
        if not self.api_key:
            # Allow env fallback
            self.api_key = self.api_key or os.getenv("OPENAI_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
        kwargs: Dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)

    def generate(
        self,
        *,
        system_prompt: str,
        history: List[Dict[str, str]],
        user_text: str,
        images: List[ImageInput],
    ) -> VLMResult:
        # history: [{"role":"user"|"assistant","content":"..."}]
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        messages.extend(history)

        user_content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
        for img in images:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{img.mime_type};base64,{img.data_base64}",
                        "detail": img.detail,
                    },
                }
            )

        messages.append({"role": "user", "content": user_content})

        resp = self._client.chat.completions.create(model=self.model, messages=messages)
        msg = resp.choices[0].message.content or ""
        usage: Optional[Dict[str, Any]] = None
        try:
            if resp.usage is not None:
                usage = {
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                    "total_tokens": resp.usage.total_tokens,
                }
        except Exception:
            usage = None
        return VLMResult(text=msg, usage=usage)

