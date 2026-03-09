from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from vlm.providers.base import ImageInput
from vlm.providers.openai_compatible import OpenAICompatibleProvider


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    모델이 JSON 앞뒤로 텍스트/코드펜스를 붙이는 경우를 방어적으로 파싱.
    """
    if not text:
        return None
    text = text.strip()
    # If it's already a plain JSON object
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # Try to find the first {...} block (naive but effective)
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


@dataclass(frozen=True)
class FindLabel:
    target_visible: bool
    target_direction: str  # left|center|right|unknown


@dataclass(frozen=True)
class GuideLabel:
    door_ahead: bool
    obstacle_type: str  # e.g. person/chair/unknown


FIND_LABELER_SYSTEM_PROMPT = """You are a perception labeler for an assistive navigation system.
You will be given images from a forward-facing camera.
Return ONLY a single JSON object. No prose. No markdown. No code fences.

Schema:
{
  "target_visible": boolean,
  "target_direction": "left" | "center" | "right" | "unknown"
}

Rules:
- target_direction refers to where the target object appears in the image (left/center/right).
- If you are not confident, use "unknown".
"""


GUIDE_LABELER_SYSTEM_PROMPT = """You are a perception labeler for an assistive navigation system.
You will be given images from a forward-facing camera.
Return ONLY a single JSON object. No prose. No markdown. No code fences.

Schema:
{
  "door_ahead": boolean,
  "obstacle_type": string
}

Rules:
- door_ahead: true only if you can clearly see a door ahead.
- obstacle_type: a short noun like "person", "chair", "table", "wall", "stairs", or "unknown".
"""


def label_find(
    *,
    provider: OpenAICompatibleProvider,
    target: str,
    images: List[ImageInput],
) -> FindLabel:
    user_text = f'Target to find: "{target}". Determine where it is (left/center/right).'
    result = provider.generate(
        system_prompt=FIND_LABELER_SYSTEM_PROMPT,
        history=[],
        user_text=user_text,
        images=images,
    )
    obj = _extract_first_json_object(result.text) or {}
    vis = bool(obj.get("target_visible", False))
    direction = str(obj.get("target_direction", "unknown")).lower().strip()
    if direction not in {"left", "center", "right", "unknown"}:
        direction = "unknown"
    return FindLabel(target_visible=vis, target_direction=direction)


def label_guide(
    *,
    provider: OpenAICompatibleProvider,
    images: List[ImageInput],
) -> GuideLabel:
    user_text = "Label if there is a door ahead and name the most relevant obstacle type."
    result = provider.generate(
        system_prompt=GUIDE_LABELER_SYSTEM_PROMPT,
        history=[],
        user_text=user_text,
        images=images,
    )
    obj = _extract_first_json_object(result.text) or {}
    door = bool(obj.get("door_ahead", False))
    obstacle_type = str(obj.get("obstacle_type", "unknown")).strip()
    if not obstacle_type:
        obstacle_type = "unknown"
    return GuideLabel(door_ahead=door, obstacle_type=obstacle_type)

