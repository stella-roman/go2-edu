from __future__ import annotations

import argparse
import copy
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml
from dotenv import load_dotenv


ENV_PATTERN = re.compile(r"^\$\{ENV:([A-Z0-9_]+)(?::(.*))?\}$")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_env(value: Any) -> Any:
    if isinstance(value, str):
        m = ENV_PATTERN.match(value.strip())
        if not m:
            return value
        key = m.group(1)
        default = m.group(2)
        env_val = os.getenv(key)
        if env_val is None:
            return "" if default is None else default
        return env_val
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    return value


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _resolve_env(data)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="agent-stella VLM interactive runner")
    ap.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "config" / "default.yaml"),
        help="Path to YAML config (default: agent-stella/config/default.yaml)",
    )
    ap.add_argument("--provider", choices=["gemini", "openai"], default=None)
    ap.add_argument("--model", default=None, help="Override provider model")
    ap.add_argument("--fps", type=int, default=None, help="Override saved video fps (15/30)")
    ap.add_argument("--max-images", type=int, default=None, help="Override max images per turn")
    ap.add_argument("--no-rgb", action="store_true", help="Do not save rgb.mp4")
    ap.add_argument("--no-depth-overlay", action="store_true", help="Do not save depth_overlay.mp4")
    return ap.parse_args(argv)


def _args_to_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    ov: Dict[str, Any] = {}
    if args.fps is not None:
        ov = _deep_merge(ov, {"storage": {"fps": args.fps}})
    if args.max_images is not None:
        ov = _deep_merge(ov, {"selection": {"max_images_per_turn": args.max_images}})
    if args.no_rgb:
        ov = _deep_merge(ov, {"storage": {"save_rgb_mp4": False}})
    if args.no_depth_overlay:
        ov = _deep_merge(ov, {"storage": {"save_depth_overlay_mp4": False}})
    if args.provider is not None:
        ov = _deep_merge(ov, {"vlm": {"provider": args.provider}})
    if args.model is not None:
        provider = args.provider  # might be None; handled later
        ov = _deep_merge(
            ov,
            {
                "vlm": {
                    (provider or "gemini"): {"model": args.model},
                    (provider or "openai"): {"model": args.model},
                }
            },
        )
    return ov


@dataclass(frozen=True)
class LoadedConfig:
    config: Dict[str, Any]
    source_path: Path


def load_config(args: argparse.Namespace) -> LoadedConfig:
    # Load .env (repo uses this already)
    repo_root = Path(__file__).resolve().parents[1]  # agent-stella/
    load_dotenv(dotenv_path=repo_root / ".env", override=False)

    base = load_yaml_config(args.config)
    merged = _deep_merge(base, _args_to_overrides(args))

    # If YAML left api_key blank, try standard env key automatically
    provider = merged.get("vlm", {}).get("provider", "gemini")
    if provider == "gemini":
        merged.setdefault("vlm", {}).setdefault("gemini", {})
        if not merged["vlm"]["gemini"].get("api_key"):
            merged["vlm"]["gemini"]["api_key"] = os.getenv("GEMINI_API_KEY", "")
    elif provider == "openai":
        merged.setdefault("vlm", {}).setdefault("openai", {})
        if not merged["vlm"]["openai"].get("api_key"):
            merged["vlm"]["openai"]["api_key"] = os.getenv("OPENAI_API_KEY", "")

    return LoadedConfig(config=merged, source_path=Path(args.config).resolve())


def dump_effective_config(cfg: Dict[str, Any]) -> str:
    # Avoid leaking API keys in run_config.yaml
    scrubbed = copy.deepcopy(cfg)
    try:
        if "vlm" in scrubbed:
            for k in ("openai", "gemini"):
                if k in scrubbed["vlm"] and isinstance(scrubbed["vlm"][k], dict):
                    if "api_key" in scrubbed["vlm"][k]:
                        scrubbed["vlm"][k]["api_key"] = "***"
    except Exception:
        pass
    return yaml.safe_dump(scrubbed, sort_keys=False, allow_unicode=True)

