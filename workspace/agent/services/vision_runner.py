from __future__ import annotations

# IMPORTANT: make `vlm.*` imports work when running as a script:
#   python agent-stella/vlm/main.py --config agent-stella/config/default.yaml
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]  # agent-stella/
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import base64
import time
import threading
import re
from typing import Any, Dict, List

from vlm.camera.depthai_stream import start_camera_stream
from vlm.config import dump_effective_config, load_config, parse_args
from vlm.providers.base import ImageInput
from vlm.providers.openai_compatible import OpenAICompatibleProvider
from vlm.navigation.guidance_loop import GuidanceConfig, GuidanceLoop
from vlm.navigation.labeler import label_find, label_guide
from vlm.navigation.mode import NavMode, NavState
from vlm.navigation.shared_state import DepthSharedState
from vlm.selection.ring_buffer import RingBuffer
from vlm.selection.selector import FrameSelector, KeyframeStore, SelectionConfig
from vlm.session import Session
from vlm.storage.interaction_logger import InteractionLogger
from vlm.storage.run_dir import create_run_dirs
from vlm.storage.video_writer import VideoWriters
import rclpy
from rclpy.executors import MultiThreadedExecutor
from workspace.agent.tools.speech_generation import Go2Speaker


def _make_provider(cfg: Dict[str, Any]) -> OpenAICompatibleProvider:
    vlm_cfg = cfg["vlm"]
    provider = str(vlm_cfg.get("provider", "gemini"))
    if provider == "gemini":
        g = vlm_cfg["gemini"]
        return OpenAICompatibleProvider(
            name="gemini",
            model=str(g["model"]),
            api_key=str(g.get("api_key", "")),
            base_url=str(g.get("base_url", "")),
        )
    o = vlm_cfg["openai"]
    return OpenAICompatibleProvider(
        name="openai",
        model=str(o["model"]),
        api_key=str(o.get("api_key", "")),
        base_url=str(o.get("base_url", "")),
    )


def _clean_target(raw: str) -> str:
    s = (raw or "").strip().strip("\"'“”‘’")
    s = re.sub(r"^\s*(my|the|a|an)\s+", "", s, flags=re.IGNORECASE)
    # Cut off common trailing clauses
    cut_tokens = [
        " please",
        " pls",
        " and",
        " then",
        " explain",
        " how",
        ".",
        ",",
        ";",
        "!",
        "?",
    ]
    low = s.lower()
    cut_at = None
    for tok in cut_tokens:
        idx = low.find(tok)
        if idx != -1:
            cut_at = idx if cut_at is None else min(cut_at, idx)
    if cut_at is not None:
        s = s[:cut_at]
    return s.strip().strip("\"'“”‘’")


def _route_nav_intent(user_text: str) -> Dict[str, Any]:
    """
    Rule-based intent router.
    Returns: {"intent": "find"|"guide"|"idle"|"none", "target": str|None}
    """
    t = (user_text or "").strip()
    low = t.lower()

    # Stop/idle intent (Korean + English)
    if low in {"idle", "stop", "cancel", "end", "quit nav", "navigation off", "off"} or any(
        k in low for k in ["그만", "중지", "멈춰", "정지", "취소", "스탑"]
    ):
        return {"intent": "idle", "target": None}

    # Guide intent
    if low == "guide" or any(k in low for k in ["guide me", "navigate", "navigation", "안내", "가이드", "길 안내"]):
        return {"intent": "guide", "target": None}

    # Find intent (use word boundary to avoid "located", "finding" etc. in descriptive questions)
    if re.search(r"\bfind\b", low) or re.search(r"\blocate\b", low) or ("where is" in low) or any(k in low for k in ["찾아", "어디 있", "어딨", "위치"]):
        target = ""
        patterns = [
            r"\bfind\s+(?:my|the|a|an)\s+(.+)$",
            r"\bfind\s+(.+)$",
            r"\blocate\s+(?:my|the|a|an)\s+(.+)$",
            r"\blocate\s+(.+)$",
            r"\bwhere\s+is\s+(?:my|the|a|an)\s+(.+)$",
            r"\bwhere\s+is\s+(.+)$",
        ]
        for p in patterns:
            m = re.search(p, t, flags=re.IGNORECASE)
            if m:
                target = _clean_target(m.group(1))
                break
        if not target:
            # Korean: "<target> 찾아줘"
            m2 = re.search(r"(.+?)\s*(찾아줘|찾아|찾아봐|찾아줄래)\b", t)
            if m2:
                target = _clean_target(m2.group(1))
        return {"intent": "find", "target": target or None}

    return {"intent": "none", "target": None}


def _is_scene_query(user_text: str) -> bool:
    t = (user_text or "").strip()
    low = t.lower()
    # English
    if any(
        k in low
        for k in [
            "what do you see",
            "what do you see now",
            "describe",
            "describe the scene",
            "what is in front",
            "what's in front",
            "what is around",
            "what's around",
            "explain what you see",
        ]
    ):
        return True
    # Korean
    if any(k in low for k in ["지금 뭐", "뭐 보여", "뭐 보", "설명해", "설명해줘", "주변", "앞에 뭐", "앞에 뭐가"]):
        return True
    return False


def ask_vision(question) -> None:
    rclpy.init()

    speaker = Go2Speaker()

    executor = MultiThreadedExecutor()
    executor.add_node(speaker)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    args = parse_args()
    loaded = load_config(args)
    cfg = loaded.config

    run_dirs = create_run_dirs(cfg["app"]["output_root"])
    (run_dirs.run_dir / "run_config.yaml").write_text(dump_effective_config(cfg), encoding="utf-8")

    # Buffers / selection
    ring = RingBuffer(
        max_seconds=float(cfg["ring_buffer"]["max_seconds"]),
        max_frames=int(cfg["ring_buffer"]["max_frames"]),
    )
    keyframes = KeyframeStore(max_pinned_keyframes=int(cfg["keyframes"]["max_pinned_keyframes"]))
    selector = FrameSelector(
        ring=ring,
        keyframes=keyframes,
        cfg=SelectionConfig(
            max_images_per_turn=int(cfg["selection"]["max_images_per_turn"]),
            recent_window_sec=float(cfg["selection"]["recent_window_sec"]),
            recent_max_images=int(cfg["selection"]["recent_max_images"]),
            keyframe_window_sec=float(cfg["selection"]["keyframe_window_sec"]),
            keyframe_max_images=int(cfg["selection"]["keyframe_max_images"]),
        ),
    )

    # Video writers
    rgb_path = run_dirs.video_dir / "rgb.mp4" if bool(cfg["storage"]["save_rgb_mp4"]) else None
    depth_path = (
        run_dirs.video_dir / "depth_overlay.mp4"
        if bool(cfg["storage"]["save_depth_overlay_mp4"])
        else None
    )
    depth_only_path = (
        run_dirs.video_dir / "depth.mp4" if bool(cfg["storage"].get("save_depth_mp4", True)) else None
    )
    frame_size = tuple(map(int, cfg["camera"]["rgb"]["preview_size"]))
    writers = VideoWriters(
        rgb_path=rgb_path,
        depth_overlay_path=depth_path,
        depth_path=depth_only_path,
        fps=int(cfg["storage"]["fps"]),
        frame_size=(frame_size[0], frame_size[1]),
    )
    writers.open()

    depth_state = DepthSharedState()

    # Start camera stream (background)
    cam_rt = start_camera_stream(cfg=cfg, ring=ring, keyframes=keyframes, video_writers=writers, depth_state=depth_state)

    logger = InteractionLogger(run_dirs.logs_dir / "interaction.jsonl")
    session = Session(max_turns=20)

    provider = _make_provider(cfg)
    system_prompt = str(cfg["vlm"]["system_prompt"])
    scene_prompt = str((cfg.get("vlm") or {}).get("scene_prompt") or system_prompt)

    nav_state = NavState()

    # Label caches (updated by low-rate VLM calls from the main thread)
    label_cache: Dict[str, Any] = {
        "find_visible_raw": False,
        "find_last_seen_ts": 0.0,
        "find_false_streak": 0,
        "find_direction": "unknown",
        "find_target": None,
        "find_last_ts": 0.0,
        "door_ahead": False,
        "obstacle_type": "unknown",
        "guide_last_ts": 0.0,
    }

    nav_cfg = cfg.get("navigation") or {}
    gcfg_raw = nav_cfg.get("guidance") or {}
    guidance_cfg = GuidanceConfig(
        hz=float(gcfg_raw.get("hz", 8.0)),
        cooldown_sec=float(gcfg_raw.get("cooldown_sec", 2.0)),
        required_consecutive=int(gcfg_raw.get("required_consecutive", 3)),
        find_near_warn_cm=float((nav_cfg.get("find") or {}).get("near_warn_cm", 50.0)),
        find_stop_cm=float((nav_cfg.get("find") or {}).get("stop_cm", 30.0)),
        obstacle_threshold_cm=float((nav_cfg.get("guide") or {}).get("obstacle_threshold_cm", 80.0)),
        door_announce_cm=float((nav_cfg.get("guide") or {}).get("door_announce_cm", 30.0)),
        door_tolerance_cm=float((nav_cfg.get("guide") or {}).get("door_tolerance_cm", 10.0)),
        label_refresh_sec=float(gcfg_raw.get("label_refresh_sec", 1.0)),
        distance_round_step_cm=int(gcfg_raw.get("distance_round_step_cm", 5)),
        find_guidance_cooldown_sec=float(gcfg_raw.get("find_guidance_cooldown_sec", 3.0)),
        find_moving_away_delta_cm=float(gcfg_raw.get("find_moving_away_delta_cm", 15.0)),
        find_moving_away_window_sec=float(gcfg_raw.get("find_moving_away_window_sec", 2.0)),
        find_moving_away_cooldown_sec=float(gcfg_raw.get("find_moving_away_cooldown_sec", 6.0)),
    )
    find_visible_grace_sec = float(gcfg_raw.get("find_visible_grace_sec", 2.5))
    find_search_required_false_streak = int(gcfg_raw.get("find_search_required_false_streak", 3))

    def _get_find_label() -> Any:
        now = time.time()
        visible_raw = bool(label_cache.get("find_visible_raw", False))
        last_seen = float(label_cache.get("find_last_seen_ts", 0.0))
        visible = visible_raw or (last_seen > 0.0 and (now - last_seen) <= find_visible_grace_sec)
        direction = str(label_cache.get("find_direction", "unknown"))
        target = str(nav_state.find_target or label_cache.get("find_target") or "the target")
        false_streak = int(label_cache.get("find_false_streak", 0))
        should_search = (not visible) and (false_streak >= find_search_required_false_streak)
        return visible, direction, target, should_search

    def _get_door_and_obstacle() -> Any:
        return bool(label_cache.get("door_ahead", False)), str(label_cache.get("obstacle_type", "unknown"))

    def _on_utterance(text: str) -> None:
        # Guidance messages are intentionally short and English-only.
        speaker.speak(f"{text}")

    guidance = GuidanceLoop(
        depth_state=depth_state,
        nav_state=nav_state,
        cfg=guidance_cfg,
        get_find_label=_get_find_label,
        get_door_and_obstacle=_get_door_and_obstacle,
        on_utterance=_on_utterance,
    )
    guidance.start()

    # Background label refresh loop (low-rate VLM calls)
    label_stop = False

    def _uniform_sample(items: List[Any], k: int) -> List[Any]:
        if k <= 0 or not items:
            return []
        if len(items) <= k:
            return items
        idxs = [round(i * (len(items) - 1) / (k - 1)) for i in range(k)]
        out = []
        last = None
        for idx in idxs:
            if last == idx:
                continue
            out.append(items[int(idx)])
            last = idx
        return out[:k]

    def _select_recent_images() -> List[ImageInput]:
        # Use ring buffer only (thread-safe); avoid keyframes store (no lock).
        nav = cfg.get("navigation") or {}
        lcfg = nav.get("labeler") or {}
        window_sec = float(lcfg.get("recent_window_sec", 1.5))
        max_images = int(lcfg.get("max_images", 4))
        pkts = ring.within(window_sec)
        sel = _uniform_sample(pkts, max_images)
        out_imgs: List[ImageInput] = []
        for pkt in sel:
            out_imgs.append(
                ImageInput(
                    mime_type="image/jpeg",
                    data_base64=base64.b64encode(pkt.jpeg_bytes).decode("ascii"),
                    detail="low",
                )
            )
        return out_imgs

    def _label_loop() -> None:
        nonlocal label_stop
        while not label_stop:
            try:
                now = time.time()
                if nav_state.mode == NavMode.find and nav_state.find_target:
                    if (now - float(label_cache.get("find_last_ts", 0.0))) >= float(guidance_cfg.label_refresh_sec) or (
                        label_cache.get("find_target") != nav_state.find_target
                    ):
                        imgs = _select_recent_images()
                        if imgs:
                            fl = label_find(provider=provider, target=nav_state.find_target, images=imgs)
                            label_cache["find_visible_raw"] = bool(fl.target_visible)
                            if fl.target_visible:
                                label_cache["find_last_seen_ts"] = now
                                label_cache["find_false_streak"] = 0
                            else:
                                label_cache["find_false_streak"] = int(label_cache.get("find_false_streak", 0)) + 1
                            # Keep last known direction unless we get a confident one.
                            if fl.target_direction in {"left", "center", "right"}:
                                label_cache["find_direction"] = fl.target_direction
                            label_cache["find_last_ts"] = now
                            label_cache["find_target"] = nav_state.find_target
                elif nav_state.mode == NavMode.guide:
                    if (now - float(label_cache.get("guide_last_ts", 0.0))) >= float(guidance_cfg.label_refresh_sec):
                        imgs = _select_recent_images()
                        if imgs:
                            gl = label_guide(provider=provider, images=imgs)
                            label_cache["door_ahead"] = bool(gl.door_ahead)
                            label_cache["obstacle_type"] = str(gl.obstacle_type)
                            label_cache["guide_last_ts"] = now
            except Exception:
                pass
            time.sleep(0.05)

    label_thread = threading.Thread(target=_label_loop, name="vlm-label-loop", daemon=True)
    label_thread.start()

    print("\n=== agent-stella VLM interactive ===")
    print(f"- output: {run_dirs.run_dir}")
    print("- type 'quit' to exit")
    print("- commands: 'find <object>', 'guide', 'idle' (or natural language)")

    try:
        while True:
            user_text = question
            if not user_text:
                continue
            if user_text.lower() in {"quit", "exit"}:
                break

            # Simple intent routing for navigation modes
            low = user_text.strip().lower()
            if low == "idle":
                nav_state.mode = NavMode.idle
                nav_state.find_target = None
                speaker.speak("Navigation mode set to idle.")
                continue
            if low.startswith("find"):
                parts = user_text.split(maxsplit=1)
                target = parts[1].strip() if len(parts) > 1 else ""
                if not target:
                    speaker.speak("Please specify a target, e.g., 'find keys'.")
                    continue
                nav_state.mode = NavMode.find
                nav_state.find_target = target
                label_cache["find_target"] = target
                label_cache["find_last_ts"] = 0.0
                speaker.speak(f"Find mode enabled. Target = {target!r}")
                continue
            if low == "guide":
                nav_state.mode = NavMode.guide
                nav_state.find_target = None
                label_cache["guide_last_ts"] = 0.0
                speaker.speak("Guide mode enabled.")
                continue

            # Natural-language routing (A: rule-based router)
            routed = _route_nav_intent(user_text)
            if routed["intent"] == "idle":
                nav_state.mode = NavMode.idle
                nav_state.find_target = None
                speaker.speak("Navigation mode set to idle.")
                continue
            if routed["intent"] == "guide":
                nav_state.mode = NavMode.guide
                nav_state.find_target = None
                label_cache["guide_last_ts"] = 0.0
                speaker.speak("Guide mode enabled.")
                continue
            if routed["intent"] == "find":
                target = routed.get("target") or ""
                if not target:
                    speaker.speak("Please specify a target, e.g., 'find keys'.")
                    continue
                already = nav_state.mode == NavMode.find and (nav_state.find_target or "").lower() == target.lower()
                nav_state.mode = NavMode.find
                nav_state.find_target = target
                label_cache["find_target"] = target
                if not already:
                    label_cache["find_last_ts"] = 0.0
                    label_cache["find_visible_raw"] = False
                    label_cache["find_last_seen_ts"] = 0.0
                    label_cache["find_false_streak"] = 0
                    label_cache["find_direction"] = "unknown"
                    speaker.speak(f"Find mode enabled. Target = {target!r}")
                    speaker.speak(f"Searching for {target}. Slowly pan left and right.")
                else:
                    speaker.speak(f"Find mode is already active for {target!r}.")
                continue

            # While navigation mode is active, treat generic questions as navigation status.
            if nav_state.mode == NavMode.find and nav_state.find_target:
                vis, direction, target, should_search = _get_find_label()
                if should_search:
                    speaker.speak(f"Searching for {target}. Slowly pan left and right.")
                else:
                    if direction in {"left", "right"}:
                        speaker.speak(f"{target} is to your {direction}. Turn slightly {direction} and move forward slowly.")
                    elif direction == "center":
                        speaker.speak(f"{target} is ahead. Move forward slowly.")
                    else:
                        speaker.speak(f"{target} is ahead. Move slowly and keep the camera steady.")
                continue

            frames, frame_ids = selector.select()
            images: List[ImageInput] = []
            for pkt in frames:
                images.append(
                    ImageInput(
                        mime_type="image/jpeg",
                        data_base64=base64.b64encode(pkt.jpeg_bytes).decode("ascii"),
                        detail="low",
                    )
                )

            if not images:
                # Avoid calling the model with no images for vision questions.
                speaker.speak("No camera frames are available yet. Please wait a moment and try again.")
                logger.log_turn(
                    user_text=user_text,
                    assistant_text="No camera frames are available yet. Please wait a moment and try again.",
                    used_frame_ids=frame_ids,
                    provider=provider.name,
                    model=provider.model,
                    extra={"image_count": 0, "mode": str(nav_state.mode)},
                )
                continue

            chosen_system_prompt = scene_prompt if _is_scene_query(user_text) else system_prompt

            # Call provider with session history (text only) + selected images
            result = provider.generate(
                system_prompt=chosen_system_prompt,
                history=session.history,
                user_text=user_text,
                images=images,
            )

            answer = result.text.strip()
            speaker.speak(f"{answer}")

            # Update session (store text only)
            session.add_user(user_text)
            session.add_assistant(answer)

            logger.log_turn(
                user_text=user_text,
                assistant_text=answer,
                used_frame_ids=frame_ids,
                provider=provider.name,
                model=provider.model,
                usage=result.usage,
                inference_time=result.inference_time,
                extra={"image_count": len(images), "mode": str(nav_state.mode), "used_scene_prompt": bool(chosen_system_prompt == scene_prompt)},
            )

    finally:
        try:
            guidance.stop(timeout=2.0)
        except Exception:
            pass
        try:
            label_stop = True
            label_thread.join(timeout=2.0)
        except Exception:
            pass
        cam_rt.stop_event.set()
        cam_rt.thread.join(timeout=2.0)
        try:
            writers.close()
        except Exception as e:
            print(f"\n[WARNING] 비디오 파일 닫기 중 오류: {e}")

    speaker.destroy_node()
    rclpy.shutdown()