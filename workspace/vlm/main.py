from __future__ import annotations

# IMPORTANT: make `vlm.*` imports work when running as a script:
#   python agent-stella/vlm/main.py --config agent-stella/config/default.yaml
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]  # agent-stella/
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import base64
from typing import Any, Dict, List

from vlm.camera.depthai_stream import start_camera_stream
from vlm.config import dump_effective_config, load_config, parse_args
from vlm.providers.base import ImageInput
from vlm.providers.openai_compatible import OpenAICompatibleProvider
from vlm.selection.ring_buffer import RingBuffer
from vlm.selection.selector import FrameSelector, KeyframeStore, SelectionConfig
from vlm.session import Session
from vlm.storage.interaction_logger import InteractionLogger
from vlm.storage.run_dir import create_run_dirs
from vlm.storage.video_writer import VideoWriters


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


def main() -> None:
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
    frame_size = tuple(map(int, cfg["camera"]["rgb"]["preview_size"]))
    writers = VideoWriters(
        rgb_path=rgb_path,
        depth_overlay_path=depth_path,
        fps=int(cfg["storage"]["fps"]),
        frame_size=(frame_size[0], frame_size[1]),
    )
    writers.open()

    # Start camera stream (background)
    cam_rt = start_camera_stream(cfg=cfg, ring=ring, keyframes=keyframes, video_writers=writers)

    logger = InteractionLogger(run_dirs.logs_dir / "interaction.jsonl")
    session = Session(max_turns=20)

    provider = _make_provider(cfg)
    system_prompt = str(cfg["vlm"]["system_prompt"])

    print("\n=== agent-stella VLM interactive ===")
    print(f"- output: {run_dirs.run_dir}")
    print("- type 'quit' to exit")

    try:
        while True:
            user_text = input("\nYou: ").strip()
            if not user_text:
                continue
            if user_text.lower() in {"quit", "exit"}:
                break

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

            # Call provider with session history (text only) + selected images
            result = provider.generate(
                system_prompt=system_prompt,
                history=session.history,
                user_text=user_text,
                images=images,
            )

            answer = result.text.strip()
            print(f"\nSTELLA: {answer}")

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
            )

    finally:
        cam_rt.stop_event.set()
        cam_rt.thread.join(timeout=2.0)
        writers.close()


if __name__ == "__main__":
    main()

