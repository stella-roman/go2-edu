from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class RunDirs:
    run_dir: Path
    video_dir: Path
    logs_dir: Path


def create_run_dirs(output_root: str | Path) -> RunDirs:
    root = Path(output_root)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / ts
    video_dir = run_dir / "video"
    logs_dir = run_dir / "logs"
    video_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return RunDirs(run_dir=run_dir, video_dir=video_dir, logs_dir=logs_dir)

