from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from vlm.navigation.mode import NavMode, NavState
from vlm.navigation.shared_state import DepthSharedState
from vlm.navigation.utterances import FindUtterance, GuideUtterance, round_cm


def _argmin3(vals: Dict[str, Optional[float]]) -> Optional[str]:
    best_k = None
    best_v = None
    for k in ("left", "center", "right"):
        v = vals.get(k)
        if v is None:
            continue
        if best_v is None or v < best_v:
            best_v = v
            best_k = k
    return best_k


def _argmax2(left: Optional[float], right: Optional[float]) -> Optional[str]:
    if left is None and right is None:
        return None
    if left is None:
        return "right"
    if right is None:
        return "left"
    return "left" if left >= right else "right"


@dataclass
class GuidanceConfig:
    hz: float = 8.0
    cooldown_sec: float = 2.0
    required_consecutive: int = 3

    # Find thresholds
    find_near_warn_cm: float = 50.0
    find_stop_cm: float = 30.0

    # Guide thresholds
    obstacle_threshold_cm: float = 80.0
    door_announce_cm: float = 30.0
    door_tolerance_cm: float = 10.0

    # label refresh
    label_refresh_sec: float = 1.0

    # rounding
    distance_round_step_cm: int = 5

    # Find coarse guidance (not just near/stop)
    find_guidance_cooldown_sec: float = 3.0

    # Find "moving away" detection
    find_moving_away_delta_cm: float = 15.0
    find_moving_away_window_sec: float = 2.0
    find_moving_away_cooldown_sec: float = 6.0


class GuidanceLoop:
    """
    DepthSharedState를 주기적으로 읽어 Find/Guide 이벤트를 생성하고,
    출력 콜백으로 영어 문장을 전달한다.
    (LLM 호출은 여기서 직접 하지 않고, 외부에서 label 캐시를 주입해도 됨)
    """

    def __init__(
        self,
        *,
        depth_state: DepthSharedState,
        nav_state: NavState,
        cfg: GuidanceConfig,
        get_find_label: Callable[[], Tuple[bool, str, str, bool]],
        get_door_and_obstacle: Callable[[], Tuple[bool, str]],
        on_utterance: Callable[[str], None],
    ) -> None:
        self.depth_state = depth_state
        self.nav_state = nav_state
        self.cfg = cfg
        self.get_find_label = get_find_label
        self.get_door_and_obstacle = get_door_and_obstacle
        self.on_utterance = on_utterance

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="vlm-guidance-loop", daemon=True)

        self._last_emit_ts: Dict[str, float] = {}
        self._consec: Dict[str, int] = {}
        self._find_prev_dist_cm: Optional[float] = None
        self._find_prev_ts: float = 0.0

    def start(self) -> None:
        self._thread.start()

    def stop(self, *, timeout: float = 2.0) -> None:
        self._stop_event.set()
        self._thread.join(timeout=timeout)

    def _cooldown_ok(self, key: str, now: float) -> bool:
        last = self._last_emit_ts.get(key, 0.0)
        return (now - last) >= float(self.cfg.cooldown_sec)

    def _cooldown_ok_custom(self, key: str, now: float, cooldown: float) -> bool:
        last = self._last_emit_ts.get(key, 0.0)
        return (now - last) >= float(cooldown)

    def _hit(self, key: str, cond: bool) -> bool:
        if not cond:
            self._consec[key] = 0
            return False
        self._consec[key] = int(self._consec.get(key, 0) + 1)
        return self._consec[key] >= int(self.cfg.required_consecutive)

    def _emit(self, key: str, text: str, now: float) -> None:
        self._last_emit_ts[key] = now
        self.on_utterance(text)

    def _run(self) -> None:
        period = 1.0 / max(1e-6, float(self.cfg.hz))
        while not self._stop_event.is_set():
            t0 = time.time()
            snap = self.depth_state.latest()
            if snap is not None and self.nav_state.mode != NavMode.idle:
                left = snap.left_cm
                center = snap.center_cm
                right = snap.right_cm
                now = time.time()

                if self.nav_state.mode == NavMode.find:
                    visible, direction, target, should_search = self.get_find_label()
                    # choose depth by direction; fallback to center
                    dist = {
                        "left": left,
                        "center": center,
                        "right": right,
                    }.get(direction, None)
                    if dist is None:
                        dist = center

                    # Detect if distance is increasing (moving away)
                    away_cond = False
                    if dist is not None:
                        prev = self._find_prev_dist_cm
                        prev_ts = float(self._find_prev_ts)
                        window = float(self.cfg.find_moving_away_window_sec)
                        delta = float(self.cfg.find_moving_away_delta_cm)
                        if prev is not None and (now - prev_ts) <= window:
                            away_cond = (float(dist) - float(prev)) >= delta
                        self._find_prev_dist_cm = float(dist)
                        self._find_prev_ts = now
                    else:
                        self._find_prev_dist_cm = None
                        self._find_prev_ts = now

                    stop_cond = dist is not None and dist <= float(self.cfg.find_stop_cm)
                    warn_cond = dist is not None and dist <= float(self.cfg.find_near_warn_cm)

                    if self._hit("find_stop", stop_cond) and self._cooldown_ok("find_stop", now):
                        utt = FindUtterance(kind="stop", direction=direction).to_text_en()
                        self._emit("find_stop", utt, now)
                    elif self._hit("find_warn", warn_cond) and self._cooldown_ok("find_warn", now):
                        utt = FindUtterance(kind="near_warn", direction=direction).to_text_en()
                        self._emit("find_warn", utt, now)
                    elif (
                        self._hit("find_away", away_cond)
                        and self._cooldown_ok_custom("find_away", now, float(self.cfg.find_moving_away_cooldown_sec))
                    ):
                        # Keep this conservative and safety-oriented.
                        if direction in {"left", "right"}:
                            text = f"You are moving farther from {target}. Stop and turn slightly {direction}."
                        elif direction == "center":
                            text = f"You are moving farther from {target}. Stop and move forward slowly."
                        else:
                            text = f"You are moving farther from {target}. Stop and reorient."
                        self._emit("find_away", text, now)
                    else:
                        # Coarse guidance so the user can keep progressing even when far.
                        # Only emit when the target is visible; otherwise ask to scan.
                        if visible:
                            if direction in {"left", "right"}:
                                text = f"{target} is to your {direction}. Turn slightly {direction} and move forward slowly."
                            elif direction == "center":
                                text = f"{target} is ahead. Move forward slowly."
                            else:
                                text = f"{target} is ahead. Move slowly and keep the camera steady."
                        else:
                            # Searching is intentionally strict (e.g., require consecutive false labels).
                            if should_search:
                                text = f"Searching for {target}. Slowly pan left and right."
                            else:
                                if direction in {"left", "right"}:
                                    text = f"{target} is to your {direction}. Turn slightly {direction} and move forward slowly."
                                elif direction == "center":
                                    text = f"{target} is ahead. Move forward slowly."
                                else:
                                    text = f"Move slowly and keep the camera steady."

                        if self._cooldown_ok_custom("find_coarse", now, float(self.cfg.find_guidance_cooldown_sec)):
                            self._emit("find_coarse", text, now)

                elif self.nav_state.mode == NavMode.guide:
                    door_ahead, obstacle_type = self.get_door_and_obstacle()
                    vals = {"left": left, "center": center, "right": right}
                    min_side = _argmin3(vals)
                    ahead = center

                    # Door announce near a specific distance band
                    if door_ahead and ahead is not None:
                        near = abs(float(ahead) - float(self.cfg.door_announce_cm)) <= float(
                            self.cfg.door_tolerance_cm
                        )
                        if self._hit("guide_door", near) and self._cooldown_ok("guide_door", now):
                            d = round_cm(ahead, step=int(self.cfg.distance_round_step_cm))
                            utt = GuideUtterance(kind="door", distance_cm=d).to_text_en()
                            self._emit("guide_door", utt, now)

                    # Obstacle detection (depth-only)
                    obstacle_cond = ahead is not None and ahead <= float(self.cfg.obstacle_threshold_cm)
                    if self._hit("guide_obstacle", obstacle_cond) and self._cooldown_ok("guide_obstacle", now):
                        change_to = _argmax2(left, right) or "left"
                        d = round_cm(ahead, step=int(self.cfg.distance_round_step_cm))
                        side = min_side or "center"
                        obs_name = obstacle_type.strip() if obstacle_type else "Obstacle"
                        if obs_name.lower() == "unknown":
                            obs_name = "Obstacle"
                        utt = GuideUtterance(
                            kind="obstacle",
                            distance_cm=d,
                            obstacle_type=obs_name,
                            obstacle_side=side,
                            change_course_to=change_to,
                        ).to_text_en()
                        self._emit("guide_obstacle", utt, now)

            dt = time.time() - t0
            sleep_s = max(0.0, period - dt)
            time.sleep(sleep_s)

