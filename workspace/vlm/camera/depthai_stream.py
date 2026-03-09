from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Tuple

import cv2
import depthai as dai
import numpy as np

from vlm.selection.keyframe_detector import KeyframeConfig, SceneChangeKeyframeDetector
from vlm.selection.ring_buffer import FramePacket, RingBuffer
from vlm.selection.selector import KeyframeStore
from vlm.storage.video_writer import blend_overlay, depth_to_colormap
from vlm.navigation.depth_features import DepthFeatureConfig, DepthFeatureSmoother, depth_mm_to_roi_cm
from vlm.navigation.shared_state import DepthROISnapshot, DepthSharedState


@dataclass(frozen=True)
class CameraRuntime:
    stop_event: threading.Event
    thread: threading.Thread


def _color_res_from_str(name: str) -> dai.ColorCameraProperties.SensorResolution:
    return getattr(dai.ColorCameraProperties.SensorResolution, name)


def _median_from_str(name: str) -> dai.MedianFilter:
    return getattr(dai.MedianFilter, name)


def _encode_jpeg(frame_bgr: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return bytes(buf)


def start_camera_stream(
    *,
    cfg: Dict[str, Any],
    ring: RingBuffer,
    keyframes: KeyframeStore,
    video_writers: Optional[Any],
    depth_state: Optional[DepthSharedState] = None,
) -> CameraRuntime:
    stop_event = threading.Event()

    cam_cfg = cfg["camera"]
    rgb_cfg = cam_cfg["rgb"]
    depth_cfg = cam_cfg["depth"]

    rb_cfg = cfg["ring_buffer"]
    kf_cfg = cfg["keyframes"]
    storage_cfg = cfg["storage"]
    overlay_cfg = storage_cfg["depth_overlay"]
    nav_cfg = (cfg.get("navigation") or {}).get("depth_features") or {}
    roi_cfg = nav_cfg.get("roi") or {}

    keyframe_detector = SceneChangeKeyframeDetector(
        KeyframeConfig(
            downsample_size=int(kf_cfg["downsample_size"]),
            scene_change_threshold=float(kf_cfg["scene_change_threshold"]),
            min_keyframe_interval_sec=float(kf_cfg["min_keyframe_interval_sec"]),
        )
    )

    def _run() -> None:
        try:
            pipeline = dai.Pipeline()

            try:
                camRgb = pipeline.create(dai.node.ColorCamera)
                xoutRgb = pipeline.create(dai.node.XLinkOut)
            except AttributeError as e:
                print(f"\n[ERROR] DepthAI API 호환성 문제: {e}")
                print("       depthai 버전을 확인하거나 업데이트해주세요: pip install --upgrade depthai")
                return
            except Exception as e:
                print(f"\n[ERROR] 파이프라인 노드 생성 실패: {e}")
                return

            xoutRgb.setStreamName("rgb")

            w, h = rgb_cfg["preview_size"]
            camRgb.setPreviewSize(int(w), int(h))
            camRgb.setResolution(_color_res_from_str(rgb_cfg["sensor_resolution"]))
            camRgb.setInterleaved(bool(rgb_cfg["interleaved"]))
            camRgb.setColorOrder(getattr(dai.ColorCameraProperties.ColorOrder, rgb_cfg["color_order"]))
            camRgb.setFps(float(rgb_cfg["fps"]))

            camRgb.preview.link(xoutRgb.input)

            xoutDepth = None
            if bool(depth_cfg.get("enabled", True)):
                monoLeft = pipeline.create(dai.node.MonoCamera)
                monoRight = pipeline.create(dai.node.MonoCamera)
                stereo = pipeline.create(dai.node.StereoDepth)

                monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
                monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
                monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
                monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
                # Keep mono FPS aligned-ish with RGB fps to reduce temporal mismatch
                try:
                    monoLeft.setFps(float(rgb_cfg["fps"]))
                    monoRight.setFps(float(rgb_cfg["fps"]))
                except Exception:
                    pass

                monoLeft.out.link(stereo.left)
                monoRight.out.link(stereo.right)

                stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
                stereo.initialConfig.setConfidenceThreshold(int(depth_cfg["confidence_threshold"]))
                stereo.initialConfig.setMedianFilter(_median_from_str(depth_cfg["median_filter"]))
                stereo.setLeftRightCheck(bool(depth_cfg["lr_check"]))
                stereo.setExtendedDisparity(bool(depth_cfg["extended_disparity"]))
                stereo.setSubpixel(bool(depth_cfg["subpixel"]))
                stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
                # Match depth output size to RGB preview to avoid resize artifacts / striping
                try:
                    if hasattr(stereo, "setOutputSize"):
                        stereo.setOutputSize(int(w), int(h))
                except Exception:
                    pass

                xoutDepth = pipeline.create(dai.node.XLinkOut)
                xoutDepth.setStreamName("depth")
                stereo.depth.link(xoutDepth.input)

            try:
                device = dai.Device(pipeline)
            except RuntimeError as e:
                error_msg = str(e)
                if "No available devices" in error_msg or "no devices found" in error_msg.lower():
                    print(f"\n[ERROR] OAK-D 카메라를 찾을 수 없습니다.")
                    print("       다음을 확인해주세요:")
                    print("       1. OAK-D 카메라가 USB로 연결되어 있는지 확인")
                    print("       2. USB 케이블이 데이터 전송을 지원하는지 확인")
                    print("       3. 다른 프로그램이 카메라를 사용 중이 아닌지 확인")
                    print(f"       원본 에러: {error_msg}")
                else:
                    print(f"\n[ERROR] 카메라 초기화 실패: {error_msg}")
                return
            except Exception as e:
                print(f"\n[ERROR] 카메라 연결 중 예상치 못한 오류: {e}")
                return

            with device:
                qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
                qDepth = None
                if xoutDepth is not None:
                    qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

                frame_idx = 0
                # Buffer recent depth frames and match by timestamp to reduce temporal mismatch
                depth_buf: Deque[Tuple[float, np.ndarray]] = deque(maxlen=16)
                # If closest depth is older/newer than this, skip overlay for that RGB frame
                depth_match_max_dt_sec = 0.10

                def _frame_ts(pkt: Any) -> float:
                    # Prefer device timestamp when available (DepthAI ImgFrame)
                    try:
                        ts = pkt.getTimestamp()  # datetime.timedelta
                        return float(ts.total_seconds())
                    except Exception:
                        pass
                    try:
                        ts = pkt.getTimestampDevice()
                        return float(ts.total_seconds())
                    except Exception:
                        pass
                    return time.time()

                def _pick_best_depth(ts_rgb: float) -> Optional[np.ndarray]:
                    if not depth_buf:
                        return None
                    # Find closest by absolute time difference
                    best = None
                    best_dt = 1e9
                    for ts_d, d in depth_buf:
                        dt = abs(ts_rgb - ts_d)
                        if dt < best_dt:
                            best_dt = dt
                            best = d
                    if best is None or best_dt > depth_match_max_dt_sec:
                        return None
                    return best

                def _latest_depth() -> Optional[np.ndarray]:
                    if not depth_buf:
                        return None
                    return depth_buf[-1][1]

                smoother = DepthFeatureSmoother(ema_alpha=float(nav_cfg.get("ema_alpha", 0.7)))
                feat_cfg = DepthFeatureConfig(
                    left_roi=tuple(roi_cfg.get("left", [0.0, 0.33, 0.2, 0.8])),
                    center_roi=tuple(roi_cfg.get("center", [0.33, 0.67, 0.2, 0.8])),
                    right_roi=tuple(roi_cfg.get("right", [0.67, 1.0, 0.2, 0.8])),
                    min_mm=int(overlay_cfg.get("min_mm", 200)),
                    max_mm=int(overlay_cfg.get("max_mm", 10000)),
                    statistic=str(nav_cfg.get("statistic", "p10")),
                    ema_alpha=float(nav_cfg.get("ema_alpha", 0.7)),
                    min_valid_ratio=float(nav_cfg.get("min_valid_ratio", 0.01)),
                )

                while not stop_event.is_set():
                    inRgb = qRgb.tryGet()
                    inDepth = qDepth.tryGet() if qDepth is not None else None

                    if inDepth is not None:
                        # depth in mm (uint16)
                        depth_mm = inDepth.getFrame()
                        depth_buf.append((_frame_ts(inDepth), depth_mm))

                        if depth_state is not None:
                            try:
                                # Ensure depth is aligned to RGB preview size if needed
                                if depth_mm is not None and "w" in locals() and "h" in locals():
                                    if depth_mm.shape[:2] != (int(h), int(w)):
                                        depth_mm_for_feat = cv2.resize(
                                            depth_mm,
                                            (int(w), int(h)),
                                            interpolation=cv2.INTER_NEAREST,
                                        )
                                    else:
                                        depth_mm_for_feat = depth_mm
                                else:
                                    depth_mm_for_feat = depth_mm

                                distances_cm, valid_ratio = depth_mm_to_roi_cm(depth_mm_for_feat, cfg=feat_cfg)
                                distances_cm = smoother.update(distances_cm)
                                snap = DepthROISnapshot(
                                    ts=time.time(),
                                    left_cm=distances_cm.get("left"),
                                    center_cm=distances_cm.get("center"),
                                    right_cm=distances_cm.get("right"),
                                    valid_ratio=float(valid_ratio),
                                    width=int(depth_mm_for_feat.shape[1]),
                                    height=int(depth_mm_for_feat.shape[0]),
                                )
                                depth_state.update(snap)
                            except Exception:
                                # Never let feature computation break the camera pipeline.
                                pass

                    if inRgb is None:
                        time.sleep(0.001)
                        continue

                    rgb = inRgb.getCvFrame()
                    now = time.time()
                    frame_idx += 1
                    ts_rgb = _frame_ts(inRgb)

                    overlay_bgr = rgb

                    # Depth-only video should always have something when depth is available.
                    depth_for_video = _latest_depth()
                    depth_bgr: Optional[np.ndarray] = None
                    if depth_for_video is not None and bool(storage_cfg.get("save_depth_mp4", True)):
                        if depth_for_video.shape[:2] != rgb.shape[:2]:
                            depth_for_video = cv2.resize(
                                depth_for_video,
                                (rgb.shape[1], rgb.shape[0]),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        try:
                            depth_for_video = cv2.medianBlur(depth_for_video, 5)
                        except Exception:
                            pass
                        min_mm = int(overlay_cfg["min_mm"])
                        max_mm = int(overlay_cfg["max_mm"])
                        valid_v = (depth_for_video > 0) & (depth_for_video >= min_mm) & (depth_for_video <= max_mm)
                        depth_for_color_v = depth_for_video.copy()
                        depth_for_color_v[~valid_v] = min_mm
                        depth_bgr = depth_to_colormap(
                            depth_for_color_v,
                            min_mm=min_mm,
                            max_mm=max_mm,
                            colormap_name=str(overlay_cfg["colormap"]),
                        )
                        # Make invalid area black for cleaner depth video
                        depth_bgr[~valid_v] = 0

                    if bool(storage_cfg.get("save_depth_overlay_mp4", True)):
                        # Prefer matched depth for overlay; fallback to latest to avoid "overlay disappears"
                        best_depth = _pick_best_depth(ts_rgb)
                        depth_mm = best_depth if best_depth is not None else _latest_depth()

                        if depth_mm is not None and depth_mm.shape[:2] != rgb.shape[:2]:
                            # Prefer nearest to keep raw depth discrete, but we will mask invalid pixels below
                            depth_mm = cv2.resize(
                                depth_mm,
                                (rgb.shape[1], rgb.shape[0]),
                                interpolation=cv2.INTER_NEAREST,
                            )

                        if depth_mm is not None:
                            # Smooth depth ONLY for visualization to reduce banding/striping.
                            # Keep it lightweight; fallback silently if unsupported.
                            try:
                                depth_mm = cv2.medianBlur(depth_mm, 5)
                            except Exception:
                                pass

                            min_mm = int(overlay_cfg["min_mm"])
                            max_mm = int(overlay_cfg["max_mm"])
                            alpha = float(overlay_cfg["alpha"])

                            # Mask invalid depth to keep edges clean (no speckle/stripe in invalid areas)
                            valid = (depth_mm > 0) & (depth_mm >= min_mm) & (depth_mm <= max_mm)

                            # For colormap computation, clamp invalid to min_mm (will be removed by mask anyway)
                            depth_for_color = depth_mm.copy()
                            depth_for_color[~valid] = min_mm

                            depth_color = depth_to_colormap(
                                depth_for_color,
                                min_mm=min_mm,
                                max_mm=max_mm,
                                colormap_name=str(overlay_cfg["colormap"]),
                            )
                            blended = blend_overlay(rgb, depth_color, alpha)
                            valid3 = valid[:, :, None]
                            overlay_bgr = np.where(valid3, blended, rgb)

                    # Write videos (rate-limited inside writers)
                    if video_writers is not None:
                        # Write both on the same tick to keep rgb/depth_overlay in sync
                        if hasattr(video_writers, "maybe_write_pair"):
                            video_writers.maybe_write_pair(
                                rgb_bgr=rgb if bool(storage_cfg.get("save_rgb_mp4", True)) else None,
                                depth_overlay_bgr=overlay_bgr
                                if bool(storage_cfg.get("save_depth_overlay_mp4", True))
                                else None,
                                depth_bgr=depth_bgr if bool(storage_cfg.get("save_depth_mp4", True)) else None,
                            )
                        else:
                            # Fallback
                            if bool(storage_cfg.get("save_rgb_mp4", True)):
                                video_writers.maybe_write_rgb(rgb)
                            if bool(storage_cfg.get("save_depth_overlay_mp4", True)):
                                video_writers.maybe_write_depth_overlay(overlay_bgr)
                            if bool(storage_cfg.get("save_depth_mp4", True)) and depth_bgr is not None:
                                video_writers.maybe_write_depth(depth_bgr)

                    # Choose which image stream to buffer for VLM
                    img_source = str(cfg.get("vlm", {}).get("image_source", "depth_overlay"))
                    to_buffer = overlay_bgr if img_source == "depth_overlay" else rgb
                    jpeg = _encode_jpeg(to_buffer, int(rb_cfg["jpeg_quality"]))

                    pkt = FramePacket(
                        frame_id=f"f{frame_idx:08d}",
                        ts=now,
                        jpeg_bytes=jpeg,
                        width=int(to_buffer.shape[1]),
                        height=int(to_buffer.shape[0]),
                    )
                    ring.push(pkt)

                    is_kf, _score = keyframe_detector.is_keyframe(rgb, now=now)
                    if is_kf:
                        keyframes.add(pkt)
        except Exception as e:
            print(f"\n[ERROR] 카메라 스트림 실행 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()

    t = threading.Thread(target=_run, name="vlm-camera-stream", daemon=True)
    t.start()
    return CameraRuntime(stop_event=stop_event, thread=t)

