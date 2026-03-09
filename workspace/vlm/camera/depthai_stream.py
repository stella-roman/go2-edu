from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import cv2
import depthai as dai
import numpy as np

from vlm.selection.keyframe_detector import KeyframeConfig, SceneChangeKeyframeDetector
from vlm.selection.ring_buffer import FramePacket, RingBuffer
from vlm.selection.selector import KeyframeStore
from vlm.storage.video_writer import blend_overlay, depth_to_colormap


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
) -> CameraRuntime:
    stop_event = threading.Event()

    cam_cfg = cfg["camera"]
    rgb_cfg = cam_cfg["rgb"]
    depth_cfg = cam_cfg["depth"]

    rb_cfg = cfg["ring_buffer"]
    kf_cfg = cfg["keyframes"]
    storage_cfg = cfg["storage"]
    overlay_cfg = storage_cfg["depth_overlay"]

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

                monoLeft.out.link(stereo.left)
                monoRight.out.link(stereo.right)

                stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
                stereo.initialConfig.setConfidenceThreshold(int(depth_cfg["confidence_threshold"]))
                stereo.initialConfig.setMedianFilter(_median_from_str(depth_cfg["median_filter"]))
                stereo.setLeftRightCheck(bool(depth_cfg["lr_check"]))
                stereo.setExtendedDisparity(bool(depth_cfg["extended_disparity"]))
                stereo.setSubpixel(bool(depth_cfg["subpixel"]))
                stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

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
                last_depth: Optional[np.ndarray] = None

                while not stop_event.is_set():
                    inRgb = qRgb.tryGet()
                    inDepth = qDepth.tryGet() if qDepth is not None else None

                    if inDepth is not None:
                        # depth in mm (uint16)
                        last_depth = inDepth.getFrame()

                    if inRgb is None:
                        time.sleep(0.001)
                        continue

                    rgb = inRgb.getCvFrame()
                    now = time.time()
                    frame_idx += 1

                    overlay_bgr = rgb
                    if last_depth is not None and bool(storage_cfg.get("save_depth_overlay_mp4", True)):
                        depth_mm = last_depth
                        if depth_mm.shape[:2] != rgb.shape[:2]:
                            depth_mm = cv2.resize(depth_mm, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                        depth_color = depth_to_colormap(
                            depth_mm,
                            min_mm=int(overlay_cfg["min_mm"]),
                            max_mm=int(overlay_cfg["max_mm"]),
                            colormap_name=str(overlay_cfg["colormap"]),
                        )
                        overlay_bgr = blend_overlay(rgb, depth_color, float(overlay_cfg["alpha"]))

                    # Write videos (rate-limited inside writers)
                    if video_writers is not None:
                        if bool(storage_cfg.get("save_rgb_mp4", True)):
                            video_writers.maybe_write_rgb(rgb)
                        if bool(storage_cfg.get("save_depth_overlay_mp4", True)):
                            video_writers.maybe_write_depth_overlay(overlay_bgr)

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

