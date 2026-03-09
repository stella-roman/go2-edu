#!/usr/bin/env python3
"""
간단한 카메라 뷰어 - OAK-D 카메라 연결 상태 및 화면 확인용
'q' 키를 누르면 종료됩니다.
"""

from __future__ import annotations

import cv2
import depthai as dai
import numpy as np


def main() -> None:
    print("OAK-D 카메라 연결 확인 중...")
    
    # 파이프라인 생성
    pipeline = dai.Pipeline()
    
    # RGB 카메라 노드
    camRgb = pipeline.create(dai.node.ColorCamera)
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")
    
    # RGB 설정
    camRgb.setPreviewSize(640, 360)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    camRgb.setFps(40)
    
    camRgb.preview.link(xoutRgb.input)
    
    # Depth 카메라 노드
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    xoutDepth = pipeline.create(dai.node.XLinkOut)
    
    xoutDepth.setStreamName("depth")
    
    # Mono 카메라 설정
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    
    # Stereo 설정
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setConfidenceThreshold(240)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(False)
    stereo.setSubpixel(False)
    
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    stereo.depth.link(xoutDepth.input)
    
    try:
        device = dai.Device(pipeline)
        print("✓ 카메라 연결 성공!")
        print("'q' 키를 누르면 종료됩니다.\n")
    except RuntimeError as e:
        error_msg = str(e)
        if "No available devices" in error_msg or "no devices found" in error_msg.lower():
            print("✗ OAK-D 카메라를 찾을 수 없습니다.")
            print("  다음을 확인해주세요:")
            print("  1. OAK-D 카메라가 USB로 연결되어 있는지 확인")
            print("  2. USB 케이블이 데이터 전송을 지원하는지 확인")
            print("  3. 다른 프로그램이 카메라를 사용 중이 아닌지 확인")
        else:
            print(f"✗ 카메라 초기화 실패: {error_msg}")
        return
    except Exception as e:
        print(f"✗ 카메라 연결 중 오류: {e}")
        return
    
    with device:
        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        
        frame_count = 0
        start_time = cv2.getTickCount()
        
        while True:
            inRgb = qRgb.tryGet()
            inDepth = qDepth.tryGet()
            
            if inRgb is not None:
                frame_rgb = inRgb.getCvFrame()
                
                # FPS 계산
                frame_count += 1
                elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
                if elapsed > 0:
                    fps = frame_count / elapsed
                else:
                    fps = 0
                
                # FPS 표시
                cv2.putText(
                    frame_rgb,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                
                cv2.imshow("RGB Camera", frame_rgb)
            
            if inDepth is not None:
                frame_depth = inDepth.getFrame()
                
                # Depth를 컬러맵으로 변환
                depth_8bit = (frame_depth / 256).astype(np.uint8)
                depth_colormap = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
                
                # Depth 범위 표시
                min_depth = np.min(frame_depth[frame_depth > 0]) if np.any(frame_depth > 0) else 0
                max_depth = np.max(frame_depth)
                cv2.putText(
                    depth_colormap,
                    f"Depth: {min_depth}-{max_depth}mm",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                
                cv2.imshow("Depth Camera", depth_colormap)
            
            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) == ord('q'):
                break
        
        print("\n카메라 뷰어를 종료합니다.")


if __name__ == "__main__":
    main()
