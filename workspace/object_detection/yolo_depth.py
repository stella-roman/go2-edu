import cv2
import sys
import depthai as dai
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from agent.tools.speech_generation import play_wav, print_and_speak

nnPath = str((Path(__file__).parent / Path('models/best.blob')).resolve().absolute())

labelMap = [
    "person",   "bicycle",      "car",          "motorcycle",
    "train",    "stiar_up",     "stair_down",   "kickboard"
]

pipeline = dai.Pipeline()

# --- Color Camera ---
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setPreviewSize(640, 640)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)

# --- Mono cameras ---
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)

monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

# --- Stereo Depth ---
stereo = pipeline.create(dai.node.StereoDepth)
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(True)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

# --- Spatial YOLO ---
spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
spatialDetectionNetwork.setBlobPath(nnPath)
spatialDetectionNetwork.setConfidenceThreshold(0.5)
spatialDetectionNetwork.setNumClasses(10)
spatialDetectionNetwork.setCoordinateSize(4)
spatialDetectionNetwork.setIouThreshold(0.5)

spatialDetectionNetwork.input.setBlocking(False)

stereo.depth.link(spatialDetectionNetwork.inputDepth)
camRgb.preview.link(spatialDetectionNetwork.input)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
spatialDetectionNetwork.passthrough.link(xoutRgb.input)

xoutNN = pipeline.create(dai.node.XLinkOut)
xoutNN.setStreamName("detections")
spatialDetectionNetwork.out.link(xoutNN.input)

# ---- Run ----
with dai.Device(pipeline) as device:
    start_time = time.time()
    qRgb = device.getOutputQueue("rgb", 4, False)
    qDet = device.getOutputQueue("detections", 4, False)

    last_print_time = start_time
    print("[LAST PRINT TIME]", last_print_time)
    best_detection = None
    best_confidence = 0
    best_label = None
    best_distance = 0

    while True:
        inRgb = qRgb.tryGet()

        if inRgb is None:
            continue

        frame = inRgb.getCvFrame()

        inDet = qDet.tryGet()

        if inDet is not None:
            for detection in inDet.detections:
                x1 = int(detection.xmin * frame.shape[1])
                y1 = int(detection.ymin * frame.shape[0])
                x2 = int(detection.xmax * frame.shape[1])
                y2 = int(detection.ymax * frame.shape[0])

                label = labelMap[detection.label]
                confidence = detection.confidence * 100

                distance = int(detection.spatialCoordinates.z)

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)

                cv2.putText(frame,
                            f"{label} {confidence:.1f}%",
                            (x1+10, y1+20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,(0,255,0),1)

                cv2.putText(frame,
                            f"Dist: {distance/1000:.2f} m",
                            (x1+10, y1+40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,(255,255,0),1)
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_label = label
                    best_distance = distance/1000
                    best_detection = f"{label} | {confidence:.1f}% | {distance/1000:.2f} m"

        current_time = time.time()
        print("[CURRENT TIME]", current_time)

        if current_time - last_print_time >= 1.0:
            if best_detection is not None:
                print(best_detection)
                temp_path = f"agent/tools/emergency_audios/temp.wav"
                print(f"STELLA: {best_label} detected.")
                print_and_speak(f"{best_distance:.1f} meters away.", temp_path)
                wav_path = f"agent/tools/emergency_audios/{best_label}.wav"
                play_wav(wav_path)
                play_wav(temp_path)
            else:
                print("Nothing Detected")

            best_detection = None
            best_confidence = 0
            last_print_time = current_time

        cv2.imshow("OAK-D Camera", frame)

        if cv2.waitKey(1) == ord('q'):
            break