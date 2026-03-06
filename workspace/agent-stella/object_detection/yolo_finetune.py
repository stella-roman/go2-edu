from ultralytics import YOLO
import torch

if __name__ == "__main__":
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()

    model = YOLO("yolo11n.pt")

    model.train(
        data="emergency_10.yaml",
        epochs=100,
        imgsz=640,
        batch=64,
        device=0,
        workers=4,
        freeze=0,
        patience=30,
        cache=True,
        amp=True,
        close_mosaic=10
    )
