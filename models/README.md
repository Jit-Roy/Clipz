# Models Directory

This directory should contain the YOLO model file:
- `yolov8n.pt` - YOLOv8 nano model for object detection

## Download YOLO Model

The model will be automatically downloaded when you first run the video analysis if the `ultralytics` package is installed. Alternatively, you can manually download it:

```bash
# Using ultralytics CLI
yolo export model=yolov8n.pt format=pytorch

# Or download from Ultralytics
# https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

Place the downloaded `yolov8n.pt` file in this directory.
