
import cv2
from ultralytics import YOLO
from pathlib import Path

video_path = "videos/ooms test video.mp4"

# Load YOLO model
model = YOLO("yolov8n.pt")

# Extract a frame at 55 seconds
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_MSEC, 55000)

success, frame = cap.read()
cap.release()

if not success:
    raise ValueError("Could not extract the video frame")

Path("output").mkdir(exist_ok=True)

# Test different confidence thresholds and image sizes
for confidence in [0.5, 0.25]:
    for image_size in [640, 960, 1280]:

        results = model.predict(
            source=frame,
            conf=confidence,
            imgsz=image_size,
            classes=[0],
            verbose=False
        )

        result = results[0]
        count = len(result.boxes)

        print(
            f"Confidence: {confidence} | "
            f"Image size: {image_size} | "
            f"People detected: {count}"
        )

        # Save annotated diagnostic frame
        output_path = (
            f"output/detection_c{confidence}_s{image_size}.jpg"
        )

        result.save(filename=output_path)

print("\nDetection diagnostic completed!")
