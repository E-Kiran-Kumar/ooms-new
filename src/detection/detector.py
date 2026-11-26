
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    """Stores information about one detected object."""

    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


class PersonDetector:
    """Detects people in video frames using YOLOv8n."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.25,
        image_size: int = 960
    ):
        # Load the YOLO model
        self.model = YOLO(model_path)

        # Store detection configuration
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size

    def detect(self, image: np.ndarray) -> list[Detection]:
        """
        Detect people in one image.

        Args:
            image: Input image as a NumPy array in BGR format.

        Returns:
            A list of detected people with bounding boxes and confidence.
        """

        # Validate input image
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be a NumPy array")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Image must be a 3-channel color image")

        if image.size == 0:
            raise ValueError("Image cannot be empty")

        # Run YOLO prediction
        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            imgsz=self.image_size,
            classes=[0],
            verbose=False
        )

        detections = []

        # Extract detected people
        for result in results:
            for box in result.boxes:

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detection = Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(box.conf[0]),
                    class_id=int(box.cls[0]),
                    class_name="person"
                )

                detections.append(detection)

        return detections
