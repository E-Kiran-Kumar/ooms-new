
import cv2
import numpy as np
from pathlib import Path

# Create the videos folder if it does not exist
Path("videos").mkdir(exist_ok=True)

# Video properties
width = 640
height = 480
fps = 10

# Create a video writer
writer = cv2.VideoWriter(
    "videos/test_video.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# Generate 30 frames
for frame_number in range(30):
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (100, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    writer.write(frame)

writer.release()

print("Test video created successfully!")
