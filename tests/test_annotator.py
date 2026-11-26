
import cv2
import numpy as np

from src.visualization.annotator import FrameAnnotator
from src.tracking.tracker import TrackedPerson

# Create a blank test frame
frame = np.zeros((480, 640, 3), dtype=np.uint8)

# Simulate one tracked person
tracked_people = [
    TrackedPerson(
        track_id=1,
        bbox=(100.0, 100.0, 250.0, 350.0),
        confidence=0.95,
        class_id=0
    )
]

# Create annotator
annotator = FrameAnnotator()

# Draw bounding box and tracking ID
annotated_frame = annotator.annotate(
    frame,
    tracked_people
)

# Save the annotated image
cv2.imwrite(
    "output/annotator_test.jpg",
    annotated_frame
)

# Validate output
assert annotated_frame.shape == (480, 640, 3)
assert not np.array_equal(frame, annotated_frame)

print("Frame shape:", annotated_frame.shape)
print("Tracked people annotated:", len(tracked_people))
print("Annotated image saved: output/annotator_test.jpg")
print("Annotation test passed!")
