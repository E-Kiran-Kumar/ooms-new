
import cv2
import numpy as np

from src.tracking.tracker import TrackedPerson


class FrameAnnotator:
    """Draws detection and tracking information on video frames."""

    def annotate(
        self,
        frame: np.ndarray,
        tracked_people: list[TrackedPerson]
    ) -> np.ndarray:
        """Draw bounding boxes and tracking IDs on a frame."""

        annotated_frame = frame.copy()

        for person in tracked_people:
            x1, y1, x2, y2 = person.bbox

            # Convert coordinates to integers
            x1, y1, x2, y2 = map(
                int, (x1, y1, x2, y2)
            )

            # Draw bounding box
            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Create person ID label
            label = f"Person ID: {person.track_id}"

            # Draw label
            cv2.putText(
                annotated_frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        return annotated_frame
