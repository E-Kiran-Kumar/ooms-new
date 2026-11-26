
from dataclasses import dataclass

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker

from src.detection.detector import Detection


@dataclass
class TrackedPerson:
    """Stores information about a tracked person."""

    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int


class PersonTracker:
    """Tracks people across video frames using ByteTrack."""

    def __init__(self):
        # Initialize ByteTrack with our CCTV video settings
        self.tracker = ByteTrackTracker(
            minimum_consecutive_frames=1,
            track_activation_threshold=0.25,
            high_conf_det_threshold=0.25,
            frame_rate=25.0
        )

    def update(
        self,
        detections: list[Detection]
    ) -> list[TrackedPerson]:
        """
        Update tracking using detections from the current frame.

        Args:
            detections: YOLO detections from the current frame.

        Returns:
            List of successfully tracked people.
        """

        # Convert bounding boxes into a NumPy array
        xyxy = np.array(
            [d.bbox for d in detections],
            dtype=np.float32
        ).reshape(-1, 4)

        # Convert confidence scores
        confidence = np.array(
            [d.confidence for d in detections],
            dtype=np.float32
        )

        # Convert class IDs
        class_id = np.array(
            [d.class_id for d in detections],
            dtype=int
        )

        # Create Supervision detection object
        supervision_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )

        # Update ByteTrack
        tracked = self.tracker.update(
            supervision_detections
        )

        people = []

        # Extract successfully assigned tracking IDs
        for index in range(len(tracked)):

            track_id = tracked.tracker_id[index]

            # Ignore detections without an assigned track ID
            if track_id is None or track_id == -1:
                continue

            x1, y1, x2, y2 = tracked.xyxy[index]

            people.append(
                TrackedPerson(
                    track_id=int(track_id),
                    bbox=(
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2)
                    ),
                    confidence=float(tracked.confidence[index]),
                    class_id=int(tracked.class_id[index])
                )
            )

        return people
