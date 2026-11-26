
import numpy as np

from dataclasses import dataclass


@dataclass
class ProcessedFrame:
    """Stores a validated frame and its original dimensions."""

    frame_number: int
    timestamp: float
    image: np.ndarray
    original_width: int
    original_height: int


def process_frame(video_frame) -> ProcessedFrame:
    """
    Validate and prepare one video frame for object detection.
    """

    frame = video_frame.image

    # Check whether the frame is a valid NumPy array
    if not isinstance(frame, np.ndarray):
        raise TypeError("Frame must be a NumPy array")

    # Check whether the frame has three color channels
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Frame must be a 3-channel color image")

    # Get original frame dimensions
    height, width = frame.shape[:2]

    # Check whether the frame is empty
    if height == 0 or width == 0:
        raise ValueError("Frame cannot be empty")

    # Return the validated frame with its metadata
    return ProcessedFrame(
        frame_number=video_frame.frame_number,
        timestamp=video_frame.timestamp,
        image=frame,
        original_width=width,
        original_height=height
    )
