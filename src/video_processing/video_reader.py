
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoFrame:
    """Stores one extracted video frame and its metadata."""

    frame_number: int
    timestamp: float
    image: np.ndarray


def read_video(video_path: str) -> Iterator[VideoFrame]:
    """
    Read a video and yield its frames one by one.

    Args:
        video_path: Path to the input video.

    Yields:
        VideoFrame: Frame number, timestamp and image array.
    """

    path = Path(video_path)

    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = 0

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            if fps > 0:
                timestamp = frame_number / fps
            else:
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            yield VideoFrame(
                frame_number=frame_number,
                timestamp=timestamp,
                image=frame
            )

            frame_number += 1

    finally:
        cap.release()
