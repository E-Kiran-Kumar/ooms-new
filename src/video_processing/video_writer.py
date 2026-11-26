
from pathlib import Path

import cv2
import numpy as np


class ProcessedVideoWriter:
    """Writes processed frames into an output video."""

    def __init__(
        self,
        output_path: str,
        fps: float,
        width: int,
        height: int
    ):
        self.output_path = Path(output_path)

        # Create the output directory if needed
        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # Configure the video encoder
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            fps,
            (width, height)
        )

        if not self.writer.isOpened():
            raise ValueError(
                f"Unable to create output video: {output_path}"
            )

    def write(self, frame: np.ndarray) -> None:
        """Write one frame into the output video."""

        if not isinstance(frame, np.ndarray):
            raise TypeError("Frame must be a NumPy array")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                "Frame must be a 3-channel color image"
            )

        if frame.size == 0:
            raise ValueError("Frame cannot be empty")

        self.writer.write(frame)

    def release(self) -> None:
        """Close the video writer and save the video."""

        self.writer.release()
