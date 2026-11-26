
import cv2

from src.video_processing.video_reader import read_video
from src.video_processing.frame_processor import process_frame
from src.video_processing.video_writer import ProcessedVideoWriter

from src.detection.detector import PersonDetector
from src.tracking.tracker import PersonTracker
from src.visualization.annotator import FrameAnnotator


class OOMSPipeline:
    """Connects all OOMS video processing modules."""

    def __init__(
        self,
        input_path: str,
        output_path: str,
        confidence_threshold: float = 0.25,
        image_size: int = 960
    ):
        self.input_path = input_path
        self.output_path = output_path

        # Initialize YOLO detector
        self.detector = PersonDetector(
            confidence_threshold=confidence_threshold,
            image_size=image_size
        )

        # Initialize tracker and annotator
        self.tracker = PersonTracker()
        self.annotator = FrameAnnotator()

    def run(self):

        # Read original video FPS
        video_info = cv2.VideoCapture(self.input_path)

        if not video_info.isOpened():
            video_info.release()
            raise ValueError(
                f"Unable to open video: {self.input_path}"
            )

        fps = video_info.get(cv2.CAP_PROP_FPS)
        video_info.release()

        if fps <= 0:
            raise ValueError("Invalid video FPS")

        # Start reading video frames
        frames = read_video(self.input_path)

        writer = None

        total_frames = 0
        total_detections = 0
        total_tracks = 0

        try:

            for video_frame in frames:

                # Step 1: Validate and prepare frame
                processed_frame = process_frame(video_frame)

                # Step 2: Detect people using YOLO
                detections = self.detector.detect(
                    processed_frame.image
                )

                # Step 3: Track detected people
                tracked_people = self.tracker.update(
                    detections
                )

                # Step 4: Draw bounding boxes and IDs
                annotated_frame = self.annotator.annotate(
                    processed_frame.image,
                    tracked_people
                )

                # Step 5: Initialize output video writer
                if writer is None:

                    height, width = annotated_frame.shape[:2]

                    writer = ProcessedVideoWriter(
                        output_path=self.output_path,
                        fps=fps,
                        width=width,
                        height=height
                    )

                # Step 6: Write annotated frame
                writer.write(annotated_frame)

                # Update statistics
                total_frames += 1
                total_detections += len(detections)
                total_tracks += len(tracked_people)

                print(
                    f"Frame: {video_frame.frame_number} | "
                    f"Detections: {len(detections)} | "
                    f"Tracks: {len(tracked_people)}"
                )

        finally:

            # Release video reader
            frames.close()

            # Release video writer
            if writer is not None:
                writer.release()

        # Display pipeline summary
        print("\n========== OOMS PIPELINE SUMMARY ==========")

        print("Original FPS:", fps)
        print("Total frames:", total_frames)
        print("Total detections:", total_detections)
        print("Total tracked observations:", total_tracks)
        print("Output video:", self.output_path)


if __name__ == "__main__":

    pipeline = OOMSPipeline(
        input_path="videos/ooms test video.mp4",
        output_path="output/ooms_pipeline_output.mp4",
        confidence_threshold=0.25,
        image_size=960
    )

    pipeline.run()
