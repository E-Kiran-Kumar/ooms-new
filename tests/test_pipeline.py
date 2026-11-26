
from src.video_processing.video_reader import read_video
from src.video_processing.frame_processor import process_frame
from src.detection.detector import PersonDetector
from src.tracking.tracker import PersonTracker


def main():

    # Initialize YOLO and ByteTrack only once
    detector = PersonDetector(
        model_path="yolov8n.pt",
        confidence_threshold=0.5
    )

    tracker = PersonTracker()

    # Read the video
    frames = read_video("videos/test_video.mp4")

    total_frames = 0
    total_detections = 0
    total_tracked_people = 0

    try:
        for video_frame in frames:

            # Step 1: Validate the current frame
            processed_frame = process_frame(video_frame)

            # Step 2: Detect people using YOLO
            detections = detector.detect(processed_frame.image)

            # Step 3: Track detected people
            tracked_people = tracker.update(detections)

            total_frames += 1
            total_detections += len(detections)
            total_tracked_people += len(tracked_people)

            print(
                f"Frame: {video_frame.frame_number} | "
                f"Detections: {len(detections)} | "
                f"Tracked: {len(tracked_people)}"
            )

    finally:
        frames.close()

    print("\n========== PIPELINE SUMMARY ==========")
    print("Total frames processed:", total_frames)
    print("Total detections:", total_detections)
    print("Total tracked observations:", total_tracked_people)

    assert total_frames == 30
    assert total_detections == 0

    print("\nOOMS integration test passed!")


if __name__ == "__main__":
    main()
