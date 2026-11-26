
import cv2

from src.video_processing.video_reader import read_video
from src.detection.detector import PersonDetector
from src.tracking.tracker import PersonTracker
from src.video_processing.frame_processor import process_frame


detector = PersonDetector(
    confidence_threshold=0.25,
    image_size=960
)

tracker = PersonTracker()

frames = read_video("videos/ooms test video.mp4")

target_frame = 1375
found_target = False

try:
    for video_frame in frames:

        processed = process_frame(video_frame)

        # YOLO detection
        detections = detector.detect(processed.image)

        # ByteTrack tracking
        tracked_people = tracker.update(detections)

        if video_frame.frame_number == target_frame:

            found_target = True

            print("Frame number:", video_frame.frame_number)
            print("YOLO detections:", len(detections))
            print("ByteTrack tracks:", len(tracked_people))

            # Save image with YOLO detections
            detection_image = processed.image.copy()

            for person in detections:
                x1, y1, x2, y2 = map(int, person.bbox)

                cv2.rectangle(
                    detection_image,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

            cv2.imwrite(
                "output/yolo_diagnostic.jpg",
                detection_image
            )

            # Save image with ByteTrack results
            tracking_image = processed.image.copy()

            for person in tracked_people:
                x1, y1, x2, y2 = map(int, person.bbox)

                cv2.rectangle(
                    tracking_image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    tracking_image,
                    f"ID: {person.track_id}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            cv2.imwrite(
                "output/tracker_diagnostic.jpg",
                tracking_image
            )

            break

finally:
    frames.close()

if not found_target:
    raise ValueError("Target frame was not found")

print("Diagnostic images saved successfully!")

