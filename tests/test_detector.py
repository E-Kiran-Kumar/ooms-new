
from src.video_processing.video_reader import read_video
from src.video_processing.frame_processor import process_frame
from src.detection.detector import PersonDetector

# Load the YOLO model
detector = PersonDetector(
    model_path="yolov8n.pt",
    confidence_threshold=0.5
)

# Read the test video
frames = read_video("videos/test_video.mp4")

# Get the first frame
video_frame = next(frames)

# Process the frame
processed_frame = process_frame(video_frame)

# Run YOLO detection
detections = detector.detect(processed_frame.image)

# Display the results
print("Total people detected:", len(detections))

for index, detection in enumerate(detections, start=1):
    print(f"\nPerson {index}")
    print("Bounding box:", detection.bbox)
    print("Confidence:", detection.confidence)
    print("Class ID:", detection.class_id)
    print("Class name:", detection.class_name)

# Our generated test video contains no people
assert len(detections) == 0

print("\nYOLO detector test passed!")

# Release the video
frames.close()
