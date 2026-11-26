
from src.tracking.tracker import PersonTracker
from src.detection.detector import Detection

# Create one tracker instance
tracker = PersonTracker()

# Frame 1: Person appears
frame1_detections = [
    Detection(
        bbox=(100.0, 100.0, 200.0, 300.0),
        confidence=0.95,
        class_id=0,
        class_name="person"
    )
]

frame1_tracks = tracker.update(frame1_detections)
print("Frame 1 tracks:", frame1_tracks)


# Frame 2: Same person moves slightly
frame2_detections = [
    Detection(
        bbox=(105.0, 100.0, 205.0, 300.0),
        confidence=0.93,
        class_id=0,
        class_name="person"
    )
]

frame2_tracks = tracker.update(frame2_detections)
print("Frame 2 tracks:", frame2_tracks)


# Frame 3: Same person moves again
frame3_detections = [
    Detection(
        bbox=(110.0, 100.0, 210.0, 300.0),
        confidence=0.94,
        class_id=0,
        class_name="person"
    )
]

frame3_tracks = tracker.update(frame3_detections)
print("Frame 3 tracks:", frame3_tracks)


# Check that the person is tracked in frames 2 and 3
assert len(frame2_tracks) > 0
assert len(frame3_tracks) > 0

# Check whether the same tracking ID is maintained
assert frame2_tracks[0].track_id == frame3_tracks[0].track_id

print("\nPerson tracking test passed!")
print("Same person retained tracking ID:", frame3_tracks[0].track_id)
