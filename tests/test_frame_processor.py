
from src.video_processing.video_reader import read_video
from src.video_processing.frame_processor import process_frame, ProcessedFrame


# Read the test video
frames = read_video("videos/test_video.mp4")

# Get the first frame
first_frame = next(frames)

# Pass the extracted frame to the frame processor
processed_frame = process_frame(first_frame)

# Display the results
print("Frame number:", processed_frame.frame_number)
print("Timestamp:", processed_frame.timestamp)
print("Image shape:", processed_frame.image.shape)
print("Original width:", processed_frame.original_width)
print("Original height:", processed_frame.original_height)

# Verify the results
assert isinstance(processed_frame, ProcessedFrame)
assert processed_frame.image.shape == (480, 640, 3)
assert processed_frame.original_width == 640
assert processed_frame.original_height == 480

print("Frame processor test passed!")

# Release the generator after reading only one frame
frames.close()
