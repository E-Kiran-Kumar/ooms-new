
from src.video_processing.video_reader import read_video, VideoFrame

# Read the generated test video
frames = list(read_video("videos/test_video.mp4"))

# Check the total number of frames
print("Total frames extracted:", len(frames))

# Display information about the first frame
first_frame = frames[0]

print("Frame number:", first_frame.frame_number)
print("Timestamp:", first_frame.timestamp)
print("Image shape:", first_frame.image.shape)
print("Image data type:", first_frame.image.dtype)

# Verify the results
assert len(frames) == 30
assert isinstance(first_frame, VideoFrame)
assert first_frame.image.shape == (480, 640, 3)

print("Video reader test passed!")

