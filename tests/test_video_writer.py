
from src.video_processing.video_reader import read_video
from src.video_processing.video_writer import ProcessedVideoWriter

# Read the original test video
frames = read_video("videos/test_video.mp4")

# Create the output video writer
writer = ProcessedVideoWriter(
    output_path="output/writer_test.mp4",
    fps=10,
    width=640,
    height=480
)

total_frames = 0

try:
    # Read and write each frame
    for video_frame in frames:
        writer.write(video_frame.image)
        total_frames += 1

finally:
    # Release both resources
    frames.close()
    writer.release()

print("Total frames written:", total_frames)
print("Output video saved: output/writer_test.mp4")

assert total_frames == 30

print("Video writer test passed!")
