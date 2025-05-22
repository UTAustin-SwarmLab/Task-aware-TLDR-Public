import cv2
import numpy as np
import argparse
import subprocess
import os
from pathlib import Path
from merge_slideslive_video import remerge

# Increase FFmpeg read attempts to handle videos with multiple streams
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "10000"


def find_transition_time(video_path, second_frame_path):
    # detect when the second frame appears in the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return -1

    # read the second frame from given path
    second_frame = cv2.imread(second_frame_path)
    if second_frame is None:
        print(f"Error: Could not read second frame {second_frame_path}")
        return -1

    # Convert second frame to grayscale for comparison
    second_frame_gray = cv2.cvtColor(second_frame, cv2.COLOR_BGR2GRAY)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    threshold = 0.7  # Similarity threshold

    # We'll use this to detect when the second frame appears
    while True:
        ret, current_frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Convert current frame to grayscale
        current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Calculate similarity using template matching
        result = cv2.matchTemplate(current_frame_gray, second_frame_gray, cv2.TM_CCOEFF_NORMED)
        similarity = np.max(result)

        # If similarity rises above threshold, we've found the transition
        if similarity > threshold:
            transition_time = frame_count / fps
            print(f"Second frame appears at {transition_time:.2f} seconds (frame {frame_count})")
            cap.release()
            return transition_time

    # If no transition found
    cap.release()
    print("No transition detected - second frame never appears")
    return -1


def trim_video(video_path, output_path, trim_time):
    # Use ffmpeg to trim the video with proper re-encoding
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file if it exists
        "-i",
        video_path,
        "-ss",
        str(trim_time),  # Start time in seconds
        "-c:v",
        "libx264",  # Use H.264 codec
        "-preset",
        "medium",  # Balance between speed and compression
        "-crf",
        "23",  # Quality level (lower is better, 23 is good quality)
        "-c:a",
        "aac",  # Use AAC audio codec
        "-b:a",
        "192k",  # Audio bitrate
        output_path,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error trimming video: {e.stderr.decode()}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trim video at the transition point of the first frame")
    parser.add_argument("video_dir", help="Path to the video directory")
    parser.add_argument("--transition_time", type=float, help="Transition time in seconds")
    parser.add_argument("--remerge", action="store_true", help="Whether to remerge the audio and slides")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    video_path = video_dir / "video.mp4"
    second_frame_path = video_dir / "slide_2.png"
    output_path = video_dir / "video_trimmed.mp4"

    # Remerge audio and slides if requested
    if args.remerge:
        slideslive_id = video_dir.name
        metadata_file = Path("/nas/pohan/datasets/AIConfVideo/dataset") / f"slideslive_{slideslive_id}.json"
        audio_file = video_dir / "audio.m4a"
        print(f"Metadata file: {metadata_file}")
        print(f"Audio file: {audio_file}")
        remerge(metadata_file, audio_file, quiet=False, output_dir=video_dir)

    # find the transition time
    if not args.transition_time:
        transition_time = find_transition_time(video_path, second_frame_path)
    else:
        transition_time = args.transition_time

    # Trim the video starting from the transition frame
    if trim_video(video_path, output_path, transition_time):
        print(f"Successfully trimmed video and audio to {output_path}")
    else:
        print("Failed to trim video")
