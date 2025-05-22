import cv2
import argparse
from pathlib import Path
from tqdm import tqdm

# python check_video_duration.py /nas/pohan/datasets/SUTDTrafficQA/raw_videos/ --threshold 7 --output plotting/long_videos.csv

def get_video_duration(video_path):
    """Get the duration of a video in seconds"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video {video_path}")
        return 0

    # Get frame count and fps
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate duration
    duration = frame_count / fps if fps > 0 else 0

    cap.release()
    return duration


def check_video_durations(directory, threshold=10, output_file=None):
    """
    Check the duration of each video in a directory and record videos longer than threshold

    Args:
        directory: Directory containing video files
        threshold: Minimum duration in seconds to include in results
        output_file: Optional file to write results to

    Returns:
        List of (filename, duration) tuples for videos longer than threshold
    """
    # Common video extensions
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"]

    # Get all video files in the directory
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(Path(directory).glob(f"*{ext}")))

    # Check duration of each video
    long_videos = []

    print(f"Found {len(video_files)} video files. Checking durations...")

    for video_path in tqdm(video_files, desc="Processing videos"):
        duration = get_video_duration(str(video_path))
        # print(f"{video_path.name}: {duration:.2f}s")

        if duration > threshold:
            long_videos.append((video_path.name, duration))

    # Sort by duration (longest first)
    long_videos.sort(key=lambda x: x[1], reverse=True)

    # Write results to file if output_file is specified
    if output_file:
        with open(output_file, "w") as f:
            f.write("filename,duration\n")
            for name, duration in long_videos:
                f.write(f"{name},{duration:.2f}\n")

    # Print summary
    print(f"\nFound {len(long_videos)} videos longer than {threshold} seconds:")
    # for name, duration in long_videos:
    #     print(f"{name}: {duration:.2f}s")

    return long_videos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check video durations in a directory")
    parser.add_argument("directory", help="Directory containing video files")
    parser.add_argument("--threshold", type=float, default=10.0, help="Minimum duration in seconds (default: 10.0)")
    parser.add_argument("--output", help="Output file to write results")

    args = parser.parse_args()

    check_video_durations(args.directory, args.threshold, args.output)
