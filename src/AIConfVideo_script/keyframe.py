# used https://github.com/keplerlab/katna to extract keyframes
# python src/AIConfVideo_script/keyframe.py --video_path /path/to/video.mp4 --output_dir /path/to/output --min_fps 1.0 --override
# default is 1.0 fps, max 30 frames
import argparse
import os
import tempfile

import cv2
import numpy as np
from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter
from PIL import Image
from tabulate import tabulate


def parse_args():
    parser = argparse.ArgumentParser(description="Extract keyframes from a video")
    parser.add_argument(
        "--video_path", type=str, required=False, default="b_1a4411B7sb_clip_005.mp4", help="Path to the video file"
    )
    parser.add_argument(
        "--output_dir", type=str, required=False, default="keyframes", help="Path to the output directory"
    )
    parser.add_argument(
        "--max_frames", type=int, required=False, default=30, help="Maximum number of keyframes to extract (optional)"
    )
    parser.add_argument(
        "--override", action="store_true", help="Automatically override existing keyframes without prompting"
    )
    parser.add_argument("--not_recompute", action="store_true", help="Do not recompute keyframes even if they exist")
    parser.add_argument(
        "--min_fps", type=float, required=False, default=1.0, help="Minimum frames per second to extract (default: 1.0)"
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    return parser.parse_args()


def log(message, quiet=False, level=None):
    """Print log messages with consistent formatting based on verbosity level"""
    if quiet and level in ["INFO", "PROGRESS", None]:
        return

    if level is None:
        prefix = ""
    else:
        prefix = {"INFO": "ℹ️  ", "SUCCESS": "✅ ", "ERROR": "❌ ", "PROGRESS": "\t🔄 "}.get(level, "")

    print(f"{prefix}{message}")


def get_video_duration(video_path):
    """Get the duration of a video in seconds"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log(f"Could not open video {video_path}", level="ERROR")
        return 0

    # Get frame count and fps
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate duration
    duration = frame_count / fps if fps > 0 else 0

    cap.release()
    return duration


def extract_frames_by_katna(video_path, temp_dir, num_frames=30, quiet=False):
    """Extract keyframes using Katna's intelligent frame selection"""
    vd = Video()
    diskwriter = KeyFrameDiskWriter(location=temp_dir)

    # Extract keyframes using Katna
    vd.extract_video_keyframes(file_path=video_path, no_of_frames=num_frames, writer=diskwriter)

    # Load the extracted frames more efficiently
    frames = []
    frame_files = sorted([f for f in os.listdir(temp_dir) if f.endswith((".jpg", ".jpeg", ".png"))])

    for frame_file in frame_files:
        frame_path = os.path.join(temp_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is not None:
            frames.append(frame)

    return frames


def extract_frames_by_interval(video_path, temp_dir, interval_seconds, quiet=False):
    """Extract frames at regular intervals from a video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log(f"Could not open video {video_path}", level="ERROR")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_seconds)

    if frame_interval <= 0:
        frame_interval = 1  # Ensure we take at least some frames

    frames = []
    frame_count = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Save frame to temp directory
            frame_path = os.path.join(temp_dir, f"frame_{frame_idx:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            frames.append(frame)
            frame_idx += 1

        frame_count += 1

    cap.release()
    return frames


def extract_keyframes(
    video_path, output_dir, max_frames=None, override=False, not_recompute=False, min_fps=1.0, quiet=False
):
    # Create a unique subfolder for this video
    video_name = os.path.basename(video_path)
    video_output_dir = os.path.join(output_dir, os.path.splitext(video_name)[0])

    # Check if the directory already exists and contains keyframes
    if os.path.exists(video_output_dir):
        keyframes = [
            f
            for f in os.listdir(video_output_dir)
            if f.lower().startswith("keyframe") and f.endswith((".jpg", ".jpeg", ".png"))
        ]
        if keyframes:
            log(f"Found existing directory with {len(keyframes)} keyframes", quiet)

            # If override is true, automatically proceed with recomputation
            if override:
                log("Automatically proceeding with recomputation due to override flag", quiet, level="INFO")
            elif not_recompute:
                log("Skipping recomputation due to not_recompute flag", quiet, level="INFO")
                return video_output_dir
            else:
                # Ask user if they want to recompute keyframes
                while True:
                    response = input("Recompute keyframes? (y/n): ")
                    if response.lower() in ["y", "yes"]:
                        log("Proceeding with recomputation...", quiet, level="INFO")
                        break
                    elif response.lower() in ["n", "no"]:
                        log(f"Using existing keyframes in {video_output_dir}", quiet, level="SUCCESS")
                        return video_output_dir
                    else:
                        print("Please enter 'y' or 'n'")

    os.makedirs(video_output_dir, exist_ok=True)

    # Get video duration
    duration = get_video_duration(video_path)
    if duration <= 0:
        log("Could not determine video duration", level="ERROR")
        return video_output_dir

    # Calculate minimum number of frames needed based on min_fps
    min_frames_needed = int(duration * min_fps)
    log(f"Video duration: {duration}s, min fps: {min_fps}, min frames needed: {min_frames_needed}", quiet)

    # Create a temporary directory for initial frames
    with tempfile.TemporaryDirectory() as temp_dir:
        # Try to extract frames using Katna first
        log(f"Attempting to extract {min_frames_needed} keyframes using Katna...", quiet, level="PROGRESS")
        frames = extract_frames_by_katna(video_path, temp_dir, min_frames_needed, quiet)
        total_frames = len(frames)

        # If Katna didn't extract enough frames to meet min_fps, use interval-based extraction
        if total_frames < min_frames_needed:
            log(f"\tKatna extracted only {total_frames}/{min_frames_needed} frames", quiet)

            # Calculate interval in seconds between frames
            interval = 1.0 / min_fps

            # Extract frames at regular intervals
            log("Switching to interval-based extraction...", quiet, level="PROGRESS")
            frames = extract_frames_by_interval(video_path, temp_dir, interval, quiet)
            total_frames = len(frames)
            log(f"\tExtracted {total_frames} frames using interval-based method", quiet)

        if total_frames == 0:
            log("No frames extracted", level="ERROR")
            return video_output_dir

        # Apply max_frames limit if specified
        if max_frames is not None and total_frames > max_frames:
            log(f"Limiting frames from {total_frames} to {max_frames}...", quiet, level="PROGRESS")
            # Select frames evenly distributed across the original set
            indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
            frames = [frames[i] for i in indices]
            total_frames = len(frames)

        # Save all frames as keyframes
        for i, frame in enumerate(frames):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            output_path = os.path.join(video_output_dir, f"keyframe_{i:03d}.jpg")
            img.save(output_path)

        log(f"Complete! {total_frames} frames saved to: {video_output_dir}", quiet, level="SUCCESS")
        return video_output_dir


if __name__ == "__main__":
    args = parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Print the parameters being used (only if not in quiet mode)
    if not args.quiet:
        params = [
            ["Video", os.path.basename(args.video_path)],
            ["Output", args.output_dir],
            ["Min FPS", args.min_fps],
            ["Max frames", args.max_frames or "Auto"],
            ["Override", "Yes" if args.override else "No"],
            ["Not recompute", "Yes" if args.not_recompute else "No"],
        ]
        print(tabulate(params, tablefmt="pretty"))

    # Extract keyframes
    # log(f"Processing video: {os.path.basename(args.video_path)}", args.quiet)
    output_dir = extract_keyframes(
        args.video_path, args.output_dir, args.max_frames, args.override, args.not_recompute, args.min_fps, args.quiet
    )
