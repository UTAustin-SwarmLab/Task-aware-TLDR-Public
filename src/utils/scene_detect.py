import argparse
from pathlib import Path

import cv2
import numpy as np
from scenedetect import ContentDetector, detect


def detect_scenes_and_extract_keyframes(video_path):
    # Detect scenes using ContentDetector
    scene_list = detect(video_path, ContentDetector(threshold=27.0))
    video_path = Path(video_path)
    video_name = video_path.stem
    output_dir = video_path.parent.parent / "keyframes" / str(video_name + "_scene_detect")
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(scene_list) == 0:
        print("No scenes detected. Extracting key frames based on content change.")

        # Open video and read frames
        video = cv2.VideoCapture(video_path)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        key_frames = []
        threshold = 0.8  # Content change threshold

        ret, prev_frame = video.read()
        for frame_num in range(1, frame_count):
            ret, frame = video.read()
            if not ret:
                break

            # Calculate content difference between consecutive frames
            content_val = calculate_content_change(prev_frame, frame)

            if content_val > threshold:
                key_frames.append((frame_num, frame))

            prev_frame = frame

        print(f"Detected {len(key_frames)} key frames:")
        for idx, (frame_num, frame) in enumerate(key_frames):
            print(f"Key frame at frame {frame_num}")
            cv2.imwrite(output_dir / f"{video_name}_keyframe_{idx + 1}.jpg", frame)
    else:
        print(f"Detected {len(scene_list)} scenes!")

        for idx, scene in enumerate(scene_list):
            print(f"Scene {idx + 1} starts at {scene[0].get_seconds()} seconds")

            # Save keyframe using OpenCV
            video = cv2.VideoCapture(video_path)
            video.set(cv2.CAP_PROP_POS_FRAMES, scene[0].get_frames())
            ret, frame = video.read()
            if ret:
                cv2.imwrite(output_dir / f"{video_name}_keyframe_{idx + 1}.jpg", frame)


# Calculate content change using structural similarity or frame difference
def calculate_content_change(frame1, frame2):
    diff = cv2.absdiff(frame1, frame2)
    non_zero_count = np.count_nonzero(diff)
    total_pixels = diff.size
    return non_zero_count / total_pixels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect scenes and extract keyframes from a video.")
    parser.add_argument("video_path", type=str, help="Path to the video file.")
    args = parser.parse_args()

    detect_scenes_and_extract_keyframes(args.video_path)
