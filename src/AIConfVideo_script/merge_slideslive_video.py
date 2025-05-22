import argparse
import json
import logging
import shutil
import subprocess
from pathlib import Path

import yt_dlp

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def format_seconds_to_timestamp(seconds):
    """Convert seconds to a timestamp format (MM:SS)"""
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes}:{remaining_seconds:02d}"


def get_metadata_and_output_dir(metadata_file, output_dir=None):
    """Extract metadata and create output directory"""
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    slideslive_id = metadata.get("presentation_id", "unknown")

    # Create output directory
    if output_dir is None:
        script_dir = Path(__file__).parent.absolute()
        output_dir = script_dir.parent / "video" / str(slideslive_id)
    else:
        # Convert string to Path object if it's not already a Path
        output_dir = Path(output_dir) / str(slideslive_id)

    # Create the directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    return metadata, output_dir


def download_audio(metadata_file, quiet=False, output_dir=None):
    """Download audio from SlidesLive URL"""
    metadata, output_dir = get_metadata_and_output_dir(metadata_file, output_dir)
    url = metadata["slideslive_url"]

    if not quiet:
        print(f"Downloading audio from {url}...")

    # Try primary download method
    output_template = str(output_dir / "audio.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "m4a",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Check for downloaded file
        audio_file = output_dir / "audio.m4a"
        if audio_file.exists():
            return str(audio_file)

        # Look for any audio file
        audio_files = list(output_dir.glob("audio*"))
        if audio_files:
            return str(audio_files[0])

    except Exception as e:
        if not quiet:
            error_msg = str(e).split("\n")[0]
            print(f"Download error: {error_msg}")
            print("Trying alternative download method...")

        # Try fallback method
        simple_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(output_dir / "audio_simple.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
        }

        try:
            with yt_dlp.YoutubeDL(simple_opts) as ydl:
                ydl.download([url])

            audio_files = list(output_dir.glob("audio_simple*"))
            if audio_files:
                return str(audio_files[0])
        except Exception:
            pass

    if not quiet:
        print("Warning: No audio file found")
    return None


def download_slides(metadata_file, quiet=False, output_dir=None):
    """Download slides from metadata"""
    metadata, output_dir = get_metadata_and_output_dir(metadata_file, output_dir)

    # Import requests here to avoid dependency issues if not needed
    import requests
    from tqdm import tqdm

    # Get slides from metadata
    slides = metadata.get("slides", [])
    if not slides:
        if not quiet:
            print("No slides found in metadata")
        return str(output_dir)

    if not quiet:
        print(f"Downloading {len(slides)} slides to {output_dir}")

    # Download each slide
    successful_downloads = 0
    for slide in tqdm(slides, desc="Downloading slides", disable=quiet):
        slide_number = slide.get("slide_number")
        image_url = slide.get("image_url")

        if not image_url:
            if not quiet:
                print(f"No image URL for slide {slide_number}")
            continue

        output_file = output_dir / f"slide_{slide_number}.png"

        try:
            response = requests.get(image_url)
            if response.status_code == 200:
                with open(output_file, "wb") as f:
                    f.write(response.content)
                successful_downloads += 1
            elif not quiet:
                print(f"Failed to download slide {slide_number}: HTTP {response.status_code}")
        except Exception as e:
            if not quiet:
                print(f"Error downloading slide {slide_number}: {e}")

    if not quiet:
        print(f"Successfully downloaded {successful_downloads} of {len(slides)} slides")

    return str(output_dir)


def find_audio_file(output_dir):
    """Find audio file in output directory"""
    # Check common audio filenames
    for name in ["audio.m4a", "audio_simple.m4a", "talk_audio.m4a"]:
        audio_file = output_dir / name
        if audio_file.exists():
            return str(audio_file)

    # Try to find any audio file
    audio_files = list(output_dir.glob("*.m4a")) + list(output_dir.glob("*.mp3"))
    if audio_files:
        return str(audio_files[0])

    return None


def merge_audio_and_slides(metadata_file, quiet=False, output_dir=None):
    """Merge audio and slides into a single video"""
    if not quiet:
        print("Merging audio and slides into a single video...")
    logger.debug("Merging audio and slides into a single video...")

    try:
        metadata, output_dir = get_metadata_and_output_dir(metadata_file, output_dir)

        # Find audio file
        audio_file = find_audio_file(output_dir)
        if not audio_file:
            msg = f"No audio file found in {output_dir}"
            if not quiet:
                print(f"Error: {msg}")
            logger.error(msg)
            return False

        output_file = str(output_dir / "video.mp4")

        # Get slides and check if they exist
        slides = metadata.get("slides", [])
        if not slides:
            msg = "No slide data found in metadata"
            if not quiet:
                print(f"Error: {msg}")
            logger.error(msg)
            return False

        # Sort slides by timestamp
        slides.sort(key=lambda x: x.get("timestamp_seconds", 0))

        # Get total duration
        total_duration = metadata.get("total_duration", 0)
        if total_duration == 0 and slides:
            total_duration = slides[-1].get("timestamp_seconds", 0) + 30  # Add buffer

        if not quiet:
            print(
                f"Creating video with {len(slides)} slides and duration of {format_seconds_to_timestamp(total_duration)}"
            )

        # Create temporary directory
        temp_dir = output_dir / "temp"
        temp_dir.mkdir(exist_ok=True)
        input_file = temp_dir / "input.txt"

        # Create ffmpeg input file
        with open(input_file, "w") as f:
            f.write("ffconcat version 1.0\n")

            for i, slide in enumerate(slides):
                slide_number = slide.get("slide_number", i + 1)
                slide_file = output_dir / f"slide_{slide_number}.png"

                if not slide_file.exists():
                    logger.warning(f"Slide image {slide_file} not found, skipping")
                    continue

                # Calculate duration for this slide
                current_time = slide.get("timestamp_seconds", 0)
                if i < len(slides) - 1:
                    next_time = slides[i + 1].get("timestamp_seconds", 0)
                    duration = next_time - current_time
                else:
                    duration = total_duration - current_time

                duration = max(0.5, duration)  # Ensure minimum duration

                f.write(f"file '{slide_file.absolute()}'\n")
                f.write(f"duration {duration}\n")

        # Create video from slides
        if not quiet:
            print("Creating video with ffmpeg (this may take a while)...")

        slides_video = temp_dir / "slides.mp4"
        slides_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(input_file),
            "-vsync",
            "vfr",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # Ensure dimensions are even
            str(slides_video),
        ]

        # Run ffmpeg commands with appropriate output handling
        stdout = subprocess.PIPE if quiet else None
        stderr = subprocess.PIPE if quiet else None

        try:
            subprocess.run(slides_cmd, check=True, stdout=stdout, stderr=stderr)

            # Combine slides with audio
            if not quiet:
                print("Combining slides with audio...")

            final_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(slides_video),
                "-i",
                audio_file,
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
                "-shortest",
                output_file,
            ]

            subprocess.run(final_cmd, check=True, stdout=stdout, stderr=stderr)

            if not quiet:
                print(f"Video successfully created: {output_file}")
            return True

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode() if hasattr(e, "stderr") else str(e)
            if not quiet:
                print(f"Error running ffmpeg: {error_msg}")
            logger.error(f"ffmpeg error: {error_msg}")
            return False

    except Exception as e:
        if not quiet:
            print(f"Error creating video: {e}")
        logger.error(f"Error creating video: {e}")
        return False

    finally:
        # Clean up temporary files
        try:
            if "temp_dir" in locals() and temp_dir.exists():
                shutil.rmtree(temp_dir)
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {e}")

    return False


def remerge(metadata_file, audio_file, quiet=False, output_dir=None):
    """Merge audio and slides into a single video"""
    if not quiet:
        print("Merging audio and slides into a single video...")
    logger.debug("Merging audio and slides into a single video...")

    try:
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        output_file = str(output_dir / "video.mp4")

        # Get slides and check if they exist
        slides = metadata.get("slides", [])
        if not slides:
            msg = "No slide data found in metadata"
            if not quiet:
                print(f"Error: {msg}")
            logger.error(msg)
            return False

        # Sort slides by timestamp
        slides.sort(key=lambda x: x.get("timestamp_seconds", 0))

        # Get total duration
        total_duration = metadata.get("total_duration", 0)
        if total_duration == 0 and slides:
            total_duration = slides[-1].get("timestamp_seconds", 0) + 30  # Add buffer

        if not quiet:
            print(
                f"Creating video with {len(slides)} slides and duration of {format_seconds_to_timestamp(total_duration)}"
            )

        # Create temporary directory
        temp_dir = output_dir / "temp"
        temp_dir.mkdir(exist_ok=True)
        input_file = temp_dir / "input.txt"

        # Create ffmpeg input file
        with open(input_file, "w") as f:
            f.write("ffconcat version 1.0\n")

            for i, slide in enumerate(slides):
                slide_number = slide.get("slide_number", i + 1)
                slide_file = output_dir / f"slide_{slide_number}.png"

                if not slide_file.exists():
                    logger.warning(f"Slide image {slide_file} not found, skipping")
                    continue

                # Calculate duration for this slide
                current_time = slide.get("timestamp_seconds", 0)
                if i < len(slides) - 1:
                    next_time = slides[i + 1].get("timestamp_seconds", 0)
                    duration = next_time - current_time
                else:
                    duration = total_duration - current_time

                duration = max(0.5, duration)  # Ensure minimum duration

                f.write(f"file '{slide_file.absolute()}'\n")
                f.write(f"duration {duration}\n")

        # Create video from slides
        if not quiet:
            print("Creating video with ffmpeg (this may take a while)...")

        slides_video = temp_dir / "slides.mp4"
        slides_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(input_file),
            "-vsync",
            "vfr",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # Ensure dimensions are even
            str(slides_video),
        ]

        # Run ffmpeg commands with appropriate output handling
        stdout = subprocess.PIPE if quiet else None
        stderr = subprocess.PIPE if quiet else None

        try:
            subprocess.run(slides_cmd, check=True, stdout=stdout, stderr=stderr)

            # Combine slides with audio
            if not quiet:
                print("Combining slides with audio...")

            final_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(slides_video),
                "-i",
                audio_file,
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
                "-shortest",
                output_file,
            ]

            subprocess.run(final_cmd, check=True, stdout=stdout, stderr=stderr)

            if not quiet:
                print(f"Video successfully created: {output_file}")
            return True

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode() if hasattr(e, "stderr") else str(e)
            if not quiet:
                print(f"Error running ffmpeg: {error_msg}")
            logger.error(f"ffmpeg error: {error_msg}")
            return False

    except Exception as e:
        if not quiet:
            print(f"Error creating video: {e}")
        logger.error(f"Error creating video: {e}")
        return False

    finally:
        # Clean up temporary files
        try:
            if "temp_dir" in locals() and temp_dir.exists():
                shutil.rmtree(temp_dir)
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {e}")

    return False


def main():
    parser = argparse.ArgumentParser(description="Process SlidesLive presentations")
    parser.add_argument("metadata_file", help="Path to the metadata JSON file")
    parser.add_argument("--output-dir", help="Directory to save the output files")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip merging audio and slides into video",
    )
    args = parser.parse_args()

    # Download audio
    audio_file = download_audio(args.metadata_file, args.quiet, args.output_dir)
    if not audio_file:
        print("Error: Failed to download audio")
        return 1

    if not args.quiet:
        print(f"Success: Audio downloaded to {audio_file}")

    # Download slides
    output_dir = download_slides(args.metadata_file, args.quiet, args.output_dir)
    if not args.quiet:
        print(f"Slides and audio downloaded to {output_dir}")

    # Merge audio and slides into video
    if not args.skip_merge:
        success = merge_audio_and_slides(args.metadata_file, args.quiet, args.output_dir)
        if not success:
            print("Error: Failed to merge audio and slides into video")
            return 1
        elif not args.quiet:
            print("Successfully merged audio and slides into video")

    return 0


if __name__ == "__main__":
    exit(main())
