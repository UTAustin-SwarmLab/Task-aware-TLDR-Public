import pandas as pd
import subprocess
from pathlib import Path

# Read the CSV file
df = pd.read_csv("/nas/pohan/datasets/AIConfVideo/video/sample_10_user_study_stimulus.csv")

# Create a directory to store the downloaded videos
output_dir = Path.home() / "Downloads" / "paper10_videos"
output_dir.mkdir(exist_ok=True)


# Function to download a single video
def download_video(row):
    source_path = row["trimmed_video_path"]
    filename = f"{row['openreview_id']}_{row['slideslive_id']}.mp4"
    target_path = output_dir / filename

    # Use scp to download the file
    cmd = f"scp ta-tldr@ece-a72388:{source_path} {target_path}"
    subprocess.run(cmd, shell=True)
    print(f"Downloaded {filename}")


# Download all videos
for _, row in df.iterrows():
    download_video(row)

print("All videos have been downloaded successfully!")
