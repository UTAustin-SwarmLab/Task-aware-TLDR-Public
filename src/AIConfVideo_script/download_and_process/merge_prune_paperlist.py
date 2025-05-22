import argparse
import json
import os
import multiprocessing
from merge_slideslive_video import download_audio, download_slides, merge_audio_and_slides

DATASET_DIR = "/home/undergrads-admin/ta-tldr/dataset"
VIDEO_DIR = "/home/undergrads-admin/ta-tldr/video"
CHECKPOINT_INTERVAL = 10  # Save checkpoint every 5 papers
DEFAULT_NUM_WORKERS = 8  # Default number of processes to use

def save_checkpoint(papers, checkpoint_file):
    """Save current progress to checkpoint file"""
    with open(checkpoint_file, "w") as f:
        json.dump(papers, f, indent=4)
    print(f"Checkpoint saved to {checkpoint_file}\n")

def process_paper(paper, idx, args, total_papers):
    """Process a single paper for multiprocessing"""
    slideslive_id = paper["slideslive_id"]
    metadata_file = os.path.join(args.dataset_dir, f"slideslive_{slideslive_id}.json")
    video_file = os.path.join(args.video_dir, f"{slideslive_id}", "video.mp4")
    
    if os.path.exists(video_file) and os.path.getsize(video_file) > 0:
        print(f"[{idx+1}/{total_papers}] ⏩ Skipping {slideslive_id}: video already exists")
        return (True, paper)
    else:
        try:
            download_audio(metadata_file, quiet=True, output_dir=args.video_dir)
            download_slides(metadata_file, quiet=True, output_dir=args.video_dir)
            merge_audio_and_slides(metadata_file, quiet=True, output_dir=args.video_dir)
            print(f"[{idx+1}/{total_papers}] ✅ Merged {slideslive_id}")
            return (True, paper)
        except Exception as e:
            print(f"[{idx+1}/{total_papers}] ❌ Error processing {slideslive_id}: {e}")
            return (False, None)

def main():
    parser = argparse.ArgumentParser(description="Prune paperlist")
    parser.add_argument("--dataset-dir", default=DATASET_DIR, type=str, help="Path to the dataset directory")
    parser.add_argument("--video-dir", default=VIDEO_DIR, type=str, help="Path to the video directory")
    parser.add_argument("--checkpoint-interval", default=CHECKPOINT_INTERVAL, type=int, 
                        help="Number of papers to process before saving checkpoint")
    parser.add_argument("--num-workers", default=DEFAULT_NUM_WORKERS, type=int,
                        help="Number of worker processes to use")
    args = parser.parse_args()
    
    # Limit number of workers based on CPU cores
    num_workers = min(args.num_workers, multiprocessing.cpu_count())
    print(f"Using {num_workers} worker processes")

    for file in os.listdir(args.dataset_dir):
        # for each paperlist file
        if file.startswith("paperlist_") and file.endswith(".json"):
            conf_name = file.split("_")[1].split(".")[0]
            print(f"------- Processing {conf_name} -------")

            # prune paperlist
            with open(os.path.join(args.dataset_dir, file), "r") as f:
                papers = json.load(f)

            # Create new filename for the pruned list
            new_file = file.replace("paperlist_", "pruned_paperlist_")
            checkpoint_file = os.path.join(args.dataset_dir, new_file)
            
            # Check if we have a checkpoint to resume from
            pruned_papers = []
            if os.path.exists(checkpoint_file):
                print(f"Found existing checkpoint at {checkpoint_file}")
                with open(checkpoint_file, "r") as f:
                    pruned_papers = json.load(f)
                    
            # Create a list of papers that have not been processed yet
            processed_ids = {paper["slideslive_id"] for paper in pruned_papers}
            papers_to_process = [p for p in papers if p["slideslive_id"] not in processed_ids]
            
            if not papers_to_process:
                print(f"All papers in {conf_name} have already been processed. Skipping.")
                continue
                
            print(f"Processing {len(papers_to_process)} remaining papers out of {len(papers)} total")
            
            # Process papers in batches to allow for checkpointing
            for i in range(0, len(papers_to_process), args.checkpoint_interval):
                batch = papers_to_process[i:i+args.checkpoint_interval]
                
                # Process the batch in parallel
                with multiprocessing.Pool(processes=num_workers) as pool:
                    # Create a list of tuples with arguments for each paper
                    process_args = [
                        (paper, i+idx, args, len(papers)) 
                        for idx, paper in enumerate(batch)
                    ]
                    
                    # Use starmap to process papers in parallel
                    results = pool.starmap(process_paper, process_args)
                
                # Update pruned_papers with successful results
                for success, paper in results:
                    if success and paper is not None and paper not in pruned_papers:
                        pruned_papers.append(paper)
                
                # Save checkpoint after each batch
                save_checkpoint(pruned_papers, checkpoint_file)
                
            print(f"Completed processing {conf_name}. Successfully processed {len(pruned_papers)} papers.")

if __name__ == "__main__":
    main()