import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from extract_metadata import (
    extract_from_openreview,
    extract_from_slideslive,
    extract_slideslive_openreview_id,
)


def filter_required_fields(paperlist_path):
    # Load paper data
    with open(paperlist_path, "r") as f:
        papers = json.load(f)

    # Define required fields and filter papers
    required_fields = [
        "id",
        "title",
        "track",
        "status",
        "keywords",
        "author",
        "aff",
        "site",
        "primary_area",
    ]

    filtered_papers = []
    for paper in papers:
        # Check if paper has all required fields and none are empty
        if all(key in paper and paper.get(key) for key in required_fields):
            filtered_paper = {key: paper[key] for key in required_fields}
            filtered_papers.append(filtered_paper)
    print(f"Filtered {len(filtered_papers)} papers out of {len(papers)}")
    return filtered_papers


def save_checkpoint(
    processed_papers,
    ignored_papers,
    output_file,
    ignored_file,
    temp_file,
    temp_ignored_file,
):
    """Save checkpoint for both processed and ignored papers"""
    # Save processed papers to temporary file first to avoid corruption if interrupted
    with open(temp_file, "w") as f:
        json.dump(processed_papers, f, indent=4)

    # Save ignored papers to temporary file
    with open(temp_ignored_file, "w") as f:
        json.dump(ignored_papers, f, indent=4)

    # Then rename to the actual output files
    if os.path.exists(temp_file):
        shutil.move(temp_file, output_file)

    if os.path.exists(temp_ignored_file):
        shutil.move(temp_ignored_file, ignored_file)


def get_openreview_metadata(paper, openreview_id):
    """
    Extract TLDR and abstract from OpenReview for papers missing these fields.

    Args:
        paper (dict): The paper dictionary
        openreview_id (str): The OpenReview ID for the paper

    Returns:
        dict: Updated paper with openreview_id, tldr and abstract if available
    """
    updated_paper = paper.copy()

    # Always add openreview_id to the paper
    updated_paper["openreview_id"] = openreview_id

    # Check if both tldr and abstract are already present
    if "tldr" in updated_paper and "abstract" in updated_paper and updated_paper["tldr"] and updated_paper["abstract"]:
        print(f"TLDR and abstract already exist for paper {paper.get('id')}")
        return updated_paper

    # Fields to extract from OpenReview
    missing_fields = []
    if "tldr" not in updated_paper or not updated_paper.get("tldr"):
        missing_fields.append("tldr")
    if "abstract" not in updated_paper or not updated_paper.get("abstract"):
        missing_fields.append("abstract")

    print(f"Extracting from OpenReview: {', '.join(missing_fields)}")

    try:
        # Call extract_from_openreview to get the missing metadata
        metadata = extract_from_openreview(openreview_id, quiet=True)

        # Update the paper with the extracted metadata
        if "tldr" in missing_fields and "tldr" in metadata:
            updated_paper["tldr"] = metadata["tldr"]

        if "abstract" in missing_fields and "abstract" in metadata:
            updated_paper["abstract"] = metadata["abstract"]

        return updated_paper
    except Exception as e:
        print(f"Error extracting metadata from OpenReview: {str(e)}")
        return updated_paper


# Add a new function to process a single paper
def process_paper(paper, i, total_papers, slides_dir, processed_ids, ignored_ids):
    """Process a single paper to extract metadata"""
    result = {"processed": None, "ignored": None}
    paper_id = paper.get("id")

    # Skip already processed papers
    if paper_id in processed_ids:
        print(f"[{i + 1}/{total_papers}] ⏩ Already processed")
        return result

    # Skip papers that should be ignored
    if paper_id in ignored_ids:
        print(f"[{i + 1}/{total_papers}] ⏩ Ignoring (previously failed)")
        return result

    if not paper.get("site"):
        print(f"[{i + 1}/{total_papers}] ❌ No site found")
        # Add to ignored papers
        result["ignored"] = paper
        return result

    # Check missing fields
    missing_slideslive = "slideslive_id" not in paper or not paper.get("slideslive_id")
    missing_openreview = "openreview_id" not in paper or not paper.get("openreview_id")
    missing_tldr = "tldr" not in paper or not paper.get("tldr")
    missing_abstract = "abstract" not in paper or not paper.get("abstract")

    # Extract IDs if needed
    if missing_slideslive or missing_openreview:
        slideslive_id, openreview_id = extract_slideslive_openreview_id(paper["site"], quiet=True)

        if missing_slideslive and not slideslive_id:
            print(f"[{i + 1}/{total_papers}] ❌ No SlidesLive ID found")
            # Add to ignored papers
            result["ignored"] = paper
            return result

        if missing_openreview and not openreview_id:
            print(f"[{i + 1}/{total_papers}] ❌ No OpenReview ID found")
            # Add to ignored papers
            result["ignored"] = paper
            return result

    # Process SlidesLive metadata if missing
    if missing_slideslive:
        try:
            slides_dir_result = extract_from_slideslive(slideslive_id, slides_dir=slides_dir, quiet=True)
            if slides_dir_result:
                paper["slideslive_id"] = slideslive_id
                # print(f"[{i + 1}/{total_papers}] ✅ Added SlidesLive metadata: {slideslive_id}")
            else:
                print(f"[{i + 1}/{total_papers}] ❌ No slide metadata")
                # Add to ignored papers
                result["ignored"] = paper
                return result
        except Exception as e:
            print(f"[{i + 1}/{total_papers}] ❌ Error extracting SlidesLive metadata: {str(e)}")
            # Add to ignored papers
            result["ignored"] = paper
            return result

    # Process OpenReview metadata if missing openreview_id, tldr, or abstract
    if missing_openreview or missing_tldr or missing_abstract:
        if missing_openreview:  # We need to get the ID first
            openreview_id = openreview_id  # Use the one extracted earlier
            paper["openreview_id"] = openreview_id
            # print(f"[{i + 1}/{total_papers}] ✅ Added OpenReview ID: {openreview_id}")
        else:
            openreview_id = paper["openreview_id"]  # Use existing ID

        # Extract tldr and abstract if missing
        if missing_tldr or missing_abstract:
            try:
                tldr, abstract = extract_from_openreview(openreview_id, quiet=True)

                if tldr is not None:
                    paper["tldr"] = tldr
                    # print(f"[{i + 1}/{total_papers}] ✅ Added TLDR")
                else:
                    print(f"[{i + 1}/{total_papers}] ❌ No TLDR found")
                    # Add to ignored papers
                    result["ignored"] = paper
                    return result

                if abstract is not None:
                    paper["abstract"] = abstract
                    # print(f"[{i + 1}/{total_papers}] ✅ Added abstract")
                else:
                    print(f"[{i + 1}/{total_papers}] ❌ No abstract found")
                    # Add to ignored papers
                    result["ignored"] = paper
                    return result

            except Exception as e:
                print(f"[{i + 1}/{total_papers}] ❌ Error extracting OpenReview metadata: {str(e)}")
                result["ignored"] = paper
                return result

    # Mark as processed
    result["processed"] = paper
    print(f"[{i + 1}/{total_papers}] ✅ Paper {paper_id} processed; all required fields added")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a paperlist")
    parser.add_argument(
        "--paperlist",
        type=str,
        default="paperlist/nips2024.json",
        help="Path to the paperlist",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Path to the output directory",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of worker processes (default: number of CPU cores)",
    )
    args = parser.parse_args()

    # Use number of CPU cores if num_workers not specified
    if args.num_workers is None:
        # Use 75% of available cores to avoid overwhelming the system
        # args.num_workers = max(1, int(multiprocessing.cpu_count() * 0.75))
        args.num_workers = 8  # set to 8 for conservative use

    print(f"Using {args.num_workers} worker processes")

    # get file name from path
    file_name = os.path.basename(args.paperlist).split(".")[0]
    # print(f"Processing {file_name}")

    # filter required fields
    filtered_papers = filter_required_fields(args.paperlist)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize empty lists to store processed and ignored papers
    processed_papers = []
    ignored_papers = []

    # Define checkpoint frequency (save every N papers)
    checkpoint_frequency = 10
    output_file = f"{args.output_dir}/paperlist_{file_name}.json"
    ignored_file = f"{args.output_dir}/ignored_papers_{file_name}.json"
    temp_file = f"{args.output_dir}/paperlist_temp_{file_name}.json"
    temp_ignored_file = f"{args.output_dir}/ignored_papers_temp_{file_name}.json"

    # Load any existing processed papers if the file exists
    try:
        with open(output_file, "r") as f:
            all_papers = json.load(f)

            # filter out papers that have all required fields and not empty
            processed_papers = [
                paper
                for paper in all_papers
                if all(paper.get(field) for field in ["slideslive_id", "openreview_id", "tldr", "abstract"])
            ]
            print(f"Loaded {len(processed_papers)} previously processed papers with all required fields")
    except FileNotFoundError:
        processed_papers = []

    # Load any existing ignored papers if the file exists
    try:
        with open(ignored_file, "r") as f:
            ignored_papers = json.load(f)
            print(f"Loaded {len(ignored_papers)} previously ignored papers")
    except FileNotFoundError:
        ignored_papers = []

    # Track which papers have already been processed or should be ignored
    processed_ids = {paper.get("id") for paper in processed_papers}
    ignored_ids = {paper.get("id") for paper in ignored_papers}

    # Create a partial function with common arguments
    process_paper_with_args = partial(
        process_paper,
        total_papers=len(filtered_papers),
        slides_dir=args.output_dir,
        processed_ids=processed_ids,
        ignored_ids=ignored_ids,
    )

    # Process papers in parallel
    new_processed = []
    new_ignored = []
    processed_count = 0

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit all papers for processing
        future_to_paper = {
            executor.submit(process_paper_with_args, paper, i): (paper, i)
            for i, paper in enumerate(filtered_papers)
            if paper.get("id") not in processed_ids and paper.get("id") not in ignored_ids
        }

        # Process results as they complete
        for future in future_to_paper:
            try:
                result = future.result()
                if result["processed"]:
                    new_processed.append(result["processed"])
                if result["ignored"]:
                    new_ignored.append(result["ignored"])

                # Save checkpoint periodically
                processed_count += 1
                if processed_count % checkpoint_frequency == 0:
                    all_processed = processed_papers + new_processed
                    all_ignored = ignored_papers + new_ignored
                    save_checkpoint(
                        all_processed,
                        all_ignored,
                        output_file,
                        ignored_file,
                        temp_file,
                        temp_ignored_file,
                    )
                    print(
                        f"Checkpoint saved: {len(all_processed)} papers processed, {len(all_ignored)} papers ignored so far\n"
                    )
            except Exception as e:
                paper, i = future_to_paper[future]
                print(f"[{i + 1}/{len(filtered_papers)}] ❌ Error processing paper {paper.get('id')}: {str(e)}")
                new_ignored.append(paper)

    # Update the processed and ignored lists with new results
    processed_papers.extend(new_processed)
    ignored_papers.extend(new_ignored)

    # Write final results to the output files
    save_checkpoint(
        processed_papers,
        ignored_papers,
        output_file,
        ignored_file,
        temp_file,
        temp_ignored_file,
    )

    print(f"Processed {len(processed_papers)} papers with metadata out of {len(filtered_papers)}")
    print(f"Ignored {len(ignored_papers)} papers that couldn't be processed")
