"""Run this script to generate a TL;DR for each video in the paper dataset using the VLLM model.
CUDA_VISIBLE_DEVICES=1 python src/AIConfVideo_script/vllm_generate_tldr.py
"""

import os
import random
import sys
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))
from src.AIConfVideo_script.paper_ib_score import GetMetaData
from src.client.vllm_client import VLlmClient
from src.utils.primary_area import subarea_map


@hydra.main(version_base=None, config_path="../../config", config_name="AIVideoConf")
def GenerateVideoTLDR(cfg: DictConfig) -> None:
    """ "
    Generate a TL;DR for each video in the paper dataset using the VLLM model."
    """
    ds_cfg = cfg.AIConfVideo
    video_root = Path(ds_cfg.paperlist_root, "video")
    # get all the video directories
    random.seed(ds_cfg.seed)
    video_dirs = random.sample([d for d in video_root.iterdir() if d.is_dir()], k=ds_cfg.SAMPLE_VIDEO_NUM)

    # get metadata
    metadata = GetMetaData(ds_cfg)
    video_id2openreview_id = {}

    # check if the csv file already exists
    result_csv = None
    result_file_path = Path(video_root, "vlm_response.csv")
    if result_file_path.exists():
        print("===================================================================================================")
        print("File vlm_response.csv already exists. Loading...")
        result_csv = pd.read_csv(Path(video_root, "vlm_response.csv"), delimiter="|")
        print("===================================================================================================")
    else:
        # create a csv file to save the results
        with open(result_file_path, "w") as f:
            # Write header only when creating the file
            f.write("video_id|openreview_id|task_aware_TLDR|task_agnostic_TLDR\n")

    vllm_model = VLlmClient(cfg.vllm)
    vllm_model.MODEL = ""
    vllm_model.SYSTEM_MESSAGE = ""
    vllm_model.COT_PROMPT = ""

    # predict the masked tokens from the video
    for video_dir in tqdm(video_dirs, desc="Generating TLDR"):
        video_id = video_dir.name
        if video_id not in metadata.keys():
            print("===================================================================================================")
            print(f"Video ID {video_id} not found in metadata. Skipping...")
            print("===================================================================================================")
            continue
        if result_csv is not None:
            if int(video_id) in result_csv["video_id"].values:
                print(
                    "==================================================================================================="
                )
                print(f"Video ID {video_id} already processed. Skipping...")
                print(
                    "==================================================================================================="
                )
                continue
        if not any(f.suffix.lower() == ".png" for f in video_dir.iterdir()):
            print("=========================================================================================")
            print(f"Video ID {video_id} has no PNG image files. Skipping...")
            print("=========================================================================================")
            continue
        openreview_id = metadata[video_id]["openreview_id"]
        video_id2openreview_id[video_id] = openreview_id
        openreview_id = video_id2openreview_id[video_id]

        # get task-aware tldr
        task_aware_TLDR = vllm_model.prompt_video(
            prompt="You are viewing slides from a paper presentation. Generate a concise one-sentence TL;DR based on the content so that the reader of the TL;DR can infer the primary area of the paper.",
            path=str(Path(video_root, video_id)),
        ).text

        # get task-agnostic tldr
        task_agnostic_TLDR = vllm_model.prompt_video(
            prompt="You are viewing slides from a paper presentation. Generate a concise one-sentence TL;DR based on the content.",
            path=str(Path(video_root, video_id)),
        ).text

        print(f"Video ID: {video_id}, OpenReview ID: {openreview_id}")
        print(f"Task-aware TLDR: {task_aware_TLDR}")
        print(f"Task-agnostic TLDR: {task_agnostic_TLDR}")
        print("===================================================================================================")

        # save the results in a csv file
        with open(result_file_path, "a") as f:
            f.write(f"{video_id}|{openreview_id}|{task_aware_TLDR}|{task_agnostic_TLDR}\n")


@hydra.main(version_base=None, config_path="../../config", config_name="AIVideoConf")
def GenerateVideoCoT(cfg: DictConfig) -> None:
    """ "
    Generate a CoT for each video in the paper dataset using the VLLM model."
    """
    ds_cfg = cfg.AIConfVideo
    video_root = Path(ds_cfg.paperlist_root, "video")
    # get all the video directories
    random.seed(ds_cfg.seed)
    video_dirs = random.sample([d for d in video_root.iterdir() if d.is_dir()], k=ds_cfg.SAMPLE_VIDEO_NUM)

    # get metadata
    metadata = GetMetaData(ds_cfg)
    video_id2openreview_id = {}

    # check if the csv file already exists
    result_csv = None
    result_file_path = Path(video_root, "vlm_CoT.csv")
    if result_file_path.exists():
        print("===================================================================================================")
        print("File vlm_CoT.csv already exists. Loading...")
        result_csv = pd.read_csv(Path(video_root, "vlm_CoT.csv"), delimiter="|")
        print("===================================================================================================")
    else:
        # create a csv file to save the results
        with open(result_file_path, "w") as f:
            # Write header only when creating the file
            f.write("video_id|openreview_id|vlm_CoT|vlm_reasoning|vlm_answer\n")

    vllm_model = VLlmClient(cfg.vllm)
    vllm_model.MODEL = ""
    vllm_model.SYSTEM_MESSAGE = ""
    vllm_model.COT_PROMPT = ""

    # generate options for the task
    OPTIONS = sorted(list(set(subarea_map.values())))
    options = " ".join([f"({chr(65 + i)}) {option}" for i, option in enumerate(OPTIONS)])

    # predict the masked tokens from the video
    for video_dir in tqdm(video_dirs, desc="Generating CoT"):
        video_id = video_dir.name
        if video_id not in metadata.keys():
            print("===================================================================================================")
            print(f"Video ID {video_id} not found in metadata. Skipping...")
            print("===================================================================================================")
            continue
        if result_csv is not None:
            if int(video_id) in result_csv["video_id"].values:
                print(
                    "==================================================================================================="
                )
                print(f"Video ID {video_id} already processed. Skipping...")
                print(
                    "==================================================================================================="
                )
                continue
        if not any(f.suffix.lower() == ".png" for f in video_dir.iterdir()):
            print("=========================================================================================")
            print(f"Video ID {video_id} has no PNG image files. Skipping...")
            print("=========================================================================================")
            continue
        openreview_id = metadata[video_id]["openreview_id"]
        video_id2openreview_id[video_id] = openreview_id
        openreview_id = video_id2openreview_id[video_id]

        # get CoT from vllm
        CoT = vllm_model.prompt_video(
            prompt=f"""You are viewing slides from a paper presentation. Determine the most suitable area for this paper from the given options: {options}.
            Begin by reasoning your thoughts step-by-step in a chain-of-thought manner. Finally, provide the answer in the format:
            Template:
            <CoT> Your reasoning
            <Answer> Your choice in the format (A), (B), (C), (D), etc.""",
            path=str(Path(video_root, video_id)),
        ).text.replace("\n", " ")

        reason = CoT.split("<CoT>")[-1].split("<Answer>")[0].strip()
        answer = CoT.split("<Answer>")[-1].strip()

        print(f"Video ID: {video_id}, OpenReview ID: {openreview_id}")
        print(f"CoT: {CoT}")
        print(f"Reason: {reason}")
        print(f"Answer: {answer}")
        print("===================================================================================================")

        # save the results in a csv file
        with open(result_file_path, "a") as f:
            f.write(f"{video_id}|{openreview_id}|{CoT}|{reason}|{answer}\n")


@hydra.main(version_base=None, config_path="../../config", config_name="AIVideoConf")
def GenerateVideoMore(cfg: DictConfig) -> None:
    """ "
    Generate a TL;DR for each video in the paper dataset using the VLLM model."
    """
    ds_cfg = cfg.AIConfVideo
    video_root = Path(ds_cfg.paperlist_root, "video")
    # get all the video directories
    random.seed(ds_cfg.seed)
    video_dirs = random.sample([d for d in video_root.iterdir() if d.is_dir()], k=ds_cfg.SAMPLE_VIDEO_NUM)

    N = ds_cfg.div_response
    # get metadata
    metadata = GetMetaData(ds_cfg)
    video_id2openreview_id = {}

    # check if the csv file already exists
    result_csv = None
    result_file_path = Path(video_root, "vlm_respoonse_more.csv")
    if result_file_path.exists():
        print("===================================================================================================")
        print("File vlm_respoonse_more.csv already exists. Loading...")
        result_csv = pd.read_csv(Path(video_root, "vlm_respoonse_more.csv"), delimiter="|")
        print("===================================================================================================")
    else:
        # create a csv file to save the results
        with open(result_file_path, "w") as f:
            # Write header only when creating the file
            f.write("video_id|openreview_id|response|temperature\n")

    vllm_model = VLlmClient(cfg.vllm)
    vllm_model.MODEL = ""
    vllm_model.SYSTEM_MESSAGE = ""
    vllm_model.COT_PROMPT = ""

    # predict the masked tokens from the video
    for video_dir in tqdm(video_dirs, desc="Generating TLDR"):
        video_id = video_dir.name
        if video_id not in metadata.keys():
            print("===================================================================================================")
            print(f"Video ID {video_id} not found in metadata. Skipping...")
            print("===================================================================================================")
            continue
        if result_csv is not None:
            if int(video_id) in result_csv["video_id"].values:
                print(
                    "==================================================================================================="
                )
                print(f"Video ID {video_id} already processed. Skipping...")
                print(
                    "==================================================================================================="
                )
                continue
        if not any(f.suffix.lower() == ".png" for f in video_dir.iterdir()):
            print("=========================================================================================")
            print(f"Video ID {video_id} has no PNG image files. Skipping...")
            print("=========================================================================================")
            continue
        openreview_id = metadata[video_id]["openreview_id"]
        video_id2openreview_id[video_id] = openreview_id
        openreview_id = video_id2openreview_id[video_id]

        print(f"Video ID: {video_id}, OpenReview ID: {openreview_id}")

        # get more diversity in the TLDR
        responses = []
        for i in range(N):
            temperature = 1 + 0.25 * i
            # get task-agnostic tldr
            response = vllm_model.prompt_video(
                prompt="You are viewing slides from a paper presentation. Generate a concise one-sentence TL;DR based on the content so that the reader of the TL;DR can infer the primary area of the paper.",
                path=str(Path(video_root, video_id)),
                temperature=temperature,
                logprobs=False,
                top_logprobs=1,
                top_k=50,
                top_p=1,
            ).text
            responses.append(response)

            print(f"Responses: {temperature} | {response}")
            print("===================================================================================================")

            # save the results in a csv file
            with open(result_file_path, "a") as f:
                f.write(f"{video_id}|{openreview_id}|{response}|{temperature}\n")


if __name__ == "__main__":
    GenerateVideoTLDR()
    GenerateVideoCoT()
    GenerateVideoMore()
