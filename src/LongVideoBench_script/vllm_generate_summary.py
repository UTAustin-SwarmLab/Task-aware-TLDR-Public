"""Run this script to generate a TL;DR for each video in the paper dataset using the VLLM model.
CUDA_VISIBLE_DEVICES=1,2 python src/LongVideoBench_script/vllm_generate_summary.py
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
from src.client.vllm_client import VLlmClient
from src.LongVideoBench_script.load_data import LoadLongVideoBench


@hydra.main(version_base=None, config_path="../../config", config_name="LongVideoBench")
def GenerateVideoTLDR(cfg: DictConfig) -> None:
    """ "
    Generate a TLDR for each video in the MVBench using the VLLM model."
    """
    ds_cfg = cfg.LongVideoBench
    csv_root = Path(ds_cfg.dataset_root, "csv")
    csv_root.mkdir(parents=True, exist_ok=True)
    # load the metadata
    id2idx, metadata_list, ds = LoadLongVideoBench(cfg)
    random.seed(ds_cfg.seed)

    # check if the csv file already exists
    result_csv = None
    result_file_path = Path(csv_root, "vlm_responses.csv")
    if result_file_path.exists():
        print("===================================================================================================")
        print("File vlm_responses.csv already exists. Loading...")
        result_csv = pd.read_csv(result_file_path, delimiter="|")
        print("===================================================================================================")
    else:
        # create a csv file to save the results
        with open(result_file_path, "w") as f:
            # Write header only when creating the file
            f.write("video_id|record_id|response|temperature\n")

    vllm_model = VLlmClient(cfg.vllm)
    vllm_model.MODEL = ""
    vllm_model.SYSTEM_MESSAGE = ""
    vllm_model.COT_PROMPT = ""

    # predict the masked tokens from the video
    for idx in tqdm(id2idx.values(), desc="Generating TLDR"):
        metadata = metadata_list[idx]
        video_id = metadata["video_id"]
        record_id = metadata["id"]
        question = metadata["question"]
        prompt = """Summarize a TLDR of the video in 1-2 sentences to help readers understand the content without directly listing answers to the following question: {}""".format(
            question
        )
        print("Video ID: ", video_id, "Record ID: ", record_id)
        print(f"Prompt: {prompt}")
        inputs = ds[idx]["inputs"]

        # get more diversity in the TLDR
        for temperature in ds_cfg.temperatures:
            if result_csv is not None and len(result_csv) > 0:
                if (
                    int(record_id) in result_csv["record_id"].values
                    and temperature in result_csv[result_csv["record_id"] == int(record_id)]["temperature"].values
                ):
                    print("=================================================================================")
                    print(f"Record ID: {record_id}, Temperature: {temperature} already processed. Skipping...")
                    print("=================================================================================")
                    continue
            response = vllm_model.prompt_video(
                prompt=prompt,
                path=inputs,
                temperature=temperature,
                logprobs=False,
                top_logprobs=1,
                top_k=50,
                top_p=1,
            ).text.replace("\n", " ")

            print(f"Record ID: {record_id}")
            print(f"Response: {response}")
            print("==================================================================================")

            # save the results in a csv file
            with open(result_file_path, "a") as f:
                f.write(f"{video_id}|{record_id}|{response}|{temperature}\n")


@hydra.main(version_base=None, config_path="../../config", config_name="LongVideoBench")
def GenerateVideoCoT(cfg: DictConfig) -> None:
    """ "
    Generate a CoT for each video in MVBench dataset using the VLLM model."
    """
    ds_cfg = cfg.LongVideoBench
    csv_root = Path(ds_cfg.dataset_root, "csv")
    csv_root.mkdir(parents=True, exist_ok=True)
    # load the metadata
    id2idx, metadata_list, ds = LoadLongVideoBench(cfg)
    random.seed(ds_cfg.seed)

    # check if the csv file already exists
    result_csv = None
    result_file_path = Path(csv_root, "vlm_CoT.csv")
    if result_file_path.exists():
        print("===================================================================================================")
        print("File vlm_CoT.csv already exists. Loading...")
        result_csv = pd.read_csv(result_file_path, delimiter="|")
        print("===================================================================================================")
    else:
        # create a csv file to save the results
        with open(result_file_path, "w") as f:
            # Write header only when creating the file
            f.write(
                "video_id|record_id|question_category|topic_category|question|vlm_CoT|vlm_reasoning|vlm_answer|answer\n"
            )

    vllm_model = VLlmClient(cfg.vllm)
    vllm_model.MODEL = ""
    vllm_model.SYSTEM_MESSAGE = ""
    vllm_model.COT_PROMPT = ""

    # predict the masked tokens from the video
    for idx in tqdm(id2idx.values(), desc="Generating TLDR"):
        metadata = metadata_list[idx]
        video_id = metadata["video_id"]
        record_id = metadata["id"]
        question = metadata["question"]
        question_category = metadata["question_category"]
        topic_category = metadata["topic_category"]
        options = metadata["candidates"]
        options_char = [f"({chr(65 + i)}) {option}" for i, option in enumerate(options)]
        # find the index of the answer in the options
        answer_idx = metadata["correct_choice"]
        answer_char = chr(65 + answer_idx)
        inputs = ds[idx]["inputs"]

        if result_csv is not None:
            if record_id in result_csv["record_id"].values:
                print("==============================================================================================")
                print(f"Record ID {record_id} already processed. Skipping...")
                print("==============================================================================================")
                continue
        prompt = f"""This is a video clip. Answer the question from the given options:
Question: {question} Options: {options_char}.
Begin by reasoning your thoughts step-by-step in a chain-of-thought manner. Finally, provide the answer in the format:
Template:
<CoT> Your reasoning
<Answer> Your choice in the format (A), (B), (C), (D)."""
        print("Record_id: ", record_id, "Video ID: ", video_id)
        print(f"Prompt: {prompt}")

        # get CoT from vllm
        CoT = (
            vllm_model.prompt_video(
                prompt=prompt,
                path=inputs,
            )
            .text.replace("\n", " ")
            .replace("\t", " ")
            .replace("\r", " ")
        )

        reason = CoT.split("<CoT>")[-1].split("<Answer>")[0].strip()
        pred_answer = CoT.split("<Answer>")[-1].strip()

        print("Record_id: ", record_id, "Video ID: ", video_id)
        print(f"Question: {question}")
        print(f"Reason: {reason}")
        print(f"Predicted Answer: {pred_answer}")
        print(f"Answer: {answer_char}")
        print("===================================================================================================")

        # save the results in a csv file
        with open(result_file_path, "a") as f:
            f.write(
                f"{video_id}|{record_id}|{question_category}|{topic_category}|{question}|{CoT}|{reason}|{pred_answer}|{answer_char}\n"
            )


if __name__ == "__main__":
    GenerateVideoCoT()
    GenerateVideoTLDR()
