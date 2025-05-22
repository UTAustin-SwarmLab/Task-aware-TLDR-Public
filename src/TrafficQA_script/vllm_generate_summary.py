"""Run this script to generate a TL;DR for each video in the paper dataset using the VLLM model.
CUDA_VISIBLE_DEVICES=0,1 python src/TrafficQA_script/vllm_generate_summary.py
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
from src.TrafficQA_script.trafficQA_jsonl_reader import load_data


@hydra.main(version_base=None, config_path="../../config", config_name="TrafficQA")
def GenerateVideoTLDR(cfg: DictConfig) -> None:
    """ "
    Generate a TLDR for each video in the TrafficQA dataset using the VLLM model."
    """
    ds_cfg = cfg.TrafficQA
    video_root = Path(ds_cfg.dataset_root, "raw_videos")
    csv_root = Path(ds_cfg.dataset_root, "csv")
    csv_root.mkdir(parents=True, exist_ok=True)
    # load the metadata
    videoid2metadata = load_data(ds_cfg)
    random.seed(ds_cfg.seed)

    # sampled_ids = random.sample(list(videoid2metadata.keys()), ds_cfg.N)
    selected_df = pd.read_csv("./user_study/trafficqa/data/sampled_stimulus_100.csv")
    sampled_ids = selected_df["video_id"].unique()

    # check if the csv file already exists
    result_csv = None
    result_file_path = Path(csv_root, "vlm_responses_new4.csv")
    if result_file_path.exists():
        print("===================================================================================================")
        print("File vlm_responses_new4.csv already exists. Loading...")
        result_csv = pd.read_csv(result_file_path, delimiter="|")
        print("===================================================================================================")
    else:
        # create a csv file to save the results
        with open(result_file_path, "w") as f:
            # Write header only when creating the file
            f.write("video_id|vid_filename|response|temperature\n")

    vllm_model = VLlmClient(cfg.vllm)
    vllm_model.MODEL = ""
    vllm_model.SYSTEM_MESSAGE = ""
    vllm_model.COT_PROMPT = ""

    # predict the masked tokens from the video
    for sampled_id in tqdm(sampled_ids, desc="Generating TLDR"):
        video_data = videoid2metadata[sampled_id]
        record = video_data[0]  # Only take the first record for TLDR generation
        vid_filename = record["filename"]
        questions = []
        for idx, row in selected_df[selected_df["video_id"] == sampled_id].iterrows():
            questions.append(row["question"])
        prompt = """Summarize the video in 2–3 sentences, focusing on the main events, any potential accidents, and the vehicles involved. Write in a natural, flowing style that helps readers understand the content without directly listing answers to the following questions: {}""".format(
            ", ".join(questions)
        )
        print("vid_filename: ", vid_filename, "Video ID: ", sampled_id)
        print(f"Prompt: {prompt}")

        # get more diversity in the TLDR
        for temperature in ds_cfg.temperatures:
            if result_csv is not None and len(result_csv) > 0:
                if (
                    int(sampled_id) in result_csv["video_id"].values
                    and temperature in result_csv[result_csv["video_id"] == int(sampled_id)]["temperature"].values
                ):
                    print("=================================================================================")
                    print(f"Vid: {sampled_id}, Temperature: {temperature} already processed. Skipping...")
                    print("=================================================================================")
                    continue
            response = vllm_model.prompt_video(
                prompt=prompt,
                path=str(Path(video_root, vid_filename)),
                temperature=temperature,
                logprobs=False,
                top_logprobs=1,
                top_k=50,
                top_p=1,
            ).text.replace("\n", " ")

            print(f"Vid_filename: {vid_filename}")
            print(f"Response: {response}")
            print("==================================================================================")

            # save the results in a csv file
            with open(result_file_path, "a") as f:
                f.write(f"{sampled_id}|{vid_filename}|{response}|{temperature}\n")


@hydra.main(version_base=None, config_path="../../config", config_name="TrafficQA")
def GenerateVideoCoT(cfg: DictConfig) -> None:
    """ "
    Generate a CoT for each video in TrafficQA paper dataset using the VLLM model."
    """
    ds_cfg = cfg.TrafficQA
    video_root = Path(ds_cfg.dataset_root, "raw_videos")
    csv_root = Path(ds_cfg.dataset_root, "csv")
    csv_root.mkdir(parents=True, exist_ok=True)

    # load the metadata
    videoid2metadata = load_data(ds_cfg)
    random.seed(ds_cfg.seed)
    sampled_ids = random.sample(list(videoid2metadata.keys()), ds_cfg.N)

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
            f.write("video_id|record_id|vid_filename|q_type|question|vlm_CoT|vlm_reasoning|vlm_answer|answer\n")

    vllm_model = VLlmClient(cfg.vllm)
    vllm_model.MODEL = ""
    vllm_model.SYSTEM_MESSAGE = ""
    vllm_model.COT_PROMPT = ""

    # predict the masked tokens from the video
    for sampled_id in tqdm(sampled_ids, desc="CoT generation"):
        video_data = videoid2metadata[sampled_id]
        for record in video_data:
            vid_filename = record["filename"]
            record_id = record["record_id"]
            q_type = record["q_type"]
            question = record["question"]
            answer = record["answer"]
            options = [record["option0"], record["option1"], record["option2"], record["option3"]]
            if result_csv is not None:
                if int(record_id) in result_csv["record_id"].values:
                    print("==========================================================================================")
                    print(f"Record ID {record_id} already processed. Skipping...")
                    print("==========================================================================================")
                    continue

            options_char = [f"({chr(65 + i)}) {option}" for i, option in enumerate(options)]
            prompt = f"""This is a video of an accident. Answer the question from the given options:
                Question: {question} Options: {options_char}.
                Begin by reasoning your thoughts step-by-step in a chain-of-thought manner. Finally, provide the answer in the format:
                Template:
                <CoT> Your reasoning
                <Answer> Your choice in the format (A), (B), (C), (D)."""
            # print(f"Prompt: {prompt}")

            # get CoT from vllm
            CoT = (
                vllm_model.prompt_video(
                    prompt=prompt,
                    path=str(Path(video_root, vid_filename)),
                )
                .text.replace("\n", " ")
                .replace("\t", " ")
                .replace("\r", " ")
            )

            reason = CoT.split("<CoT>")[-1].split("<Answer>")[0].strip()
            pred_answer = CoT.split("<Answer>")[-1].strip()

            print(f"Record ID: {record_id}, vid_filename: {vid_filename}")
            print(f"Question: {question}")
            print(f"CoT: {CoT}")
            print(f"Reason: {reason}")
            print(f"Predicted Answer: {pred_answer}")
            print(f"Answer: {answer}")
            print("===================================================================================================")

            # save the results in a csv file
            with open(result_file_path, "a") as f:
                f.write(
                    f"{sampled_id}|{record_id}|{vid_filename}|{q_type}|{question}|{CoT}|{reason}|{pred_answer}|{answer}\n"
                )


if __name__ == "__main__":
    # GenerateVideoCoT()
    GenerateVideoTLDR()
