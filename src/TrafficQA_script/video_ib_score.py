"""Run IB score on traffic accident videos. To run:
CUDA_VISIBLE_DEVICES=0,1 python src/TrafficQA_script/video_ib_score.py
"""

import json
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
from src.utils.fill_in_mask import FillMaskText2LogProb, MaskedVideo2LogProb


def GetResponsesAndKeywords(ds_cfg: DictConfig) -> dict:
    """Get the vlm responses from the dataset (vlm responses).
    Returns:
        dataframe: A dataframe where each row is a vlm response.
        dict: A dictionary where the key is the video_id + record_id/temperature and the value is the keywords.
    """
    return (
        pd.read_csv(Path(ds_cfg.dataset_root, "csv", "vlm_CoT.csv"), delimiter="|"),
        json.load(open(Path(ds_cfg.dataset_root, "json", "CoT_keywords.json"))),
        pd.read_csv(Path(ds_cfg.dataset_root, "csv", "vlm_responses_new4.csv"), delimiter="|"),
        json.load(open(Path(ds_cfg.dataset_root, "json", "vlm_response_keywords_new4.json"))),
    )


def FilterWrongQA(CoT_response_csv: pd.DataFrame, videoid2metadata: dict) -> dict:
    """Filter the wrong QA pairs from the vlm responses. Only keep the correct ones answered by the VLM.
    Args:
        CoT_response_csv (pd.dataframe): A dataframe where each row is a vlm response.
        videoid2metadata (dict): A dictionary where the key is the video_id and the value is the metadata.
    Returns:
        dict: A dictionary where the key is the video_id and the value is a list of metadata.
    """
    correct_videoid2metadata = {}
    for vide_id, metadata_list in videoid2metadata.items():
        for metadata in metadata_list:
            record_id = metadata["record_id"]
            if int(record_id) not in CoT_response_csv["record_id"].values:
                print(f"Record ID {record_id} not found in vlm responses. Skipping...")
                continue

            answer = CoT_response_csv.loc[CoT_response_csv["record_id"] == record_id, "answer"].values[0]
            vlm_answer = CoT_response_csv.loc[CoT_response_csv["record_id"] == record_id, "vlm_answer"].values[0]

            if vlm_answer.strip("(")[0] == chr(65 + int(answer)):
                if record_id == "53992" or record_id == 53992:
                    print(answer, vlm_answer)
                    input()
                if vide_id not in correct_videoid2metadata.keys():
                    correct_videoid2metadata[vide_id] = []
                correct_videoid2metadata[vide_id].append(
                    {
                        "record_id": metadata["record_id"],
                        "filename": metadata["filename"],
                        "q_type": metadata["q_type"],
                        "question": metadata["question"],
                        "option0": metadata["option0"],
                        "option1": metadata["option1"],
                        "option2": metadata["option2"],
                        "option3": metadata["option3"],
                        "answer": metadata["answer"],
                    }
                )
    return correct_videoid2metadata


@hydra.main(version_base=None, config_path="../../config", config_name="TrafficQA")
def TextAndTask(cfg: DictConfig):
    """Run IB score on traffic accident videos."""
    ds_cfg = cfg.TrafficQA
    video_root = Path(ds_cfg.dataset_root, "raw_videos")
    postfix = "" if "Qwen2.5-VL" in cfg.vllm.model_name else f"_{cfg.vllm.model_name.split('/')[-1]}"
    postfix += "_new4"
    random.seed(ds_cfg.seed)

    # get metadata
    CoT_response_csv, _, response_csv, _ = GetResponsesAndKeywords(ds_cfg)
    if ds_cfg.CORPUS_FROM == "CoT":
        response_csv = CoT_response_csv
    videoid2metadata = load_data(ds_cfg)
    correct_videoid2metadata = videoid2metadata

    # check if the csv file already exists
    result_csv = None
    if Path(ds_cfg.dataset_root, "csv", f"{ds_cfg.CORPUS_FROM}_mutual_info_text_task{postfix}.csv").exists():
        print("=========================================================================================")
        print(f"File {ds_cfg.CORPUS_FROM}_mutual_info_text_task{postfix}.csv already exists. Loading...")
        result_csv = pd.read_csv(
            Path(ds_cfg.dataset_root, "csv", f"{ds_cfg.CORPUS_FROM}_mutual_info_text_task{postfix}.csv"), delimiter="|"
        )
        print("=========================================================================================")
    else:
        # create a csv file to save the results
        with open(
            Path(ds_cfg.dataset_root, "csv", f"{ds_cfg.CORPUS_FROM}_mutual_info_text_task{postfix}.csv"), "w"
        ) as f:
            # Write header only when creating the file
            f.write(
                "video_id|record_id|vid_filename|q_type|group|predicted_answer_text|predicted_answer|correct_answer|informativeness_score|logprob_text|logprob|rank_text|rank\n"
            )

    # initiate a vllm model
    vllm_model = VLlmClient(cfg.vllm)
    selected_df = pd.read_csv("./user_study/trafficqa/data/sampled_stimulus_100.csv")

    # predict the masked tokens from the video
    for idx, row in tqdm(response_csv.iterrows(), desc="Predicting task answers"):
        video_id = row["video_id"]
        vid_filename = row["vid_filename"]
        group = f"temperature_{row['temperature']}" if ds_cfg.CORPUS_FROM == "response" else ds_cfg.CORPUS_FROM

        # if str(video_id) not in correct_videoid2metadata.keys():
        if video_id not in correct_videoid2metadata.keys():
            print(f"Video ID {video_id} not found in correct metadata. Skipping...")
            continue
        # for record in correct_videoid2metadata[video_id]:
        for _, ref_row in selected_df[selected_df["video_id"] == video_id].iterrows():
            q_type = ref_row["q_type"]
            record_id = ref_row["record_id"]
            question = ref_row["question"]
            answer_idx = ref_row["answer"]
            if ds_cfg.CORPUS_FROM == "CoT":
                if row["record_id"] != record_id:
                    continue
            print(f"Video ID: {video_id}")
            print(f"Record ID: {record_id}")
            print(f"Group: {group}")
            print(f"Question: {question}")
            print(f"Answer: {answer_idx}")
            print("=========================================================================================")

            if result_csv is not None:
                if (
                    int(ref_row["record_id"]) in result_csv["record_id"].values
                    and group in result_csv[result_csv["record_id"] == ref_row["record_id"]]["group"].values
                ):
                    print(("========================================================================================="))
                    print(f"Record ID {ref_row['record_id']} with group {group} already processed. Skip.")
                    print(("========================================================================================="))
                    continue

            options = [ref_row["option0"], ref_row["option1"], ref_row["option2"], ref_row["option3"]]
            options_with_alphabet = [chr(65 + i) + ": " + str(opt) + "." for i, opt in enumerate(options)]
            answer = chr(65 + answer_idx)  # A=0, B=1, C=2, D=3
            summary = row["response"] if ds_cfg.CORPUS_FROM == "response" else row["vlm_reasoning"]
            print(options_with_alphabet)
            print(answer)
            print(summary)

            # get P(Y|T, Vmask)
            pred_answer_text, logprob_text, rank_text = MaskedVideo2LogProb(
                orig_prompt=f"""Analyze the cropped video of a vehicle accident and the provided summary of the video. Summary: {summary} Now, answer the question: {question} Options: {" ".join(options_with_alphabet).replace("[", "").replace("]", "")} Answer with the corresponding letter: """,
                answer_choice=answer,
                video_path=Path(video_root, vid_filename),
                vllm_model=vllm_model,
                min_logprob=-10,
                crop=True,
            )

            # get P(Y|Vmask)
            pred_answer, logprob, rank = MaskedVideo2LogProb(
                orig_prompt=f"""Analyze the cropped video of a vehicle accident. Now, answer the question: {question} Options: {" ".join(options_with_alphabet).replace("[", "").replace("]", "")}
                Answer with the corresponding letter: """,
                answer_choice=answer,
                video_path=Path(video_root, vid_filename),
                vllm_model=vllm_model,
                min_logprob=-10,
                crop=True,
            )

            informativeness_score = logprob_text - logprob
            print(f"Video ID: {video_id}, Record ID: {record_id}")
            print(f"Pred answer with text: {pred_answer_text}, without {pred_answer} ; Correct answer: {answer}")
            print(f"Logprob_text: {logprob_text}, Logprob: {logprob}")
            print(f"Informativeness score: {informativeness_score}")
            print("=========================================================================================")

            # save the results in a csv file
            with open(
                Path(ds_cfg.dataset_root, "csv", f"{ds_cfg.CORPUS_FROM}_mutual_info_text_task{postfix}.csv"), "a"
            ) as f:
                f.write(
                    f"{video_id}|{record_id}|{vid_filename}|{q_type}|{group}|{pred_answer_text}|{pred_answer}|{answer}|{informativeness_score}|{logprob_text}|{logprob}|{rank_text}|{rank}\n"
                )
    return


@hydra.main(version_base=None, config_path="../../config", config_name="TrafficQA")
def VideoAndText(cfg: DictConfig):
    """Run IB score on traffic accident videos."""
    ds_cfg = cfg.TrafficQA
    video_root = Path(ds_cfg.dataset_root, "raw_videos")
    postfix = "" if "Qwen2.5-VL" in cfg.vllm.model_name else f"_{cfg.vllm.model_name.split('/')[-1]}"
    postfix += "_new4"

    # get metadata
    CoT_response_csv, CoT_keywords_dict, response_csv, keywords_dict = GetResponsesAndKeywords(ds_cfg)
    if ds_cfg.CORPUS_FROM == "CoT":
        response_csv = CoT_response_csv
        keywords_dict = CoT_keywords_dict
    videoid2metadata = load_data(ds_cfg)
    correct_videoid2metadata = videoid2metadata

    # check if the csv file already exists
    result_csv = None
    if Path(ds_cfg.dataset_root, "csv", f"{ds_cfg.CORPUS_FROM}_mutual_info_video_text{postfix}.csv").exists():
        print("=========================================================================================")
        print(f"File {ds_cfg.CORPUS_FROM}_mutual_info_video_text{postfix}.csv already exists. Loading...")
        result_csv = pd.read_csv(
            Path(ds_cfg.dataset_root, "csv", f"{ds_cfg.CORPUS_FROM}_mutual_info_video_text{postfix}.csv"), delimiter="|"
        )
        print("=========================================================================================")
    else:
        # create a csv file to save the results
        record_id = "" if ds_cfg.CORPUS_FROM == "response" else "record_id|"
        with open(
            Path(ds_cfg.dataset_root, "csv", f"{ds_cfg.CORPUS_FROM}_mutual_info_video_text{postfix}.csv"), "w"
        ) as f:
            # Write header only when creating the file
            f.write(
                f"video_id|{record_id}vid_filename|group|predicted_word_video|predicted_word|orig_word|logprob_video|logprob|diff|rank_video|rank\n"
            )

    # initiate a vllm model
    vllm_model = VLlmClient(cfg.vllm)

    if ds_cfg.CORPUS_FROM == "response":
        # predict the masked tokens from the video
        for idx, row in tqdm(response_csv.iterrows(), desc="Predicting masked tokens"):
            video_id = row["video_id"]
            vid_filename = row["vid_filename"]
            group = f"temperature_{row['temperature']}" if ds_cfg.CORPUS_FROM == "response" else ds_cfg.CORPUS_FROM
            orig_prompt = row["response"]
            keywords = keywords_dict[str(video_id)][str(row["temperature"])]["keywords"]

            # if str(video_id) not in correct_videoid2metadata.keys():
            if video_id not in correct_videoid2metadata.keys():
                print(f"Video ID {video_id} not found in correct metadata. Skipping...")
                continue
            # if results already exist, skip
            if result_csv is not None:
                if (
                    int(row["video_id"]) in result_csv["video_id"].values
                    and group in result_csv[result_csv["video_id"] == video_id]["group"].values
                ):
                    print(("========================================================================================="))
                    print(f"Video ID {row['video_id']} already processed with group {row['temperature']}. Skipping...")
                    print(("========================================================================================="))
                    continue
            if len(keywords) == 0:  # if no keywords, skip
                with open(
                    Path(ds_cfg.dataset_root, "csv", f"{ds_cfg.CORPUS_FROM}_mutual_info_video_text{postfix}.csv"), "a"
                ) as f:
                    f.write(f"{video_id}|{vid_filename}|{group}|NA|NA|NA|-10|-10|0|-1|-1\n")
                continue
            print(f"Video ID: {video_id}")
            print(f"Group: {group}")
            print(f"Keywords: {keywords}")
            print(f"Original prompt: {orig_prompt}")
            print("=========================================================================================")

            # fill in masked text with masked video. get P(T|V, Tmask)
            tok_prob_rank_video = FillMaskText2LogProb(
                orig_prompt=orig_prompt,
                masked_words_set=keywords,
                video_path=Path(video_root, vid_filename),
                vllm_model=vllm_model,
                min_logprob=-10,
            )

            # fill in masked text. get P(T|Tmask)
            tok_prob_rank = FillMaskText2LogProb(
                orig_prompt=orig_prompt,
                masked_words_set=keywords,
                video_path=None,
                vllm_model=vllm_model,
                min_logprob=-10,
            )

            # save the results in a csv file
            complexity_score = 0
            with open(
                Path(ds_cfg.dataset_root, "csv", f"{ds_cfg.CORPUS_FROM}_mutual_info_video_text{postfix}.csv"), "a"
            ) as f:
                for tuple_video, tuple in zip(tok_prob_rank_video, tok_prob_rank):
                    pred_word_video, logprob_video, rank_video, orig_word = tuple_video
                    pred_word, logprob, rank, orig_word = tuple
                    diff = logprob_video - logprob
                    complexity_score += diff
                    f.write(
                        f"{video_id}|{vid_filename}|{group}|{pred_word_video}|{pred_word}|{orig_word}|{logprob_video}|{logprob}|{diff}|{rank_video}|{rank}\n"
                    )
            print(f"Video ID: {video_id} Complexity score: {complexity_score}")
            print("=========================================================================================")
    elif ds_cfg.CORPUS_FROM == "CoT":
        for idx, row in tqdm(response_csv.iterrows(), desc="Predicting masked tokens"):
            video_id = row["video_id"]
            vid_filename = row["vid_filename"]
            group = "CoT"
            orig_prompt = row["vlm_reasoning"]
            keywords = keywords_dict[str(video_id)][str(row["record_id"])]["keywords"]

            # if str(video_id) not in correct_videoid2metadata.keys():
            if video_id not in correct_videoid2metadata.keys():
                print(f"Video ID {video_id} not found in correct metadata. Skipping...")
                continue
            # if results already exist, skip
            if result_csv is not None:
                if int(row["record_id"]) in result_csv["record_id"].values:
                    print(("========================================================================================="))
                    print(f"Video ID {row['video_id']} already processed. Skipping...")
                    print(("========================================================================================="))
                    continue
            if len(keywords) == 0:  # if no keywords, skip
                with open(
                    Path(ds_cfg.dataset_root, "csv", f"{ds_cfg.CORPUS_FROM}_mutual_info_video_text{postfix}.csv"), "a"
                ) as f:
                    f.write(f"{video_id}|{vid_filename}|{group}|NA|NA|NA|-10|-10|0|-1|-1\n")
                continue
            print(f"Video ID: {video_id}")
            print(f"Group: {group}")
            print(f"Keywords: {keywords}")
            print(f"Original prompt: {orig_prompt}")
            print("=========================================================================================")

            # fill in masked text with masked video. get P(T|V, Tmask)
            tok_prob_rank_video = FillMaskText2LogProb(
                orig_prompt=orig_prompt,
                masked_words_set=keywords,
                video_path=Path(video_root, vid_filename),
                vllm_model=vllm_model,
                min_logprob=-10,
            )

            # fill in masked text. get P(T|Tmask)
            tok_prob_rank = FillMaskText2LogProb(
                orig_prompt=orig_prompt,
                masked_words_set=keywords,
                video_path=None,
                vllm_model=vllm_model,
                min_logprob=-10,
            )

            # save the results in a csv file
            complexity_score = 0
            with open(
                Path(ds_cfg.dataset_root, "csv", f"{ds_cfg.CORPUS_FROM}_mutual_info_video_text{postfix}.csv"), "a"
            ) as f:
                for tuple_video, tuple in zip(tok_prob_rank_video, tok_prob_rank):
                    pred_word_video, logprob_video, rank_video, orig_word = tuple_video
                    pred_word, logprob, rank, orig_word = tuple
                    diff = logprob_video - logprob
                    complexity_score += diff
                    f.write(
                        f"{video_id}|{row['record_id']}|{vid_filename}|{group}|{pred_word_video}|{pred_word}|{orig_word}|{logprob_video}|{logprob}|{diff}|{rank_video}|{rank}\n"
                    )
            print(f"Video ID: {video_id} Complexity score: {complexity_score}")
            print("=========================================================================================")
    else:
        raise ValueError(f"Unknown corpus {ds_cfg.CORPUS_FROM}. Please check the config file.")
    return


if __name__ == "__main__":
    # please run text_and_task first to obtain the masked videos
    TextAndTask()
    # then, run video_and_text
    # VideoAndText()
