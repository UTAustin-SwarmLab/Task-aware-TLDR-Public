"""Run IB score on traffic accident videos. To run:
CUDA_VISIBLE_DEVICES=1,2 python src/LongVideoBench_script/video_ib_score.py
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
from src.LongVideoBench_script.load_data import LoadLongVideoBench
from src.utils.fill_in_mask import FillMaskText2LogProb, MaskedVideo2LogProb


def GetResponsesAndKeywords(ds_cfg: DictConfig) -> dict:
    if ds_cfg.CORPUS_FROM == "CoT":
        return (
            pd.read_csv(Path(ds_cfg.dataset_root, "csv", "vlm_CoT.csv"), delimiter="|"),
            json.load(open(Path(ds_cfg.dataset_root, "json", "CoT_keywords.json"))),
        )
    elif ds_cfg.CORPUS_FROM == "response":
        return (
            pd.read_csv(Path(ds_cfg.dataset_root, "csv", "vlm_responses.csv"), delimiter="|"),
            json.load(open(Path(ds_cfg.dataset_root, "json", "vlm_response_keywords.json"))),
        )
    else:
        raise ValueError(f"Unknown corpus: {ds_cfg.CORPUS_FROM}. Please choose from 'CoT' or 'response'.")


@hydra.main(version_base=None, config_path="../../config", config_name="LongVideoBench")
def TextAndTask(cfg: DictConfig):
    """Run IB score on long videos."""
    ds_cfg = cfg.LongVideoBench
    postfix = "" if "Qwen2.5-VL" in cfg.vllm.model_name else f"_{cfg.vllm.model_name.split('/')[-1]}"
    random.seed(ds_cfg.seed)

    # get metadata
    response_csv, keywords = GetResponsesAndKeywords(ds_cfg)

    id2idx, metadata_list, ds = LoadLongVideoBench(cfg)

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
                "video_id|record_id|group|question|predicted_answer_text|predicted_answer|correct_answer|informativeness_score|logprob_text|logprob|rank_text|rank\n"
            )

    # initiate a vllm model
    vllm_model = VLlmClient(cfg.vllm)

    # predict the masked tokens from the video
    for idx, row in tqdm(response_csv.iterrows(), desc="Predicting task answers"):
        video_id = row["video_id"]
        record_id = row["record_id"]
        group = f"temperature_{row['temperature']}" if ds_cfg.CORPUS_FROM == "response" else ds_cfg.CORPUS_FROM

        metadata = metadata_list[id2idx[record_id]]
        assert metadata["video_id"] == video_id, f"Metadata {metadata['video_id']} does not match response {video_id}."
        assert metadata["id"] == record_id, f"Metadata {metadata['id']} does not match response {record_id}."
        question = metadata["question"]
        answer_idx = metadata["correct_choice"]

        print(f"Video ID: {video_id}")
        print(f"Record ID: {record_id}")
        print(f"Group: {group}")
        print(f"Question: {question}")
        print(f"Answer: {answer_idx}")
        print("=========================================================================================")

        if result_csv is not None and row["record_id"] in result_csv["record_id"].values:
            if ds_cfg.CORPUS_FROM == "CoT":
                print(("========================================================================================="))
                print(f"Record ID {row['record_id']} with group {group} already processed. Skip.")
                print(("========================================================================================="))
                continue
            elif ds_cfg.CORPUS_FROM == "response":
                if group in result_csv[result_csv["record_id"] == record_id]["group"].values:
                    print(("========================================================================================="))
                    print(f"Record ID {row['record_id']} with group {group} already processed. Skip.")
                    print(("========================================================================================="))
                    continue

        options = metadata["candidates"]
        options_with_alphabet = [chr(65 + i) + ": " + str(opt) + "." for i, opt in enumerate(options)]
        answer = chr(65 + answer_idx)  # A=0, B=1, C=2, D=3
        summary = row["response"] if ds_cfg.CORPUS_FROM == "response" else row["vlm_reasoning"]
        inputs = ds[idx]["inputs"]
        print(options_with_alphabet)
        print(answer)
        print(summary)

        # get P(Y|T, Vmask)
        pred_answer_text, logprob_text, rank_text = MaskedVideo2LogProb(
            orig_prompt=f"""Analyze the cropped video and the provided summary of the video. Summary: {summary} Now, answer the question: {question} Options: {" ".join(options_with_alphabet).replace("[", "").replace("]", "")} Answer with the corresponding letter: """,
            answer_choice=answer,
            video_path=inputs,
            vllm_model=vllm_model,
            min_logprob=-10,
            crop=True,
        )

        # get P(Y|Vmask)
        pred_answer, logprob, rank = MaskedVideo2LogProb(
            orig_prompt=f"""Analyze the cropped video. Now, answer the question: {question} Options: {" ".join(options_with_alphabet).replace("[", "").replace("]", "")}
            Answer with the corresponding letter: """,
            answer_choice=answer,
            video_path=inputs,
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
                f"{video_id}|{record_id}|{group}|{question}|{pred_answer_text}|{pred_answer}|{answer}|{informativeness_score}|{logprob_text}|{logprob}|{rank_text}|{rank}\n"
            )
    return


@hydra.main(version_base=None, config_path="../../config", config_name="LongVideoBench")
def VideoAndText(cfg: DictConfig):
    """Run IB score on long videos."""
    ds_cfg = cfg.LongVideoBench
    postfix = "" if "Qwen2.5-VL" in cfg.vllm.model_name else f"_{cfg.vllm.model_name.split('/')[-1]}"
    random.seed(ds_cfg.seed)

    # get metadata
    response_csv, keywords_dict = GetResponsesAndKeywords(ds_cfg)
    id2idx, metadata_list, ds = LoadLongVideoBench(cfg)

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
        with open(
            Path(ds_cfg.dataset_root, "csv", f"{ds_cfg.CORPUS_FROM}_mutual_info_video_text{postfix}.csv"), "w"
        ) as f:
            # Write header only when creating the file
            f.write(
                "video_id|record_id|group|predicted_word_video|predicted_word|orig_word|logprob_video|logprob|diff|rank_video|rank\n"
            )

    # initiate a vllm model
    vllm_model = VLlmClient(cfg.vllm)

    # predict the masked tokens from the video
    for idx, row in tqdm(response_csv.iterrows(), desc="Predicting task answers"):
        video_id = row["video_id"]
        record_id = row["record_id"]
        group = f"temperature_{row['temperature']}" if ds_cfg.CORPUS_FROM == "response" else ds_cfg.CORPUS_FROM

        if result_csv is not None and row["record_id"] in result_csv["record_id"].values:
            if ds_cfg.CORPUS_FROM == "CoT":
                print(("========================================================================================="))
                print(f"Record ID {record_id} with group {group} already processed. Skip.")
                print(("========================================================================================="))
                continue
            elif ds_cfg.CORPUS_FROM == "response":
                if group in result_csv[result_csv["record_id"] == record_id]["group"].values:
                    print(("========================================================================================="))
                    print(f"Record ID {record_id} with group {group} already processed. Skip.")
                    print(("========================================================================================="))
                    continue

        metadata = metadata_list[id2idx[record_id]]
        assert metadata["video_id"] == video_id, f"Metadata {metadata['id']} does not match response {record_id}."
        assert metadata["id"] == record_id, f"Metadata record_id {metadata['id']} does not match response {record_id}."
        orig_prompt = row["vlm_reasoning"] if ds_cfg.CORPUS_FROM == "CoT" else row["response"]
        keywords = (
            keywords_dict[record_id]["keywords"]
            if ds_cfg.CORPUS_FROM == "CoT"
            else keywords_dict[record_id][str(row["temperature"])]["keywords"]
        )
        inputs = ds[idx]["inputs"]

        if len(keywords) == 0:  # if no keywords, skip
            print(f"Video ID: {video_id} has no keywords. Skip.")
            print("=========================================================================================")
            with open(
                Path(ds_cfg.dataset_root, "csv", f"{ds_cfg.CORPUS_FROM}_mutual_info_video_text{postfix}.csv"), "a"
            ) as f:
                f.write(f"{video_id}|{record_id}|{group}|NA|NA|NA|-10|-10|0|-1|-1\n")
            continue

        print(f"Video ID: {video_id}")
        print(f"Record ID: {record_id}")
        print(f"Group: {group}")
        print(f"Keywords: {keywords}")
        print(f"Original prompt: {orig_prompt}")
        print("=========================================================================================")

        # fill in masked text with masked video. get P(T|V, Tmask)
        tok_prob_rank_video = FillMaskText2LogProb(
            orig_prompt=orig_prompt,
            masked_words_set=keywords,
            video_path=inputs,
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
                    f"{video_id}|{record_id}|{group}|{pred_word_video}|{pred_word}|{orig_word}|{logprob_video}|{logprob}|{diff}|{rank_video}|{rank}\n"
                )
        print(f"Video ID: {video_id} Complexity score: {complexity_score}")
        print("=========================================================================================")
    return


if __name__ == "__main__":
    # please run text_and_task first to obtain the masked videos
    TextAndTask()
    # then, run video_and_text
    VideoAndText()
