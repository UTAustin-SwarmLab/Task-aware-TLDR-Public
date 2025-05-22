"""Run IB score on paper tldr/abstracts. To run:
CUDA_VISIBLE_DEVICES=3 python src/AIConfVideo_script/paper_ib_score.py
"""

import json
import os
import random
import sys
from pathlib import Path

import easyocr
import hydra
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))
from src.client.vllm_client import VLlmClient
from src.utils.easyocr_utils import EasyOCR
from src.utils.fill_in_mask import FillMaskText2LogProb, MaskedVideo2LogProb
from src.utils.primary_area import subarea_map


def GetMetaData(ds_cfg: DictConfig) -> dict:
    """Get metadata from the dataset.
    Returns:
        dict: A dictionary where the key is the slideslive_id and the value is a dictionary with the metadata.
    """
    # list of dictionaries, each dictionary is a paper
    iclr_metadata = json.load(open(Path(ds_cfg.paperlist_root, "dataset/pruned_paperlist_iclr2024.json")))
    nips_metadata = json.load(open(Path(ds_cfg.paperlist_root, "dataset/pruned_paperlist_nips2024.json")))
    merged_metadata = iclr_metadata + nips_metadata
    # only keep the metadata that we need: openreview_id, title, abstract, tldr, keywords, slideslive_id, primary_area
    merged_metadata = {
        paper_dict["slideslive_id"]: {
            "openreview_id": paper_dict["openreview_id"],
            "title": paper_dict["title"],
            "abstract": paper_dict["abstract"],
            "tldr": paper_dict["tldr"],
            "keywords": paper_dict["keywords"],
            "primary_area": paper_dict["primary_area"],
        }
        for paper_dict in merged_metadata
    }
    return merged_metadata


def GetKeywords(keywords_dict, id) -> list[str]:
    """Get all keywords (openreview + tfidf) for a paper decribed by a openreview id."""
    or_keywords = keywords_dict[id]["openreview_keywords"]
    tfidf_keywords = keywords_dict[id]["tf_idf_keywords"]
    # merge the two lists
    all_keywords = list(set(or_keywords + tfidf_keywords))
    return all_keywords


@hydra.main(version_base=None, config_path="../../config", config_name="AIVideoConf")
def TextAndTask(cfg: DictConfig):
    """Run IB score on paper tldr/abstracts and the task (to determine the primary area)."""
    ds_cfg = cfg.AIConfVideo
    video_root = Path(ds_cfg.paperlist_root, "video")
    # get all the video directories
    random.seed(ds_cfg.seed)
    video_dirs = random.sample([d for d in video_root.iterdir() if d.is_dir()], k=ds_cfg.SAMPLE_VIDEO_NUM)
    df_id = pd.read_csv(Path(ds_cfg.paperlist_root, "video", "abstract_mutual_info_text_task.csv"))
    ids = df_id["video_id"].values
    video_dirs = [Path(video_root, str(id)) for id in ids]

    # get metadata
    metadata = GetMetaData(ds_cfg)
    keywords_dict = json.load(open(Path(ds_cfg.paperlist_root, f"dataset/{ds_cfg.CORPUS_FROM}_keywords.json")))
    video_id2openreview_id = {}
    reader = easyocr.Reader(["en"])  # this needs to run only once to load the model into memory

    for video_dir in tqdm(video_dirs, desc="Masking videos"):
        video_id = video_dir.name  # slideslive_id
        if video_id not in metadata.keys():
            print("=========================================================================================")
            print(f"Video ID {video_id} not found in metadata. Skipping...")
            print("=========================================================================================")
            continue
        # Skip video if no png image files are present
        if not any(f.suffix.lower() == ".png" for f in video_dir.iterdir()):
            print("=========================================================================================")
            print(f"Video ID {video_id} has no png image files. Skipping...")
            print("=========================================================================================")
            continue

        # get keywords for the video
        openreview_id = metadata[video_id]["openreview_id"]
        video_id2openreview_id[video_id] = openreview_id
        keywords = GetKeywords(keywords_dict, openreview_id)
        # Skip if the video has already been masked
        if Path(video_root, video_id, f"textmasked_{ds_cfg.CORPUS_FROM}").exists():
            print("===================================================================================================")
            print(f"Video ID {video_id} already masked. Skipping...")
            print("===================================================================================================")
            continue
        png_files = sorted([f for f in video_dir.iterdir() if f.is_file() and f.suffix == ".png"])[
            ds_cfg.max_num_slides :
        ]
        EasyOCR(
            png_files, f"_{ds_cfg.CORPUS_FROM}", keywords=keywords, output_dir=Path(video_root, video_id), reader=reader
        )

    postfix = "" if "Qwen2.5-VL" in cfg.vllm.model_name else f"_{cfg.vllm.model_name.split('/')[-1]}"
    # check if the csv file already exists
    result_csv = None
    if Path(video_root, f"{ds_cfg.CORPUS_FROM}_mutual_info_text_task{postfix}.csv").exists():
        print("=========================================================================================")
        print(f"File {ds_cfg.CORPUS_FROM}_mutual_info_text_task{postfix}.csv already exists. Loading...")
        result_csv = pd.read_csv(Path(video_root, f"{ds_cfg.CORPUS_FROM}_mutual_info_text_task{postfix}.csv"))
        print("=========================================================================================")
    else:
        # create a csv file to save the results
        with open(Path(video_root, f"{ds_cfg.CORPUS_FROM}_mutual_info_text_task{postfix}.csv"), "w") as f:
            # Write header only when creating the file
            f.write(
                "video_id,openreview_id,predicted_answer_text,predicted_answer,correct_answer,informativeness_score,logprob_text,logprob,rank_text,rank\n"
            )

    if ds_cfg.CORPUS_FROM == "aware" or ds_cfg.CORPUS_FROM == "agnostic":
        vlm_response = pd.read_csv(Path(ds_cfg.paperlist_root, "video/vlm_response.csv"), delimiter="|")
    elif ds_cfg.CORPUS_FROM == "CoT":
        vlm_response = pd.read_csv(Path(ds_cfg.paperlist_root, "video/vlm_CoT.csv"), delimiter="|")
    elif "response" in ds_cfg.CORPUS_FROM:
        idx = int(ds_cfg.CORPUS_FROM.replace("response", ""))
        N = ds_cfg.div_response
        vlm_response = pd.read_csv(Path(ds_cfg.paperlist_root, "video/vlm_response_more.csv"), delimiter="|")
        vlm_response = vlm_response[vlm_response.index % N == idx]

    # initiate a vllm model
    vllm_model = VLlmClient(cfg.vllm)
    # generate options for the task
    OPTIONS = sorted(list(set(subarea_map.values())))
    options = " ".join([f"({chr(65 + i)}) {option}" for i, option in enumerate(OPTIONS)])
    # print(f"Options: {options}")

    # predict the masked tokens from the video
    for video_dir in tqdm(video_dirs, desc="Predicting masked tokens"):
        video_id = video_dir.name
        if video_id not in video_id2openreview_id.keys():
            print("=========================================================================================")
            print(f"Video ID {video_id} not found in metadata. Skipping...")
            print("=========================================================================================")
            continue
        if result_csv is not None:
            if int(video_id) in result_csv["video_id"].values:
                print(("========================================================================================="))
                print(f"Video ID {video_id} already processed. Skipping...")
                print(("========================================================================================="))
                continue
        # Skip video if no png image files are present
        if not any(f.suffix.lower() == ".png" for f in video_dir.iterdir()):
            print("=========================================================================================")
            print(f"Video ID {video_id} has no png image files. Skipping...")
            print("=========================================================================================")
            continue

        openreview_id = video_id2openreview_id[video_id]

        # get the correct answer
        primary_area = subarea_map[metadata[video_id]["primary_area"].replace("_", " ").lower()]
        correct_answer = chr(65 + OPTIONS.index(primary_area))
        if ds_cfg.CORPUS_FROM == "aware" or ds_cfg.CORPUS_FROM == "agnostic":
            summary = vlm_response[vlm_response["openreview_id"] == openreview_id][f"task_{ds_cfg.CORPUS_FROM}_TLDR"]
        elif ds_cfg.CORPUS_FROM == "CoT":
            summary = vlm_response[vlm_response["openreview_id"] == openreview_id]["vlm_reasoning"]
            summary = str(summary.values[0])
        elif "response" in ds_cfg.CORPUS_FROM:
            summary = vlm_response[vlm_response["openreview_id"] == openreview_id]["response"]
            summary = str(summary.values[0])
        else:
            summary = metadata[video_id][ds_cfg.CORPUS_FROM]

        # get P(Y|T, Vmask)
        pred_answer_text, logprob_text, rank_text = MaskedVideo2LogProb(
            orig_prompt=f"""Analyze the slides from a paper presentation and the provided summary to identify the relevant research area.
            Summary: {summary}
            Respond only with the corresponding letter: {options}.""",
            answer_choice=correct_answer,
            video_path=Path(video_root, video_id, f"textmasked_{ds_cfg.CORPUS_FROM}"),
            vllm_model=vllm_model,
            min_logprob=-10,
        )

        # get P(Y|Vmask)
        pred_answer, logprob, rank = MaskedVideo2LogProb(
            orig_prompt=f"""Analyze the slides from a paper presentation to identify the relevant research area.
            Respond only with the corresponding letter: {options}.""",
            answer_choice=correct_answer,
            video_path=Path(video_root, video_id, f"textmasked_{ds_cfg.CORPUS_FROM}"),
            vllm_model=vllm_model,
            min_logprob=-10,
        )

        informativeness_score = logprob_text - logprob
        print(f"Video ID: {video_id}, OpenReview ID: {openreview_id}")
        print(f"Pred answer with text: {pred_answer_text}, without {pred_answer} ; Correct answer: {correct_answer}")
        print(f"Logprob_text: {logprob_text}, Logprob: {logprob}")
        print(f"Informativeness score: {informativeness_score}")
        print("=========================================================================================")

        # save the results in a csv file
        with open(Path(video_root, f"{ds_cfg.CORPUS_FROM}_mutual_info_text_task{postfix}.csv"), "a") as f:
            f.write(
                f"{video_id},{openreview_id},{pred_answer_text},{pred_answer},{correct_answer},{informativeness_score},{logprob_text},{logprob},{rank_text},{rank}\n"
            )
    return


@hydra.main(version_base=None, config_path="../../config", config_name="AIVideoConf")
def VideoAndText(cfg: DictConfig):
    """Run IB score on paper video and tldr/abstracts."""
    ds_cfg = cfg.AIConfVideo
    video_root = Path(ds_cfg.paperlist_root, "video")
    # random.seed(ds_cfg.seed)
    # video_dirs = random.sample([d for d in video_root.iterdir() if d.is_dir()], k=ds_cfg.SAMPLE_VIDEO_NUM)
    df_id = pd.read_csv(Path(ds_cfg.paperlist_root, "video", "abstract_mutual_info_text_task.csv"))
    ids = df_id["video_id"].values
    video_dirs = [Path(video_root, str(id)) for id in ids]

    metadata = GetMetaData(ds_cfg)
    keywords_dict = json.load(open(Path(ds_cfg.paperlist_root, f"dataset/{ds_cfg.CORPUS_FROM}_keywords.json")))
    video_id2openreview_id = {}
    postfix = "" if "Qwen2.5-VL" in cfg.vllm.model_name else f"_{cfg.vllm.model_name.split('/')[-1]}"

    # check if the csv file already exists
    result_csv = None
    if Path(video_root, f"{ds_cfg.CORPUS_FROM}_mutual_info_video_text{postfix}.csv").exists():
        print("=========================================================================================")
        print(f"File {ds_cfg.CORPUS_FROM}_mutual_info_video_text{postfix}.csv already exists. Loading...")
        result_csv = pd.read_csv(Path(video_root, f"{ds_cfg.CORPUS_FROM}_mutual_info_video_text{postfix}.csv"))
        print("=========================================================================================")
    else:
        # create a csv file to save the results
        with open(Path(video_root, f"{ds_cfg.CORPUS_FROM}_mutual_info_video_text{postfix}.csv"), "w") as f:
            # Write header only when creating the file
            f.write(
                "video_id,openreview_id,predicted_word_video,predicted_word,orig_word,logprob_video,logprob,diff,rank_video,rank\n"
            )

    if ds_cfg.CORPUS_FROM == "aware" or ds_cfg.CORPUS_FROM == "agnostic":
        vlm_response = pd.read_csv(Path(ds_cfg.paperlist_root, "video/vlm_response.csv"), delimiter="|")
    elif ds_cfg.CORPUS_FROM == "CoT":
        vlm_response = pd.read_csv(Path(ds_cfg.paperlist_root, "video/vlm_CoT.csv"), delimiter="|")
    elif "response" in ds_cfg.CORPUS_FROM:
        idx = int(ds_cfg.CORPUS_FROM.replace("response", ""))
        N = ds_cfg.div_response
        vlm_response = pd.read_csv(Path(ds_cfg.paperlist_root, "video/vlm_response_more.csv"), delimiter="|")
        vlm_response = vlm_response[vlm_response.index % N == idx]

    # initiate a vllm model
    vllm_model = VLlmClient(cfg.vllm)

    for video_dir in tqdm(video_dirs, desc="Processing videos"):
        video_id = video_dir.name  # slideslive_id
        if result_csv is not None:
            if int(video_id) in result_csv["video_id"].values:
                print("=========================================================================================")
                print(f"Video ID {video_id} already processed. Skipping...")
                print("=========================================================================================")
                continue
        if not any(f.suffix.lower() == ".png" for f in video_dir.iterdir()):
            print("=========================================================================================")
            print(f"Video ID {video_id} has no PNG image files. Skipping...")
            print("=========================================================================================")
            continue
        if video_id not in metadata.keys():
            print("=========================================================================================")
            print(f"Video ID {video_id} not found in metadata. Skipping...")
            print("=========================================================================================")
            continue
        # get keywords for the video
        openreview_id = metadata[video_id]["openreview_id"]
        video_id2openreview_id[video_id] = openreview_id
        keywords = set(GetKeywords(keywords_dict, openreview_id))

        print("Obtaining NON-masked video from", video_dir)

        if ds_cfg.CORPUS_FROM == "aware" or ds_cfg.CORPUS_FROM == "agnostic":
            orig_prompt = vlm_response[vlm_response["openreview_id"] == openreview_id][
                f"task_{ds_cfg.CORPUS_FROM}_TLDR"
            ]
            orig_prompt = str(orig_prompt.values[0])
        elif ds_cfg.CORPUS_FROM == "CoT":
            orig_prompt = vlm_response[vlm_response["openreview_id"] == openreview_id]["vlm_reasoning"]
            orig_prompt = str(orig_prompt.values[0])
        elif "response" in ds_cfg.CORPUS_FROM:
            orig_prompt = vlm_response[vlm_response["openreview_id"] == openreview_id]["response"]
            orig_prompt = str(orig_prompt.values[0])
        else:
            orig_prompt = metadata[video_id][ds_cfg.CORPUS_FROM]

        # fill in masked text with masked video. get P(T|V, Tmask)
        tok_prob_rank_video = FillMaskText2LogProb(
            orig_prompt=orig_prompt,
            masked_words_set=keywords,
            video_path=video_dir,
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
        with open(Path(video_root, f"{ds_cfg.CORPUS_FROM}_mutual_info_video_text{postfix}.csv"), "a") as f:
            for tuple_video, tuple in zip(tok_prob_rank_video, tok_prob_rank):
                pred_word_video, logprob_video, rank_video, orig_word = tuple_video
                pred_word, logprob, rank, orig_word = tuple
                diff = logprob_video - logprob
                complexity_score += diff
                f.write(
                    f"{video_id},{openreview_id},{pred_word_video},{pred_word},{orig_word},{logprob_video},{logprob},{diff},{rank_video},{rank}\n"
                )
        print(f"Video ID: {video_id}, OpenReview ID: {openreview_id}")
        print(f"Complexity score: {complexity_score}")
        print("=========================================================================================")
    return


if __name__ == "__main__":
    # please run text_and_task first to obtain the masked videos
    # TextAndTask()
    # then, run video_and_text
    VideoAndText()
