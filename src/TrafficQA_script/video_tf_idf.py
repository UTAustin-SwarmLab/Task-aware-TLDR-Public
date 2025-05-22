"""
This script extracts keywords from the vlm responses of TrafficQA dataset.
The keywords are saved in a new json file with the record_id as the key and the list of keywords as the value.
The keywords are lemmatized and combined to remove singular/plural forms.
"""

import json
import os
import re
import sys
from pathlib import Path

import hydra
import nltk
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))
from src.utils.tf_idf import tf_idf

IGNORE = [
    "video",
    "accident",
    "car",
    "cars",
    "vehicle",
    "vehicles",
    "bus",
    "buses",
    "truck",
    "trucks",
    "van",
    "vans",
    "road",
    "street",
    "driving",
    "including",
    "traffic",
]


@hydra.main(version_base=None, config_path="../../config", config_name="TrafficQA")
def main(cfg: DictConfig):
    ds_cfg = cfg.TrafficQA
    Path(ds_cfg.dataset_root, "json").mkdir(parents=True, exist_ok=True)

    corpus = {}
    CoT_df = pd.read_csv(Path(ds_cfg.dataset_root, "csv", "vlm_CoT.csv"), delimiter="|")
    for idx, row in CoT_df.iterrows():
        if row["video_id"] not in corpus.keys():
            corpus[row["video_id"]] = {row["record_id"]: row["vlm_reasoning"]}
        else:
            corpus[row["video_id"]][row["record_id"]] = row["vlm_reasoning"]
    responses_df = pd.read_csv(Path(ds_cfg.dataset_root, "csv", "vlm_responses_new4.csv"), delimiter="|")
    for idx, row in responses_df.iterrows():
        corpus[row["video_id"]][row["temperature"]] = row["response"]

    # Run TF-IDF
    key_tokens, scores = tf_idf(
        [response for video_responses in corpus.values() for response in video_responses.values()],
        top_k=ds_cfg.tf_idf.top_k,
        cutoff_thres=ds_cfg.tf_idf.cutoff_thres,
        max_df=ds_cfg.tf_idf.max_df,
        min_df=ds_cfg.tf_idf.min_df,
        ngram_range=(1, 1),
    )
    print(f"Top tokens: {key_tokens[-10:]}, Scores: {scores[-10:]}, len(key_tokens): {len(key_tokens)}")

    # The json's key is the openreview_id and the value is the list of keywords from openreview and td-idf
    id2keywords = {}
    lens_keywords = []
    for _, row in tqdm(CoT_df.iterrows(), desc="Saving CoT keywords", total=len(CoT_df)):
        video_id, record_id = row["video_id"], row["record_id"]
        keywords = []
        scores_list = []
        for key_token, score in zip(key_tokens, scores):
            # CoT keywords are the keywords that are in the vlm reasoning
            if re.search(rf"\b{re.escape(key_token)}\b", row["vlm_reasoning"]) and key_token not in IGNORE:
                keywords.append(key_token)
                scores_list.append(round(score, 4))

        if video_id not in id2keywords.keys():
            id2keywords[video_id] = {}
        # Use record_id as the key
        id2keywords[video_id][record_id] = {"keywords": keywords, "CoT_scores": scores_list}
        lens_keywords.append(len(keywords))

    print(f"Average number of keywords per CoT: {sum(lens_keywords) / len(lens_keywords)}")
    with open(Path(ds_cfg.dataset_root, "json", "CoT_keywords_new4.json"), "w") as f:
        json.dump(id2keywords, f, indent=2)

    # Responses keywords are the keywords that are in the vlm responses
    id2keywords = {}
    lens_keywords = []
    for idx, row in tqdm(responses_df.iterrows(), desc="Saving vlm response keywords", total=len(responses_df)):
        video_id = row["video_id"]
        keywords = []
        scores_list = []
        for key_token, score in zip(key_tokens, scores):
            if key_token in IGNORE:
                continue
            # CoT keywords are the keywords that are in the vlm reasoning
            if re.search(rf"\b{re.escape(key_token)}\b", row["response"]) and key_token not in IGNORE:
                keywords.append(key_token)
                scores_list.append(round(score, 4))
        if video_id not in id2keywords.keys():
            id2keywords[video_id] = {}
        id2keywords[video_id][row["temperature"]] = {"keywords": keywords, "scores": scores_list}
        lens_keywords.append(len(keywords))

    print(f"Average number of keywords per response: {sum(lens_keywords) / len(lens_keywords)}")
    with open(Path(ds_cfg.dataset_root, "json", "vlm_response_keywords_new4.json"), "w") as f:
        json.dump(id2keywords, f, indent=2)


@hydra.main(version_base=None, config_path="../../config", config_name="TrafficQA")
def MaskNounAndVerb(cfg: DictConfig):
    nltk.download("punkt")
    nltk.download("averaged_perceptron_tagger")
    nltk.download("stopwords")
    stop_words = set(nltk.corpus.stopwords.words("english"))

    ds_cfg = cfg.TrafficQA
    Path(ds_cfg.dataset_root, "json").mkdir(parents=True, exist_ok=True)

    CoT_df = pd.read_csv(Path(ds_cfg.dataset_root, "csv", "vlm_CoT.csv"), delimiter="|")
    id2v_n = {}
    for idx, row in CoT_df.iterrows():
        dict_v_n = DetectVerbNoun(row["vlm_reasoning"], stop_words)
        if row["video_id"] not in id2v_n.keys():
            id2v_n[row["video_id"]] = {row["record_id"]: dict_v_n}
        else:
            id2v_n[row["video_id"]][row["record_id"]] = dict_v_n

    responses_df = pd.read_csv(Path(ds_cfg.dataset_root, "csv", "vlm_responses.csv"), delimiter="|")
    for idx, row in responses_df.iterrows():
        dict_v_n = DetectVerbNoun(row["response"], stop_words)  # Changed to row["response"]
        id2v_n[row["video_id"]][row["temperature"]] = dict_v_n

    with open(Path(ds_cfg.dataset_root, "json", "vlm_response_verbs_nouns.json"), "w") as f:
        json.dump(id2v_n, f, indent=4)


def DetectVerbNoun(text: str, stop_words: set) -> dict:
    """
    Detect the verbs and nouns in the vlm responses.
    This function uses nltk to detect verbs and nouns.
    """
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalnum()]
    tagged_tokens = nltk.pos_tag(filtered_tokens)
    verbs = [word for word, tag in tagged_tokens if tag.startswith("VB") and word.lower() not in IGNORE]
    nouns = [word for word, tag in tagged_tokens if tag.startswith("NN") and word.lower() not in IGNORE]
    return {"verbs": verbs, "nouns": nouns}


if __name__ == "__main__":
    main()
    MaskNounAndVerb()
