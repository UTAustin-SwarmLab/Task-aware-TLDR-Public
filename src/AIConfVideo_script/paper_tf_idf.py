"""
This script extracts keywords from the OpenReview dataset and runs TF-IDF to extract keywords from the paper corpus
(abstract or tldr) defined in the main.yaml config.
The keywords are saved in a new json file with the openreview_id as the key and the list of keywords as the value.
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
from nltk.stem import WordNetLemmatizer
from omegaconf import DictConfig

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))
from tqdm import tqdm

from src.utils.tf_idf import tf_idf

IGNORED = ["learning", "model", "models", "neural", "network", "networks", "deep", "and", "problem", "problems"]
TEST_NUM = 10000

# Initialize lemmatizer for combining singular/plural forms
lemmatizer = WordNetLemmatizer()

# Download required NLTK data if not already available
try:
    nltk.data.find("wordnet")
except LookupError:
    nltk.download("wordnet")


@hydra.main(version_base=None, config_path="../../config", config_name="AIVideoConf")
def main(cfg: DictConfig):
    ds_cfg = cfg.AIConfVideo
    # list of dictionaries, each dictionary is a paper
    iclr_metadata = json.load(open(Path(ds_cfg.paperlist_root, "dataset/paperlist_iclr2024.json")))[:TEST_NUM]
    nips_metadata = json.load(open(Path(ds_cfg.paperlist_root, "dataset/paperlist_nips2024.json")))[:TEST_NUM]

    corpus = {}
    if ds_cfg.CORPUS_FROM == "abstract" or ds_cfg.CORPUS_FROM == "tldr":
        for paper in iclr_metadata:
            corpus[paper["openreview_id"]] = paper[ds_cfg.CORPUS_FROM]
        for paper in nips_metadata:
            corpus[paper["openreview_id"]] = paper[ds_cfg.CORPUS_FROM]
    elif ds_cfg.CORPUS_FROM == "CoT":
        df = pd.read_csv(Path(ds_cfg.paperlist_root, "video/vlm_CoT.csv"), delimiter="|")
        for idx, row in df.iterrows():
            corpus[row["openreview_id"]] = row["vlm_reasoning"]
    else:
        N = ds_cfg.div_response
        aware_corpus = {}
        agnostic_corpus = {}
        responses_corpus = [{} for _ in range(N)]
        df = pd.read_csv(Path(ds_cfg.paperlist_root, "video/vlm_response.csv"), delimiter="|")
        for idx, row in df.iterrows():
            aware_corpus[row["openreview_id"]] = row["task_aware_TLDR"]
            agnostic_corpus[row["openreview_id"]] = row["task_agnostic_TLDR"]
            corpus[(row["openreview_id"], 3)] = row["task_aware_TLDR"]
            corpus[(row["openreview_id"], 4)] = row["task_agnostic_TLDR"]
        df = pd.read_csv(Path(ds_cfg.paperlist_root, "video/vlm_response_more.csv"), delimiter="|")
        for idx, row in df.iterrows():
            responses_corpus[idx % N][row["openreview_id"]] = row["response"]
            corpus[(row["openreview_id"], idx % N)] = row["response"]

    # Get keywords from OpenReview
    openreview_keywords = {}
    openreview_raw_cnt = {}  # Original keyword counts (before lemmatization)
    openreview_lemma_map = {}  # Maps lemmatized keywords to their original forms
    openreview_cnt = {}  # Lemmatized keyword counts

    for paper in iclr_metadata + nips_metadata:
        id = paper["openreview_id"]
        raw_keywords = paper["keywords"].lower().replace("(", "").replace(")", "").replace(";", " ").split(" ")
        if "" in raw_keywords:
            raw_keywords.remove("")

        # Process keywords to combine singular/plural forms
        lemmatized_keywords = []
        for keyword in raw_keywords:
            # Lemmatize the keyword (converts plural to singular)
            lemma = lemmatizer.lemmatize(keyword, pos="n")
            lemmatized_keywords.append(lemma)

            # Update the mapping of lemmas to original forms
            if lemma not in openreview_lemma_map:
                openreview_lemma_map[lemma] = set()
            openreview_lemma_map[lemma].add(keyword)

            # Count original keywords
            if keyword not in openreview_raw_cnt:
                openreview_raw_cnt[keyword] = 0
            openreview_raw_cnt[keyword] += 1

            # Count lemmatized keywords
            if lemma not in openreview_cnt:
                openreview_cnt[lemma] = 0
            openreview_cnt[lemma] += 1

        openreview_keywords[id] = lemmatized_keywords

    # Run TF-IDF
    print("=== TF-IDF Key Tokens ===")
    key_tokens, scores = tf_idf(
        list(corpus.values()),
        top_k=ds_cfg.tf_idf.top_k,
        cutoff_thres=ds_cfg.tf_idf.cutoff_thres,
        max_df=ds_cfg.tf_idf.max_df,
        min_df=ds_cfg.tf_idf.min_df,
    )
    print(f"Top tokens: {key_tokens[-10:]}, Scores: {scores[-10:]}, len(key_tokens): {len(key_tokens)}")

    # Save all keywords per paper in a new json file.
    # The json's key is the openreview_id and the value is the list of keywords from openreview and td-idf
    if ds_cfg.CORPUS_FROM == "abstract" or ds_cfg.CORPUS_FROM == "tldr" or ds_cfg.CORPUS_FROM == "CoT":
        id2keywords = {}
        lens_keywords = []
        for id, keywords in tqdm(openreview_keywords.items(), desc="Processing papers"):
            if id not in corpus:
                print(f"Paper {id} not in corpus. Skipping...")
                continue
            keywords = list(set([keyword for keyword in keywords if keyword not in IGNORED]))
            tf_idf_keywords = []
            scores_list = []
            for key_token, score in zip(key_tokens, scores):
                if key_token in IGNORED:
                    continue
                if re.search(rf"\b{re.escape(key_token)}\b", corpus[id]):
                    tf_idf_keywords.append(key_token)
                    scores_list.append(round(score, 4))
            id2keywords[id] = {
                "openreview_keywords": keywords,
                "tf_idf_keywords": tf_idf_keywords,
                "tf_idf_scores": scores_list,
            }
            lens_keywords.append(len(tf_idf_keywords))

        print(f"Average number of keywords per paper: {sum(lens_keywords) / len(lens_keywords)}")
        with open(Path(ds_cfg.paperlist_root, f"dataset/{ds_cfg.CORPUS_FROM}_keywords.json"), "w") as f:
            json.dump(id2keywords, f)
    else:
        # Save keywords for task-aware tldr
        id2keywords = {}
        lens_keywords = []
        for id, keywords in tqdm(openreview_keywords.items(), desc="Processing papers"):
            if id not in aware_corpus:
                print(f"Paper {id} not in aware_corpus. Skipping...")
                continue
            keywords = list(set([keyword for keyword in keywords if keyword not in IGNORED]))
            tf_idf_keywords = []
            scores_list = []
            for key_token, score in zip(key_tokens, scores):
                if key_token in IGNORED:
                    continue
                if re.search(rf"\b{re.escape(key_token)}\b", aware_corpus[id]):
                    tf_idf_keywords.append(key_token)
                    scores_list.append(round(score, 4))
            id2keywords[id] = {
                "openreview_keywords": keywords,
                "tf_idf_keywords": tf_idf_keywords,
                "tf_idf_scores": scores_list,
            }
            lens_keywords.append(len(tf_idf_keywords))
        print(f"Average number of keywords per paper: {sum(lens_keywords) / len(lens_keywords)}")
        with open(Path(ds_cfg.paperlist_root, "dataset/aware_keywords.json"), "w") as f:
            json.dump(id2keywords, f)

        # Save keywords for task-agnostic tldr
        id2keywords = {}
        lens_keywords = []
        for id, keywords in tqdm(openreview_keywords.items(), desc="Processing papers"):
            if id not in agnostic_corpus:
                print(f"Paper {id} not in agnostic_corpus. Skipping...")
                continue
            keywords = list(set([keyword for keyword in keywords if keyword not in IGNORED]))
            tf_idf_keywords = []
            scores_list = []
            for key_token, score in zip(key_tokens, scores):
                if key_token in IGNORED:
                    continue
                if re.search(rf"\b{re.escape(key_token)}\b", agnostic_corpus[id]):
                    tf_idf_keywords.append(key_token)
                    scores_list.append(round(score, 4))
            id2keywords[id] = {
                "openreview_keywords": keywords,
                "tf_idf_keywords": tf_idf_keywords,
                "tf_idf_scores": scores_list,
            }
            lens_keywords.append(len(tf_idf_keywords))
        print(f"Average number of keywords per paper: {sum(lens_keywords) / len(lens_keywords)}")
        with open(Path(ds_cfg.paperlist_root, "dataset/agnostic_keywords.json"), "w") as f:
            json.dump(id2keywords, f)

        # Save keywords for each response
        for idx in range(ds_cfg.div_response):
            id2keywords = {}
            lens_keywords = []
            for id, keywords in tqdm(openreview_keywords.items(), desc="Processing papers"):
                if id not in responses_corpus[idx]:
                    print(f"Paper {id} not in responses_corpus[{idx}]. Skipping...")
                    continue
                keywords = list(set([keyword for keyword in keywords if keyword not in IGNORED]))
                tf_idf_keywords = []
                scores_list = []
                for key_token, score in zip(key_tokens, scores):
                    if key_token in IGNORED:
                        continue
                    if re.search(rf"\b{re.escape(key_token)}\b", responses_corpus[idx][id]):
                        tf_idf_keywords.append(key_token)
                        scores_list.append(round(score, 4))
                id2keywords[id] = {
                    "openreview_keywords": keywords,
                    "tf_idf_keywords": tf_idf_keywords,
                    "tf_idf_scores": scores_list,
                }
                lens_keywords.append(len(tf_idf_keywords))
            print(f"Average number of keywords per paper: {sum(lens_keywords) / len(lens_keywords)}")
            with open(Path(ds_cfg.paperlist_root, f"dataset/response{idx}_keywords.json"), "w") as f:
                json.dump(id2keywords, f)


if __name__ == "__main__":
    main()
