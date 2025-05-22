"""
This script calculates the number of tokens in a given string (vlm response) using the specified encoding.
"""

import os
import sys
from pathlib import Path

import hydra
import pandas as pd
from transformers import AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))
from src.utils.fill_in_mask import HFTokenizeText


@hydra.main(version_base=None, config_path="../../config", config_name="TrafficQA")
def cal_vlm_response_length(cfg) -> None:
    """
    Calculate the number of tokens in a given string using the specified encoding.

    Args:
        text (str): The input string.
        encoding_name (str): The name of the encoding to use. Default is "gpt-3.5-turbo".

    Returns:
        int: The number of tokens in the input string.
    """
    ds_cfg = cfg.TrafficQA
    csv_root = Path(ds_cfg.dataset_root, "csv")
    df = pd.DataFrame(columns=["video_id", "group", "response", "length"])
    tokenizer = AutoTokenizer.from_pretrained(cfg.vllm.model_name)

    vlm_response = pd.read_csv(csv_root / "vlm_responses.csv", delimiter="|")
    for _, row in vlm_response.iterrows():
        df.loc[len(df)] = {
            "video_id": row["video_id"],
            "group": f"temperature_{row['temperature']}",
            "response": row["response"],
            "length": len(HFTokenizeText("", row["response"], tokenizer)),
        }
    # save df
    df.to_csv(Path(ds_cfg.dataset_root, "csv", "vlm_response_length.csv"), index=False)

    CoT_response = pd.read_csv(csv_root / "vlm_COT.csv", delimiter="|")
    for idx, row in CoT_response.iterrows():
        df.loc[len(df)] = {
            "video_id": row["video_id"],
            "record_id": row["record_id"],
            "group": "CoT",
            "response": row["vlm_reasoning"],
            "length": len(HFTokenizeText("", row["vlm_reasoning"], tokenizer)),
        }
    # save df
    df.to_csv(Path(ds_cfg.dataset_root, "csv", "vlm_CoT_length.csv"), index=False)


# Example usage
if __name__ == "__main__":
    cal_vlm_response_length()
