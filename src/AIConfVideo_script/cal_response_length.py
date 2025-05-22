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
from src.AIConfVideo_script.paper_ib_score import GetMetaData
from src.utils.fill_in_mask import HFTokenizeText


@hydra.main(version_base=None, config_path="../../config", config_name="AIVideoConf")
def cal_vlm_response_length(cfg) -> None:
    """
    Calculate the number of tokens in a given string using the specified encoding.

    Args:
        text (str): The input string.
        encoding_name (str): The name of the encoding to use. Default is "gpt-3.5-turbo".

    Returns:
        int: The number of tokens in the input string.
    """
    ds_cfg = cfg.AIConfVideo
    video_root = Path(ds_cfg.paperlist_root, "video")
    df = pd.DataFrame(columns=["openreview_id", "video_id", "group", "response", "length"])

    vlm_response = pd.read_csv(video_root / "vlm_response.csv", delimiter="|")
    vlm_response_more = pd.read_csv(video_root / "vlm_response_more.csv", delimiter="|")
    CoT_response = pd.read_csv(video_root / "vlm_COT.csv", delimiter="|")
    metadata = GetMetaData(ds_cfg=ds_cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg.vllm.model_name)

    for _, row in vlm_response.iterrows():
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "openreview_id": row["openreview_id"],
                            "video_id": row["video_id"],
                            "group": "aware",
                            "response": row["task_aware_TLDR"],
                            "length": len(HFTokenizeText("", row["task_aware_TLDR"], tokenizer)),
                        },
                        {
                            "openreview_id": row["openreview_id"],
                            "video_id": row["video_id"],
                            "group": "agnostic",
                            "response": row["task_agnostic_TLDR"],
                            "length": len(HFTokenizeText("", row["task_agnostic_TLDR"], tokenizer)),
                        },
                    ]
                ),
            ],
            ignore_index=True,
        )

    for _, row in vlm_response.iterrows():
        tldr = metadata[str(row["video_id"])]["tldr"]
        abstract = metadata[str(row["video_id"])]["abstract"]
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "openreview_id": row["openreview_id"],
                            "video_id": row["video_id"],
                            "group": "tldr",
                            "response": tldr,
                            "length": len(HFTokenizeText("", tldr, tokenizer)),
                        },
                        {
                            "openreview_id": row["openreview_id"],
                            "video_id": row["video_id"],
                            "group": "abstract",
                            "response": abstract,
                            "length": len(HFTokenizeText("", abstract, tokenizer)),
                        },
                    ]
                ),
            ],
            ignore_index=True,
        )

    for idx, row in vlm_response_more.iterrows():
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "openreview_id": row["openreview_id"],
                            "video_id": row["video_id"],
                            "group": f"response{idx % 3}",
                            "response": row["response"],
                            "length": len(HFTokenizeText("", row["response"], tokenizer)),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    for idx, row in CoT_response.iterrows():
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "openreview_id": row["openreview_id"],
                            "video_id": row["video_id"],
                            "group": "CoT",
                            "response": row["vlm_reasoning"],
                            "length": len(HFTokenizeText("", row["vlm_reasoning"], tokenizer)),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    print(df.tail())

    # save df
    df.to_csv(Path(ds_cfg.paperlist_root, "dataset", "vlm_response_length.csv"), index=False)
    return


# Example usage
if __name__ == "__main__":
    cal_vlm_response_length()
