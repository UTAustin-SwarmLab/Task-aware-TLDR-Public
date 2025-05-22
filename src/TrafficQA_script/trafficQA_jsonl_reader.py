import json
import random
from pathlib import Path

import hydra
from omegaconf import DictConfig

Q_TYPE_MAP = {
    "U": "Basic Understanding",
    "A": "Attribution",
    "F": "Event Forecasting",
    "R": "Reverse Reasoning",
    "C": "Counterfactual Inference",
    "I": "Introspection",
}
SKIP_LIST = ["A", "F", "R"]


def load_data(ds_cfg: DictConfig) -> dict:
    """
    Load the TrafficQA dataset from JSONL format.
    Args:
        ds_cfg (DictConfig): Dataset configuration object containing dataset root path.
    Returns:
        dict: A dictionary mapping video IDs to their corresponding metadata.
    """
    with open(Path(ds_cfg.dataset_root, "annotations", "R2_all.jsonl")) as f:
        lines = f.readlines()

    _header = lines.pop(0)
    videoid2data = {}

    for line in lines:
        data: list = json.loads(line.strip())
        record_id: str = data[0]
        vid_id: str = data[1]
        vid_filename: str = data[2]
        q_body: str = data[4]
        q_type: str = data[5]
        options: list = data[6:10]
        answer_idx: int = data[10]
        # answer_str: str = options[answer_idx]
        if q_type in SKIP_LIST:
            continue

        q_type = Q_TYPE_MAP.get(q_type, "Unknown")
        if vid_id not in videoid2data:
            videoid2data[vid_id] = [
                {
                    "record_id": record_id,
                    "filename": vid_filename,
                    "q_type": q_type,
                    "question": q_body,
                    "option0": options[0],
                    "option1": options[1],
                    "option2": options[2],
                    "option3": options[3],
                    "answer": answer_idx,
                }
            ]
        else:
            videoid2data[vid_id].append(
                {
                    "record_id": record_id,
                    "filename": vid_filename,
                    "q_type": q_type,
                    "question": q_body,
                    "option0": options[0],
                    "option1": options[1],
                    "option2": options[2],
                    "option3": options[3],
                    "answer": answer_idx,
                }
            )

    # Remove video IDs with less than 3 records
    videoid2data = {vid_id: data for vid_id, data in videoid2data.items() if len(data) >= 6}
    print(f"Filtered video IDs, remaining count: {len(videoid2data)}")
    return videoid2data


@hydra.main(version_base=None, config_path="../../config", config_name="TrafficQA")
def main(cfg: DictConfig) -> None:
    videoid_dict = load_data(cfg.TrafficQA)
    random.seed(0)
    sampled_ids = random.sample(list(videoid_dict.keys()), 10)
    print(len(videoid_dict))
    for vid_id in sampled_ids:
        print(len(videoid_dict[vid_id]))


if __name__ == "__main__":
    main()
