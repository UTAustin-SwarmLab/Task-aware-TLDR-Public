import json

import hydra
import pandas as pd
from longvideobench import LongVideoBenchDataset
from omegaconf import DictConfig

question_category = ["E2O", "E2E", "O3O", "S2A", "S2E", "S2O", "SOS"]


@hydra.main(version_base=None, config_path="../../config", config_name="LongVideoBench")
def LoadLongVideoBench(cfg: DictConfig) -> None:
    ds_cfg = cfg.LongVideoBench
    ds_path = ds_cfg.dataset_root
    # validation
    dataset = LongVideoBenchDataset(ds_path, "lvb_val.json", max_num_frames=cfg.vllm.images_per_prompt)
    meta_data = json.load(open(ds_path + "/lvb_val.json", "r"))
    """
    {"video_id": "86CxyhFV9MI",
    "question": "In the video, which subtitles appear at the same time as the man with black hair, 
        dressed in grey clothes with black sleeves, on stage?", 
    "question_wo_referring_query": "Which subtitles appear at the same time?",
    "candidates": ["promisc has come to an end, in and run away countless times, i was just scared, i still",
        "run away countless times, i was just scared, i still and front of our crown, like a world of souls,",
        "promisc has come to an end, in and front of our crown, like a world of souls,", 
        "promisc has come to an end, in and captain of the godson, three three three three three three"], 
    "correct_choice": 1,
    "position": [854, 948, 1373],
    "topic_category": "NP-News-Programs",
    "question_category": "TOS",
    "level": "L2-Relation",
    "id": "86CxyhFV9MI_0",
    "video_path": "86CxyhFV9MI.mp4",
    "subtitle_path": "86CxyhFV9MI_en.json",
    "duration_group": 600,
    "starting_timestamp_for_subtitles": 0,
    "duration": 190.16,
    "view_count": 259852}
    """
    id2idx = {}
    for idx, dict_ in enumerate(meta_data):
        if (
            dict_["duration"] >= ds_cfg.duration[0]
            and dict_["duration"] <= ds_cfg.duration[1]
            and dict_["question_category"] in question_category
        ):
            id2idx[dict_["id"]] = idx
    print(f"Total {len(id2idx.items())} samples")
    return id2idx, meta_data, dataset
    """
    [<PIL.Image.Image image mode=RGB size=1280x720 at 0x7F4D26F0AF80>, 
    '[Music]',
    <PIL.Image.Image image mode=RGB size=1280x720 at 0x7F4D26F0AFB0>,
    <PIL.Image.Image image mode=RGB size=1280x720 at 0x7F4D26F0AFE0>,
    '[Music]',
    'm',
    <PIL.Image.Image image mode=RGB size=1280x720 at 0x7F4D26F0B010>,
    <PIL.Image.Image image mode=RGB size=1280x720 at 0x7F4D26F0B040>,
    <PIL.Image.Image image mode=RGB size=1280x720 at 0x7F4D26F0B070>,
    "don't know what's inside waiting at the beginning, all my tears, the day's",
    <PIL.Image.Image image mode=RGB size=1280x720 at 0x7F4D26F0B0A0>,
    'front of our crown, like a world of souls,',
    <PIL.Image.Image image mode=RGB size=1280x720 at 0x7F4D26F0B0D0>,
    <PIL.Image.Image image mode=RGB size=1280x720 at 0x7F4D26F0B100>,
    <PIL.Image.Image image mode=RGB size=1280x720 at 0x7F4D26F0B130>,
    '[Applause]',
    'La-la, in',
    <PIL.Image.Image image mode=RGB size=1280x720 at 0x7F4D26F0B160>
    <PIL.Image.Image image mode=RGB size=1280x720 at 0x7F4D26F0B190>,
    'you and I, even in',
    <PIL.Image.Image image mode=RGB size=1280x720 at 0x7F4D26F0B1C0>,
    <PIL.Image.Image image mode=RGB size=1280x720 at 0x7F4D26F0B1F0>,
    <PIL.Image.Image image mode=RGB size=1280x720 at 0x7F4D26F0B220>,
    <PIL.Image.Image image mode=RGB size=1280x720 at 0x7F4D26F0B250>,
    '[music],', 
    <PIL.Image.Image image mode=RGB size=1280x720 at 0x7F4D26F0B280>,
    <PIL.Image.Image image mode=RGB size=1280x720 at 0x7F4D26F0B2B0>,
    '[music]  ]', 
    <PIL.Image.Image image mode=RGB size=1280x720 at 0x7F4D26F0B2E0>,
    'Serara',
    <PIL.Image.Image image mode=RGB size=1280x720 at 0x7F4D26F0B310>,
    'Question: In the video, which subtitles appear at the same time as the man with black hair, dressed in grey clothes with black sleeves, on stage?',
    'A. promisc has come to an end, in and run away countless times, i was just scared, i still',
    'B. run away countless times, i was just scared, i still and front of our crown, like a world of souls,',
    'C. promisc has come to an end, in and front of our crown, like a world of souls,', 
    'D. promisc has come to an end, in and captain of the godson, three three three three three three', "Answer with the option's letter from the given choices directly."]
    """


def Acc():
    csv_path = "/nas/pohan/datasets/LongVideoBench/csv/vlm_CoT.csv"
    df = pd.read_csv(csv_path, delimiter="|")
    df["vlm_answer"] = df["vlm_answer"].apply(lambda x: x.split("(")[-1].split(")")[0])
    # count the acc (vlm_answer==answer) of each question_category
    acc_count = (
        df[df["vlm_answer"] == df["answer"]].groupby("question_category").size()
        / df.groupby("question_category").size()
    )
    print(acc_count)


if __name__ == "__main__":
    # dataset = LoadLongVideoBench()
    Acc()
