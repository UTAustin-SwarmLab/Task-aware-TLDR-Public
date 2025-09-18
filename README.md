# Task-aware Video-to-Text TLDR

This repository contains source code for **VIBE: Video-to-Text Information Bottleneck Evaluation for TL;DR**

[![website](https://img.shields.io/badge/website-76b900?style=for-the-badge&logo=safari&labelColor=555555)](https://vivianchen98.github.io/VIBE_website/)
[![Arxiv](https://img.shields.io/badge/Arxiv-b31b1b?style=for-the-badge&logo=arxiv&labelColor=555555)](https://arxiv.org/abs/2505.17423)
[![Dataset (LearningPaper24)](https://img.shields.io/badge/-LearningPaper24-ff6e00?style=for-the-badge&logo=huggingface&labelColor=555555)](https://huggingface.co/datasets/vivianchen98/LearningPaper24)

## TL;DR
VIBE is an annotation-free method that selects video summaries by scoring task relevance and visual grounding without retraining. Human studies show VIBE improves accuracy and reduces response time over naive VLM summaries and full videos across three datasets.

## Table of Contents
- [System Plot and Major Results](#system-plot-and-major-results)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Key Utility Functions](#key-utility-functions)
- [Model Configuration](#model-configuration)
- [Citation](#citation)


## System Plot and Major Results

![system_graph](https://github.com/UTAustin-SwarmLab/Task-aware-TLDR/blob/main/assets/TLDR_system_plot.png)

![main_results](https://github.com/UTAustin-SwarmLab/Task-aware-TLDR/blob/main/assets/table_results.png)
Our results show that video captions selected by VIBE can achieve a better trade-off between faster human response time and task accuracy.

## Project Structure

```
.
├── config/                 # Configuration files for different tasks
│   ├── AIVideoConf.yaml   # AI Conference Video analysis config
│   ├── LongVideoBench.yaml # Long video benchmark config
│   └── TrafficQA.yaml     # Traffic QA config
│
└── src/                   # Source code
    ├── AIConfVideo_script/    # AI Conference Video analysis
    ├── LongVideoBench_script/ # Long video benchmark
    ├── TrafficQA_script/      # Traffic QA
    ├── client/                # Client-side code
    ├── server/                # Server-side code
    └── utils/                 # Utility functions
```

## Prerequisites
See `requirements.txt` for the packages and their versions.

## Key Utility Functions

The `src/utils/` directory contains several important utility modules:

### VIBE Score Calculation
- 'src/<dataset>_script/*_ib_score.py' contains the core code to calculate grounding and utility scores.
- 'src/<dataset>_script/vllm_generate_summary.py' contains the main function to generate captions for videos using the vllm package.

### Text Processing
- `tf_idf.py`: TF-IDF based keyword extraction from text corpus
  - Extracts important keywords using Term Frequency-Inverse Document Frequency
  - Supports customizable parameters for document frequency thresholds
  - Handles n-grams and stop words

### Computer Vision
- `easyocr_utils.py`: Optical Character Recognition (OCR) utilities
  - Text detection and recognition in images
  - Keyword-based text masking
  - Support for multiple languages
  - Confidence threshold filtering

### Video Processing
- `scene_detect.py`: Scene detection utilities
  - Video scene boundary detection
  - Keyframe extraction
  - Scene transition analysis

### Image Processing
- `fill_in_mask.py`: Image inpainting utilities
  - Mask filling and image completion
  - Region-based image editing

### Natural Language Processing
- `nltk_utils.py`: NLP utilities
  - Text preprocessing
  - Tokenization and lemmatization
  - Language model integration

### Area Analysis
- `primary_area.py`: Area detection and analysis
  - Primary region detection
  - Area-based feature extraction
  - Spatial analysis utilities

## Model Configuration
The project supports multiple large language models for video understanding:
- InternVL-2.5-8B-MPO
- InternVL3-38B
- Qwen2.5-VL-72B-Instruct-AWQ
You may modify the config files (in folder `/config/`) to run any other models supported by vLLM.

## Citation
```
@inproceedings{chen2025vibevideototextinformationbottleneck,
    title={VIBE: Annotation-Free Video-to-Text Information Bottleneck Evaluation for {TL};{DR}},
    author={Shenghui Chen and Po-han Li and Sandeep Chichali and Ufuk Topcu},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year={2025},
    url={https://openreview.net/forum?id=C35FCYZBXp}
}
```
