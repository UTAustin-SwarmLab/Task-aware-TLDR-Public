#!/bin/bash

# one 448x448 image = 256 tokens
# max 17216 tokens with 0.95 GPU memory utilization

MODEL="OpenGVLab/InternVL2_5-78B-AWQ" # 4, 26, 38, 78B, OpenGVLab/InternVL2_5-38B-MPO
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES="2"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NUM_GPUS=1 # Reduced from 4 to 2
PORT=8000
vllm serve $MODEL \
    --tensor-parallel-size $NUM_GPUS \
    --port $PORT \
    --trust-remote-code \
    --dtype float16 \
    --quantization awq \
    --cpu-offload-gb 16 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 16216 \
    --enforce-eager \
    --limit-mm-per-prompt image=64 \
    # --max-model-len 81920