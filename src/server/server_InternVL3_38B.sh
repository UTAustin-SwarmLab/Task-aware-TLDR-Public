#!/bin/bash

# one 448x448 image = 256 tokens
# max 17216 tokens with 0.95 GPU memory utilization

MODEL="OpenGVLab/InternVL3-38B"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES="0,1"
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NUM_GPUS=2
PORT=8000
vllm serve $MODEL \
    --tensor-parallel-size $NUM_GPUS \
    --port $PORT \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --max-model-len 8000 \
    --enforce-eager \
    --limit-mm-per-prompt image=20 \
    # --max-model-len 81920