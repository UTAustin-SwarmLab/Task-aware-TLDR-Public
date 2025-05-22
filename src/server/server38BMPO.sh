#!/bin/bash

# one 448x448 image = 256 tokens

MODEL="OpenGVLab/InternVL2_5-38B-MPO" # 4, 26, 38, 78B, OpenGVLab/InternVL2_5-38B-MPO
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NUM_GPUS=4 # Reduced from 4 to 2
PORT=8000
vllm serve $MODEL \
    --tensor-parallel-size $NUM_GPUS \
    --port $PORT \
    --trust-remote-code \
    --limit-mm-per-prompt image=8 \
    --gpu-memory-utilization 0.95 \
    --enforce-eager \
