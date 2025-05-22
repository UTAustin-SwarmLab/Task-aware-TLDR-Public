#!/bin/bash

# one 448x448 image = 256 tokens

MODEL="OpenGVLab/InternVL2_5-1B-MPO"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES="3"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NUM_GPUS=1 # Reduced from 4 to 2
PORT=8000
vllm serve $MODEL \
    --tensor-parallel-size $NUM_GPUS \
    --port $PORT \
    --trust-remote-code \
    --limit-mm-per-prompt image=8 \
    --enforce-eager \
