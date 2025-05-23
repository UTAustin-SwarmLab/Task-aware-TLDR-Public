#!/bin/bash

MODEL="OpenGVLab/InternVL3-8B"
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
