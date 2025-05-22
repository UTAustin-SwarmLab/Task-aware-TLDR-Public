#!/bin/bash

MODEL="Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES="1"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NUM_GPUS=1 # Reduced from 4 to 2
PORT=8000
vllm serve $MODEL \
    --tensor-parallel-size $NUM_GPUS \
    --port $PORT \
    --trust-remote-code \
    --dtype float16 \
    --quantization awq \
    --gpu-memory-utilization 0.95 \
    --max-model-len 7500 \
    --enforce-eager \
    --limit-mm-per-prompt image=20,video=0 \
    # --cpu-offload-gb 16 \