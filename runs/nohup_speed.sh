#!/bin/bash
export WANDB_MODE=offline
WANDB_RUN="default" 
export NCCL_SOCKET_IFNAME=enp4s0
export CUDA_VISIBLE_DEVICES=0,1
source cluster_env.sh
source .venv/bin/activate
nohup torchrun --nproc_per_node=2 -m scripts.base_train -- \
    --optim=galore \
    --depth=12 \
    --device-batch-size=1 \
    --num-iterations=1000 \
    --run=galore \
    --window-pattern=L \
    > training.log 2>&1 &

#python -m nanochat.chat --device_batch_size=8 > training.log 2>&1 &

