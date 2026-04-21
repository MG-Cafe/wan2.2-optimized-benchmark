#!/bin/bash
set -x
echo "=== Installing dependencies ==="
pip install easydict ftfy decord librosa dashscope imageio imageio-ffmpeg opencv-python-headless 'numpy<2' 2>&1 | tail -5
echo "DEPS_OK"

echo "=== Setting NCCL config ==="
export NCCL_P2P_LEVEL=5
export NCCL_SOCKET_IFNAME=ens4
export NCCL_IB_DISABLE=1

echo "=== Starting torchrun worker ==="
mkdir -p /scratch/results/multihost_16gpu
torchrun \
  --nproc_per_node=8 \
  --nnodes=2 \
  --node_rank=1 \
  --master_addr=${MASTER_ADDR} \
  --master_port=29500 \
  /scratch/Wan2.2/generate.py \
  --task t2v-A14B \
  --ckpt_dir /scratch/models/Wan2.2-T2V \
  --ulysses_size 16 \
  --size "1280*720" \
  --frame_num 81 \
  --sample_steps 40 \
  --base_seed 42 \
  --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard." \
  2>&1 | tee /scratch/results/multihost_16gpu/worker.log

echo "=== Worker complete ==="
