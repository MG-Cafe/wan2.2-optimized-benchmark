#!/bin/bash
set -x
echo "=== Installing dependencies ==="
pip install easydict ftfy decord librosa dashscope imageio imageio-ffmpeg opencv-python-headless 'numpy<2' 2>&1 | tail -5
echo "DEPS_OK"

echo "=== Setting NCCL config ==="
export NCCL_P2P_LEVEL=5
export NCCL_SOCKET_IFNAME=ens4
export NCCL_IB_DISABLE=1

echo "=== Starting torchrun master ==="
mkdir -p /scratch/results/multihost_16gpu
START_TS=$(date +%s%N)
torchrun \
  --nproc_per_node=8 \
  --nnodes=2 \
  --node_rank=0 \
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
  --save_file /scratch/results/multihost_16gpu/t2v_output.mp4 \
  2>&1 | tee /scratch/results/multihost_16gpu/master.log

END_TS=$(date +%s%N)
ELAPSED_MS=$(( (END_TS - START_TS) / 1000000 ))
echo "${ELAPSED_MS}" > /scratch/results/multihost_16gpu/t2v_wall_time_ms.txt
echo "=== Master complete in ${ELAPSED_MS}ms ($(( ELAPSED_MS / 1000 ))s) ==="
