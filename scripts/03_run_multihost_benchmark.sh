#!/bin/bash
#
# Multi-Host Benchmark Script using Wan2.2 open-source generate.py with torchrun
#
# This script runs on BOTH nodes. It uses torchrun for distributed inference
# across 2 G4 nodes (16 GPUs total) using Ulysses sequence parallelism.
#
# Usage:
#   On master node (RANK 0):
#     MASTER_ADDR=<master_ip> NODE_RANK=0 ULYSSES_SIZE=8 bash 03_run_multihost_benchmark.sh
#   On worker node (RANK 1):
#     MASTER_ADDR=<master_ip> NODE_RANK=1 ULYSSES_SIZE=8 bash 03_run_multihost_benchmark.sh
#
# Required env vars:
#   MASTER_ADDR   - IP address of the master node
#   NODE_RANK     - 0 for master, 1 for worker
#   ULYSSES_SIZE  - Ulysses parallelism degree (e.g., 4 or 8 or 16)
#
# Optional env vars:
#   MASTER_PORT   - Port for distributed communication (default: 29500)
#   CKPT_DIR      - Checkpoint directory (default: /scratch/models/Wan2.2)
#   TASK          - Task type: t2v-A14B or i2v-A14B (default: t2v-A14B)
#
set -euo pipefail

MASTER_ADDR="${MASTER_ADDR:?ERROR: Set MASTER_ADDR}"
NODE_RANK="${NODE_RANK:?ERROR: Set NODE_RANK (0 or 1)}"
ULYSSES_SIZE="${ULYSSES_SIZE:?ERROR: Set ULYSSES_SIZE}"
MASTER_PORT="${MASTER_PORT:-29500}"
CKPT_DIR="${CKPT_DIR:-/scratch/models/Wan2.2}"
TASK="${TASK:-t2v-A14B}"
NPROC_PER_NODE=8
NNODES=2

RESULTS_DIR="/scratch/results/multihost_${ULYSSES_SIZE}_${TASK}"
mkdir -p "${RESULTS_DIR}"

echo "============================================="
echo "  Multi-Host Benchmark"
echo "  Task: ${TASK}"
echo "  Ulysses Size: ${ULYSSES_SIZE}"
echo "  Nodes: ${NNODES}, GPUs/node: ${NPROC_PER_NODE}"
echo "  Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  Node Rank: ${NODE_RANK}"
echo "  Started: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "============================================="

# Enable G4 P2P optimization
export NCCL_P2P_LEVEL=5
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

# Record system info
{
  echo "=== System Info ==="
  nvidia-smi
  echo "=== GPU Topology ==="
  nvidia-smi topo -m 2>/dev/null || true
  echo "=== Network ==="
  ip addr show 2>/dev/null | grep -E "inet |eth" || true
  echo "=== NCCL Config ==="
  echo "NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL}"
  echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
  echo "NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
} > "${RESULTS_DIR}/system_info_node${NODE_RANK}.log" 2>&1

# Clone Wan2.2 if not already present
if [ ! -d /scratch/Wan2.2 ]; then
  echo "Cloning Wan2.2 repository..."
  cd /scratch
  git clone https://github.com/Wan-Video/Wan2.2.git
  cd Wan2.2
  pip install -e . 2>&1 | tail -5
fi

# Download model weights if not present
if [ ! -d "${CKPT_DIR}" ]; then
  echo "Downloading Wan2.2 model weights..."
  pip install huggingface-hub 2>/dev/null
  if [ "${TASK}" = "t2v-A14B" ]; then
    huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir "${CKPT_DIR}"
  else
    huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir "${CKPT_DIR}"
  fi
fi

# Download I2V input image if needed
if [ "${TASK}" = "i2v-A14B" ]; then
  mkdir -p /scratch/assets
  if [ ! -f /scratch/assets/i2v_input.JPG ]; then
    curl -sL 'https://raw.githubusercontent.com/Wan-Video/Wan2.2/main/examples/i2v_input.JPG' -o /scratch/assets/i2v_input.JPG
  fi
  IMAGE_FLAG="--image /scratch/assets/i2v_input.JPG"
else
  IMAGE_FLAG=""
fi

# Set prompt based on task
if [ "${TASK}" = "t2v-A14B" ]; then
  PROMPT="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline intricate details and the refreshing atmosphere of the seaside."
else
  PROMPT="A curious raccoon"
fi

START_TS=$(date +%s%N)

cd /scratch/Wan2.2

torchrun \
  --nproc_per_node=${NPROC_PER_NODE} \
  --nnodes=${NNODES} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  generate.py \
  --task ${TASK} \
  --ckpt_dir "${CKPT_DIR}" \
  --ulysses_size ${ULYSSES_SIZE} \
  --size "1280*720" \
  --frame_num 81 \
  --sample_steps 40 \
  --base_seed 42 \
  --prompt "${PROMPT}" \
  ${IMAGE_FLAG} \
  --save_file "${RESULTS_DIR}/${TASK}_output.mp4" \
  2>&1 | tee "${RESULTS_DIR}/${TASK}_node${NODE_RANK}.log"

END_TS=$(date +%s%N)
ELAPSED_MS=$(( (END_TS - START_TS) / 1000000 ))

echo ""
echo "============================================="
echo "  Multi-Host Benchmark COMPLETE"
echo "  Task: ${TASK}, Ulysses: ${ULYSSES_SIZE}"
echo "  Node Rank: ${NODE_RANK}"
echo "  Wall time: ${ELAPSED_MS}ms ($(( ELAPSED_MS / 1000 ))s)"
echo "  Finished: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "============================================="

echo "${ELAPSED_MS}" > "${RESULTS_DIR}/${TASK}_node${NODE_RANK}_wall_time_ms.txt"
