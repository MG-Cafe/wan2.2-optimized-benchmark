#!/bin/bash
#
# Run a single SGLang benchmark inside the Docker container
#
# Usage (inside container):
#   BENCHMARK_ID=baseline_tp4 MODEL_TYPE=t2v bash /scratch/02_run_sglang_benchmark.sh
#
# Required env vars:
#   BENCHMARK_ID   - Unique identifier for this benchmark run
#   MODEL_TYPE     - "t2v" or "i2v"
#   SGLANG_FLAGS   - SGLang flags (e.g. "--num-gpus 4 --tp-size 4")
#
# Optional env vars:
#   NUM_FRAMES     - Number of frames (default: 81)
#   SEED           - Random seed (default: 42)
#
set -x

BENCHMARK_ID="${BENCHMARK_ID:?ERROR: Set BENCHMARK_ID}"
MODEL_TYPE="${MODEL_TYPE:?ERROR: Set MODEL_TYPE (t2v or i2v)}"
SGLANG_FLAGS="${SGLANG_FLAGS:?ERROR: Set SGLANG_FLAGS}"
NUM_FRAMES="${NUM_FRAMES:-81}"
SEED="${SEED:-42}"
RESULTS_DIR="/scratch/results/${BENCHMARK_ID}"
mkdir -p "${RESULTS_DIR}"

# Scenario parameters matching reference doc:
# - Resolution: 720P (1280x720) - SGLang default for A14B
# - Inference steps: 40 - SGLang default for Wan2.2
# - Guidance scale: default (3.0/4.0 for T2V, 3.5 for I2V)
# - Seed: 42

T2V_PROMPT="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline intricate details and the refreshing atmosphere of the seaside."
I2V_PROMPT="A curious raccoon"

echo "============================================="
echo "  Benchmark: ${BENCHMARK_ID} (${MODEL_TYPE})"
echo "  Flags: ${SGLANG_FLAGS}"
echo "  Frames: ${NUM_FRAMES}, Seed: ${SEED}"
echo "  Started: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "============================================="

# Enable stage logging for timing breakdown
export SGLANG_DIFFUSION_STAGE_LOGGING=true

# Record system info
{
  echo "=== System Info ==="
  nvidia-smi
  echo "=== GPU Topology ==="
  nvidia-smi topo -m 2>/dev/null || true
  echo "=== NCCL P2P Level ==="
  echo "NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-not set}"
  echo "=== Cache-DiT ==="
  echo "SGLANG_CACHE_DIT_ENABLED=${SGLANG_CACHE_DIT_ENABLED:-not set}"
  echo "=== Python/SGLang Version ==="
  python3 -c "import sglang; print('SGLang version:', sglang.__version__)" 2>/dev/null || echo "Version check failed"
} > "${RESULTS_DIR}/system_info.log" 2>&1

START_TS=$(date +%s%N)

if [ "${MODEL_TYPE}" = "t2v" ]; then
  echo "=== Running T2V Benchmark ==="
  sglang generate \
    --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --dit-layerwise-offload false \
    --text-encoder-cpu-offload false \
    --pin-cpu-memory \
    --dit-cpu-offload false \
    ${SGLANG_FLAGS} \
    --prompt "${T2V_PROMPT}" \
    --num-frames ${NUM_FRAMES} \
    --seed ${SEED} \
    --save-output \
    --output-path "${RESULTS_DIR}" \
    2>&1 | tee "${RESULTS_DIR}/t2v_output.log"

elif [ "${MODEL_TYPE}" = "i2v" ]; then
  # Download reference image (from Wan2.2 repo)
  mkdir -p /scratch/assets
  if [ ! -f /scratch/assets/i2v_input.JPG ]; then
    curl -sL 'https://raw.githubusercontent.com/Wan-Video/Wan2.2/main/examples/i2v_input.JPG' -o /scratch/assets/i2v_input.JPG || \
    curl -sL 'https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png' -o /scratch/assets/i2v_input.JPG || \
    python3 -c "from PIL import Image; Image.new('RGB',(1280,720),'blue').save('/scratch/assets/i2v_input.JPG')"
  fi

  echo "=== Running I2V Benchmark ==="
  sglang generate \
    --model-path Wan-AI/Wan2.2-I2V-A14B-Diffusers \
    --image-path /scratch/assets/i2v_input.JPG \
    --dit-layerwise-offload false \
    --text-encoder-cpu-offload false \
    --pin-cpu-memory \
    --dit-cpu-offload false \
    ${SGLANG_FLAGS} \
    --prompt "${I2V_PROMPT}" \
    --num-frames ${NUM_FRAMES} \
    --seed ${SEED} \
    --save-output \
    --output-path "${RESULTS_DIR}" \
    2>&1 | tee "${RESULTS_DIR}/i2v_output.log"
fi

END_TS=$(date +%s%N)
ELAPSED_MS=$(( (END_TS - START_TS) / 1000000 ))

echo ""
echo "============================================="
echo "  Benchmark ${BENCHMARK_ID} (${MODEL_TYPE}) COMPLETE"
echo "  Wall time: ${ELAPSED_MS}ms ($(( ELAPSED_MS / 1000 ))s)"
echo "  Finished: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "============================================="

# Save timing
echo "${ELAPSED_MS}" > "${RESULTS_DIR}/${MODEL_TYPE}_wall_time_ms.txt"
