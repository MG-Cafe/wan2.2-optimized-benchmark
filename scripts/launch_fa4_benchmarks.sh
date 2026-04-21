#!/bin/bash
# Launch FA4 + SP + FSDP benchmarks on a G4 VM
# This script installs flash-attn v4 with SM120 (Blackwell) support,
# then runs benchmarks that previously failed due to missing FA.
set -x

echo "=== Installing flash-attn v4 with SM120 (Blackwell) support ==="
pip install flash-attn --no-build-isolation 2>&1 | tail -10
echo "flash-attn installed"

# Verify FA works on SM120
python3 -c "from flash_attn import flash_attn_func; print('FA import OK')" 2>&1

echo "=== Installing SageAttention ==="
pip install sageattention 2>&1 | tail -3

echo "=== Setting up benchmark runner ==="
export SGLANG_DIFFUSION_STAGE_LOGGING=true
export NUM_FRAMES=81
export SEED=42

run_sglang() {
  local BENCH_ID=$1
  local FLAGS="$2"
  local EXTRA_ENV="$3"
  for MODEL_TYPE in t2v i2v; do
    echo ""
    echo "###############################################"
    echo "# ${BENCH_ID} - ${MODEL_TYPE} - $(date)"
    echo "###############################################"
    export BENCHMARK_ID="${BENCH_ID}"
    export MODEL_TYPE="${MODEL_TYPE}"
    export SGLANG_FLAGS="${FLAGS}"
    if [ -n "${EXTRA_ENV}" ]; then
      for kv in ${EXTRA_ENV}; do export ${kv}; done
    fi
    bash /scratch/run_benchmark.sh || echo "BENCHMARK ${BENCH_ID} ${MODEL_TYPE} FAILED"
    if [ -n "${EXTRA_ENV}" ]; then
      for kv in ${EXTRA_ENV}; do key=$(echo ${kv} | cut -d= -f1); unset ${key}; done
    fi
  done
}

echo ""
echo "============================================="
echo "  FA4 + SP Benchmarks (Blackwell SM120)"
echo "============================================="

# 1. FA backend only (no SP) - compare with SageAttn and default SDPA
run_sglang "fa_tp4" "--num-gpus 4 --tp-size 4 --attention-backend fa" ""
run_sglang "fa_tp8" "--num-gpus 8 --tp-size 8 --attention-backend fa" ""

# 2. P2P + FA combined
run_sglang "p2p_fa_tp4" "--num-gpus 4 --tp-size 4 --attention-backend fa" "NCCL_P2P_LEVEL=5"
run_sglang "p2p_fa_tp8" "--num-gpus 8 --tp-size 8 --attention-backend fa" "NCCL_P2P_LEVEL=5"

# 3. SP with FA backend (Ring Attention should now work!)
run_sglang "sp_fa_4gpu_u2r2" "--num-gpus 4 --sp-degree 4 --ulysses-degree 2 --ring-degree 2 --attention-backend fa" ""
run_sglang "sp_fa_8gpu_u4r2" "--num-gpus 8 --sp-degree 8 --ulysses-degree 4 --ring-degree 2 --attention-backend fa" ""

# 4. P2P + SP + FA combined
run_sglang "p2p_sp_fa_8gpu" "--num-gpus 8 --sp-degree 8 --ulysses-degree 4 --ring-degree 2 --attention-backend fa" "NCCL_P2P_LEVEL=5"

echo ""
echo "============================================="
echo "  torchrun + FSDP + Ulysses Benchmarks"
echo "============================================="

# 5. torchrun with FSDP + Ulysses (open-source Wan2.2 generate.py)
echo "Setting up Wan2.2 for torchrun..."
cd /scratch
if [ ! -d Wan2.2 ]; then
  git clone https://github.com/Wan-Video/Wan2.2.git
fi
cd Wan2.2
pip install easydict ftfy decord librosa peft 2>&1 | tail -3
# Patch flash_attn import
sed -i 's/except ModuleNotFoundError:/except (ModuleNotFoundError, ImportError):/' wan/utils/prompt_extend.py 2>/dev/null
pip install -e . 2>&1 | tail -3

# Download model if needed
pip install huggingface-hub 2>/dev/null
if [ ! -d /scratch/models/Wan2.2-T2V ]; then
  mkdir -p /scratch/models
  huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir /scratch/models/Wan2.2-T2V
fi

export NCCL_P2P_LEVEL=5

# FSDP + Ulysses on single node (8 GPUs)
echo "=== torchrun FSDP + Ulysses_size=8 ==="
mkdir -p /scratch/results/torchrun_fsdp_ulysses8
START_TS=$(date +%s%N)
torchrun --nproc_per_node=8 \
  /scratch/Wan2.2/generate.py \
  --task t2v-A14B \
  --ckpt_dir /scratch/models/Wan2.2-T2V \
  --t5_fsdp --dit_fsdp \
  --ulysses_size 8 \
  --size "1280*720" --frame_num 81 --sample_steps 40 --base_seed 42 \
  --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard." \
  --save_file /scratch/results/torchrun_fsdp_ulysses8/t2v_output.mp4 \
  2>&1 | tee /scratch/results/torchrun_fsdp_ulysses8/t2v_output.log
END_TS=$(date +%s%N)
echo $(( (END_TS - START_TS) / 1000000 )) > /scratch/results/torchrun_fsdp_ulysses8/t2v_wall_time_ms.txt

# FSDP only (no Ulysses) - for comparison
echo "=== torchrun FSDP only ==="
mkdir -p /scratch/results/torchrun_fsdp_only
START_TS=$(date +%s%N)
torchrun --nproc_per_node=8 \
  /scratch/Wan2.2/generate.py \
  --task t2v-A14B \
  --ckpt_dir /scratch/models/Wan2.2-T2V \
  --t5_fsdp --dit_fsdp \
  --size "1280*720" --frame_num 81 --sample_steps 40 --base_seed 42 \
  --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard." \
  --save_file /scratch/results/torchrun_fsdp_only/t2v_output.mp4 \
  2>&1 | tee /scratch/results/torchrun_fsdp_only/t2v_output.log
END_TS=$(date +%s%N)
echo $(( (END_TS - START_TS) / 1000000 )) > /scratch/results/torchrun_fsdp_only/t2v_wall_time_ms.txt

echo ""
echo "=== ALL FA4 + SP + FSDP BENCHMARKS COMPLETE at $(date) ==="
