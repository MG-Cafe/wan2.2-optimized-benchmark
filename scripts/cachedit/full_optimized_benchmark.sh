#!/bin/bash
set -x

echo "========================================="
echo "  Wan2.2 Full Optimized Benchmark Suite"
echo "  8 configs x T2V + I2V = 16 runs"
echo "  Date: $(date)"
echo "========================================="

# ========== INSTALL DEPENDENCIES ==========
echo "=== Installing dependencies ==="
pip install --upgrade 'diffusers>=0.36.0' 2>&1 | tail -1
pip install git+https://github.com/vipshop/cache-dit.git 2>&1 | tail -1
pip install accelerate einops ftfy opencv-python-headless imageio-ffmpeg transformers 2>&1 | tail -1

# Clone Wan2.2 for FSDP+Ulysses benchmarks (configs 1-3)
cd /scratch
if [ ! -d Wan2.2 ]; then
  git clone https://github.com/Wan-Video/Wan2.2.git
fi
cd /scratch/Wan2.2
pip install -e . 2>&1 | tail -1
# Fix prompt_extend import issue
sed -i 's/except ModuleNotFoundError:/except (ModuleNotFoundError, ImportError):/' wan/utils/prompt_extend.py 2>/dev/null

echo "DEPS_INSTALLED at $(date)"

# ========== CONSTANTS ==========
export NCCL_P2P_LEVEL=5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

T2V_PROMPT="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes toward the camera."
I2V_PROMPT="A curious raccoon peeks from behind a tree, eyes wide with wonder."
T2V_MODEL_CD="wan2.2_t2v"
I2V_MODEL_CD="wan2.2_i2v"

# Model paths for generate.py (will be downloaded by HF)
T2V_CKPT="/scratch/models/Wan2.2-T2V-A14B"
I2V_CKPT="/scratch/models/Wan2.2-I2V-A14B"

# Download models if not cached
mkdir -p /scratch/models
if [ ! -d "$T2V_CKPT" ]; then
  echo "Downloading T2V model..."
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Wan-AI/Wan2.2-T2V-A14B', local_dir='$T2V_CKPT')" 2>&1 | tail -3
fi
if [ ! -d "$I2V_CKPT" ]; then
  echo "Downloading I2V model..."
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Wan-AI/Wan2.2-I2V-A14B', local_dir='$I2V_CKPT')" 2>&1 | tail -3
fi

echo "Models ready at $(date)"

# ========== BENCHMARK FUNCTIONS ==========

# Function for Wan2.2 generate.py (FSDP + Ulysses)
run_wan22() {
  local ID=$1 TASK=$2 GPUS=$3 STEPS=$4 CKPT=$5 MODE=$6
  local DIR=/scratch/results/${ID}
  mkdir -p $DIR
  echo ""
  echo "###############################################"
  echo "# ${ID} (${MODE}) - GPUs=${GPUS} Steps=${STEPS}"
  echo "# Started: $(date)"
  echo "###############################################"
  local START=$(date +%s%N)
  torchrun --nproc_per_node=$GPUS /scratch/Wan2.2/generate.py \
    --task $TASK \
    --ckpt_dir $CKPT \
    --t5_fsdp --dit_fsdp \
    --ulysses_size $GPUS \
    --size "1280*720" \
    --frame_num 81 \
    --sample_steps $STEPS \
    --base_seed 42 \
    --prompt "$T2V_PROMPT" \
    --save_file $DIR/${MODE}_output.mp4 \
    2>&1 | tee $DIR/${MODE}_output.log
  local END=$(date +%s%N)
  local WALL=$(( (END - START) / 1000000 ))
  echo $WALL > $DIR/${MODE}_wall_time_ms.txt
  echo "=== ${ID} ${MODE} completed in ${WALL}ms ==="
}

# Function for cache-dit (TP + cache)
run_cachedit() {
  local ID=$1 MODEL=$2 GPUS=$3 STEPS=$4 EXTRA=$5 PROMPT="$6" MODE=$7
  local DIR=/scratch/results/${ID}
  mkdir -p $DIR
  echo ""
  echo "###############################################"
  echo "# ${ID} (${MODE}) - GPUs=${GPUS} Steps=${STEPS}"
  echo "# Started: $(date)"
  echo "###############################################"
  local START=$(date +%s%N)
  torchrun --nproc_per_node=$GPUS -m cache_dit.generate $MODEL \
    --parallel-type tp $EXTRA \
    --steps $STEPS --height 720 --width 1280 --num-frames 81 --seed 42 \
    --prompt "$PROMPT" \
    --save-path $DIR/${MODE}_output.mp4 \
    2>&1 | tee $DIR/${MODE}_output.log
  local END=$(date +%s%N)
  local WALL=$(( (END - START) / 1000000 ))
  echo $WALL > $DIR/${MODE}_wall_time_ms.txt
  echo "=== ${ID} ${MODE} completed in ${WALL}ms ==="
}

# ========== CONFIG 1: FSDP + Ulysses=4, 40 steps ==========
run_wan22 "cfg1_fsdp_u4_40s" "t2v-A14B" 4 40 "$T2V_CKPT" "t2v"
run_wan22 "cfg1_fsdp_u4_40s" "i2v-A14B" 4 40 "$I2V_CKPT" "i2v"

# ========== CONFIG 2: FSDP + Ulysses=8, 40 steps ==========
run_wan22 "cfg2_fsdp_u8_40s" "t2v-A14B" 8 40 "$T2V_CKPT" "t2v"
run_wan22 "cfg2_fsdp_u8_40s" "i2v-A14B" 8 40 "$I2V_CKPT" "i2v"

# ========== CONFIG 3: FSDP + Ulysses=8, 5 steps ==========
run_wan22 "cfg3_fsdp_u8_5s" "t2v-A14B" 8 5 "$T2V_CKPT" "t2v"
run_wan22 "cfg3_fsdp_u8_5s" "i2v-A14B" 8 5 "$I2V_CKPT" "i2v"

# ========== CONFIG 4: cache-dit TP=4 + cache, 40 steps ==========
run_cachedit "cfg4_cd_tp4_cache_40s" "$T2V_MODEL_CD" 4 40 "--cache" "$T2V_PROMPT" "t2v"
run_cachedit "cfg4_cd_tp4_cache_40s" "$I2V_MODEL_CD" 4 40 "--cache" "$I2V_PROMPT" "i2v"

# ========== CONFIG 5: cache-dit TP=8 + cache, 40 steps ==========
run_cachedit "cfg5_cd_tp8_cache_40s" "$T2V_MODEL_CD" 8 40 "--cache" "$T2V_PROMPT" "t2v"
run_cachedit "cfg5_cd_tp8_cache_40s" "$I2V_MODEL_CD" 8 40 "--cache" "$I2V_PROMPT" "i2v"

# ========== CONFIG 6: cache-dit TP=8 + cache + compile, 40 steps ==========
run_cachedit "cfg6_cd_tp8_cache_compile_40s" "$T2V_MODEL_CD" 8 40 "--cache --compile" "$T2V_PROMPT" "t2v"
run_cachedit "cfg6_cd_tp8_cache_compile_40s" "$I2V_MODEL_CD" 8 40 "--cache --compile" "$I2V_PROMPT" "i2v"

# ========== CONFIG 7: cache-dit TP=8 + cache, 5 steps ==========
run_cachedit "cfg7_cd_tp8_cache_5s" "$T2V_MODEL_CD" 8 5 "--cache" "$T2V_PROMPT" "t2v"
run_cachedit "cfg7_cd_tp8_cache_5s" "$I2V_MODEL_CD" 8 5 "--cache" "$I2V_PROMPT" "i2v"

# ========== CONFIG 8: cache-dit TP=8 + cache + compile, 5 steps ==========
run_cachedit "cfg8_cd_tp8_cache_compile_5s" "$T2V_MODEL_CD" 8 5 "--cache --compile" "$T2V_PROMPT" "t2v"
run_cachedit "cfg8_cd_tp8_cache_compile_5s" "$I2V_MODEL_CD" 8 5 "--cache --compile" "$I2V_PROMPT" "i2v"

# ========== SUMMARY ==========
echo ""
echo "========================================="
echo "  ALL 16 BENCHMARKS COMPLETE"
echo "  Date: $(date)"
echo "========================================="
echo ""
echo "RESULTS SUMMARY:"
for dir in /scratch/results/cfg*/; do
  name=$(basename $dir)
  t2v=$(cat ${dir}/t2v_wall_time_ms.txt 2>/dev/null || echo "FAIL")
  i2v=$(cat ${dir}/i2v_wall_time_ms.txt 2>/dev/null || echo "FAIL")
  echo "${name}: T2V=${t2v}ms I2V=${i2v}ms"
done
