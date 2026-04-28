#!/bin/bash
set -x

echo "========================================="
echo "  Fix & Rerun Failed Benchmarks"
echo "  FSDP configs + I2V configs"
echo "  Date: $(date)"
echo "========================================="

# Install ALL dependencies including easydict
pip install --upgrade 'diffusers>=0.36.0' 2>&1 | tail -1
pip install git+https://github.com/vipshop/cache-dit.git 2>&1 | tail -1
pip install accelerate einops ftfy opencv-python-headless imageio-ffmpeg transformers 2>&1 | tail -1
pip install easydict peft decord librosa dashscope 2>&1 | tail -1
echo "DEPS_INSTALLED at $(date)"

# Clone Wan2.2
cd /scratch
if [ ! -d Wan2.2 ]; then
  git clone https://github.com/Wan-Video/Wan2.2.git
fi
cd /scratch/Wan2.2
pip install -e . 2>&1 | tail -1
sed -i 's/except ModuleNotFoundError:/except (ModuleNotFoundError, ImportError):/' wan/utils/prompt_extend.py 2>/dev/null

# Download models (non-diffusers format for generate.py)
mkdir -p /scratch/models
T2V_CKPT="/scratch/models/Wan2.2-T2V-A14B"
I2V_CKPT="/scratch/models/Wan2.2-I2V-A14B"

if [ ! -d "$T2V_CKPT" ] || [ ! -f "$T2V_CKPT/config.json" ]; then
  echo "Downloading T2V model..."
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Wan-AI/Wan2.2-T2V-A14B', local_dir='$T2V_CKPT')" 2>&1 | tail -3
fi
if [ ! -d "$I2V_CKPT" ] || [ ! -f "$I2V_CKPT/config.json" ]; then
  echo "Downloading I2V model..."
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Wan-AI/Wan2.2-I2V-A14B', local_dir='$I2V_CKPT')" 2>&1 | tail -3
fi
echo "Models ready at $(date)"

export NCCL_P2P_LEVEL=5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

T2V_PROMPT="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes toward the camera."
I2V_PROMPT="A curious raccoon peeks from behind a tree, eyes wide with wonder."

# ========== FSDP + Ulysses (Wan2.2 generate.py) ==========

run_wan22() {
  local ID=$1 TASK=$2 GPUS=$3 STEPS=$4 CKPT=$5 PROMPT="$6" MODE=$7
  local DIR=/scratch/results/${ID}
  mkdir -p $DIR
  echo ""
  echo "### ${ID} (${MODE}) - GPUs=${GPUS} Steps=${STEPS} - $(date) ###"
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
    --prompt "$PROMPT" \
    --save_file $DIR/${MODE}_output.mp4 \
    2>&1 | tee $DIR/${MODE}_output.log
  local END=$(date +%s%N)
  local WALL=$(( (END - START) / 1000000 ))
  echo $WALL > $DIR/${MODE}_wall_time_ms.txt
  echo "=== ${ID} ${MODE} done in ${WALL}ms ==="
}

run_cachedit() {
  local ID=$1 MODEL=$2 GPUS=$3 STEPS=$4 EXTRA=$5 PROMPT="$6" MODE=$7
  local DIR=/scratch/results/${ID}
  mkdir -p $DIR
  echo ""
  echo "### ${ID} (${MODE}) - GPUs=${GPUS} Steps=${STEPS} - $(date) ###"
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
  echo "=== ${ID} ${MODE} done in ${WALL}ms ==="
}

# FSDP configs (were missing easydict)
run_wan22 "cfg1_fsdp_u4_40s" "t2v-A14B" 4 40 "$T2V_CKPT" "$T2V_PROMPT" "t2v"
run_wan22 "cfg1_fsdp_u4_40s" "i2v-A14B" 4 40 "$I2V_CKPT" "$I2V_PROMPT" "i2v"
run_wan22 "cfg2_fsdp_u8_40s" "t2v-A14B" 8 40 "$T2V_CKPT" "$T2V_PROMPT" "t2v"
run_wan22 "cfg2_fsdp_u8_40s" "i2v-A14B" 8 40 "$I2V_CKPT" "$I2V_PROMPT" "i2v"
run_wan22 "cfg3_fsdp_u8_5s" "t2v-A14B" 8 5 "$T2V_CKPT" "$T2V_PROMPT" "t2v"
run_wan22 "cfg3_fsdp_u8_5s" "i2v-A14B" 8 5 "$I2V_CKPT" "$I2V_PROMPT" "i2v"

# I2V configs (were disk space issue)
run_cachedit "cfg5_cd_tp8_cache_40s" "wan2.2_i2v" 8 40 "--cache" "$I2V_PROMPT" "i2v"
run_cachedit "cfg6_cd_tp8_cache_compile_40s" "wan2.2_i2v" 8 40 "--cache --compile" "$I2V_PROMPT" "i2v"
run_cachedit "cfg7_cd_tp8_cache_5s" "wan2.2_i2v" 8 5 "--cache" "$I2V_PROMPT" "i2v"
run_cachedit "cfg8_cd_tp8_cache_compile_5s" "wan2.2_i2v" 8 5 "--cache --compile" "$I2V_PROMPT" "i2v"

echo ""
echo "=== ALL RERUNS COMPLETE at $(date) ==="
for dir in /scratch/results/cfg*/; do
  name=$(basename $dir)
  t2v=$(cat ${dir}/t2v_wall_time_ms.txt 2>/dev/null || echo "?")
  i2v=$(cat ${dir}/i2v_wall_time_ms.txt 2>/dev/null || echo "?")
  echo "${name}: T2V=${t2v}ms I2V=${i2v}ms"
done
