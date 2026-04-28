#!/bin/bash
set -x

echo "=== Installing flash-attn from source with SM120 ==="
pip install flash-attn --no-build-isolation 2>&1 | tail -5

echo "=== Verify flash_attn_varlen_func ==="
python3 -c "from flash_attn import flash_attn_varlen_func; print('FA varlen OK')" 2>&1

echo "=== Installing deps ==="
pip install easydict peft decord librosa dashscope 2>&1 | tail -1
pip install --upgrade 'diffusers>=0.36.0' 2>&1 | tail -1

cd /scratch/Wan2.2
pip install -e . 2>&1 | tail -1
sed -i 's/except ModuleNotFoundError:/except (ModuleNotFoundError, ImportError):/' wan/utils/prompt_extend.py 2>/dev/null

export NCCL_P2P_LEVEL=5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

T2V_PROMPT="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes toward the camera."
I2V_PROMPT="A curious raccoon peeks from behind a tree, eyes wide with wonder."
T2V_CKPT="/scratch/models/Wan2.2-T2V-A14B"
I2V_CKPT="/scratch/models/Wan2.2-I2V-A14B"

# Download models if needed
if [ ! -d "$T2V_CKPT" ] || [ ! -f "$T2V_CKPT/config.json" ]; then
  echo "Downloading T2V model..."
  python3 << 'PYEOF'
from huggingface_hub import snapshot_download
snapshot_download("Wan-AI/Wan2.2-T2V-A14B", local_dir="/scratch/models/Wan2.2-T2V-A14B")
PYEOF
fi
if [ ! -d "$I2V_CKPT" ] || [ ! -f "$I2V_CKPT/config.json" ]; then
  echo "Downloading I2V model..."
  python3 << 'PYEOF'
from huggingface_hub import snapshot_download
snapshot_download("Wan-AI/Wan2.2-I2V-A14B", local_dir="/scratch/models/Wan2.2-I2V-A14B")
PYEOF
fi

echo "Models ready at $(date)"

run_wan22() {
  local ID=$1 TASK=$2 GPUS=$3 STEPS=$4 CKPT=$5 MODE=$6
  local DIR=/scratch/results/${ID}
  mkdir -p $DIR
  echo ""
  echo "### ${ID} (${MODE}) - GPUs=${GPUS} Steps=${STEPS} - $(date) ###"
  local START=$(date +%s%N)
  torchrun --nproc_per_node=$GPUS /scratch/Wan2.2/generate.py \
    --task $TASK --ckpt_dir $CKPT \
    --t5_fsdp --dit_fsdp --ulysses_size $GPUS \
    --size "1280*720" --frame_num 81 --sample_steps $STEPS --base_seed 42 \
    --prompt "$T2V_PROMPT" --save_file $DIR/${MODE}_output.mp4 \
    2>&1 | tee $DIR/${MODE}_output.log
  local END=$(date +%s%N)
  local WALL=$(( (END - START) / 1000000 ))
  echo $WALL > $DIR/${MODE}_wall_time_ms.txt
  echo "=== ${ID} ${MODE} done in ${WALL}ms ==="
}

# Config 1: FSDP + Ulysses=4, 40 steps
run_wan22 cfg1_fsdp_u4_40s t2v-A14B 4 40 "$T2V_CKPT" t2v
run_wan22 cfg1_fsdp_u4_40s i2v-A14B 4 40 "$I2V_CKPT" i2v

# Config 2: FSDP + Ulysses=8, 40 steps
run_wan22 cfg2_fsdp_u8_40s t2v-A14B 8 40 "$T2V_CKPT" t2v
run_wan22 cfg2_fsdp_u8_40s i2v-A14B 8 40 "$I2V_CKPT" i2v

# Config 3: FSDP + Ulysses=8, 5 steps
run_wan22 cfg3_fsdp_u8_5s t2v-A14B 8 5 "$T2V_CKPT" t2v
run_wan22 cfg3_fsdp_u8_5s i2v-A14B 8 5 "$I2V_CKPT" i2v

echo ""
echo "=== FSDP BENCHMARKS COMPLETE at $(date) ==="
for dir in /scratch/results/cfg[123]*/; do
  name=$(basename $dir)
  t2v=$(cat ${dir}/t2v_wall_time_ms.txt 2>/dev/null || echo "?")
  i2v=$(cat ${dir}/i2v_wall_time_ms.txt 2>/dev/null || echo "?")
  echo "${name}: T2V=${t2v}ms I2V=${i2v}ms"
done
