#!/bin/bash
set -x

echo "========================================="
echo "  cache-dit Full Benchmark Suite"
echo "  5 configs x T2V + I2V = 10 runs"
echo "  Date: $(date)"
echo "========================================="

pip install --upgrade 'diffusers>=0.36.0' 2>&1 | tail -1
pip install git+https://github.com/vipshop/cache-dit.git 2>&1 | tail -1
pip install accelerate einops ftfy opencv-python-headless imageio-ffmpeg 2>&1 | tail -1
echo "DEPS_INSTALLED"

export NCCL_P2P_LEVEL=5
T2V_PROMPT="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes toward the camera."
I2V_PROMPT="A curious raccoon peeks from behind a tree, eyes wide with wonder."

run_bench() {
  local ID=$1 MODEL=$2 GPUS=$3 STEPS=$4 EXTRA=$5 PROMPT="$6" MODE=$7
  local DIR=/scratch/results/${ID}
  mkdir -p $DIR
  echo ""
  echo "### ${ID} (${MODE}) - $(date) ###"
  local START=$(date +%s%N)
  if [ $GPUS -gt 1 ]; then
    torchrun --nproc_per_node=$GPUS -m cache_dit.generate $MODEL \
      --parallel-type tp $EXTRA \
      --steps $STEPS --height 720 --width 1280 --num-frames 81 --seed 42 \
      --prompt "$PROMPT" \
      --save-path $DIR/${MODE}_output.mp4 \
      2>&1 | tee $DIR/${MODE}_output.log
  else
    python3 -m cache_dit.generate $MODEL \
      $EXTRA \
      --steps $STEPS --height 720 --width 1280 --num-frames 81 --seed 42 \
      --prompt "$PROMPT" \
      --save-path $DIR/${MODE}_output.mp4 \
      2>&1 | tee $DIR/${MODE}_output.log
  fi
  local END=$(date +%s%N)
  local WALL=$(( (END - START) / 1000000 ))
  echo $WALL > $DIR/${MODE}_wall_time_ms.txt
  echo "=== ${ID} ${MODE} done in ${WALL}ms ==="
}

run_bench cd_tp4_cache_40s wan2.2_t2v 4 40 "--cache" "$T2V_PROMPT" t2v
run_bench cd_tp4_cache_40s wan2.2_i2v 4 40 "--cache" "$I2V_PROMPT" i2v
run_bench cd_tp8_cache_40s wan2.2_t2v 8 40 "--cache" "$T2V_PROMPT" t2v
run_bench cd_tp8_cache_40s wan2.2_i2v 8 40 "--cache" "$I2V_PROMPT" i2v
run_bench cd_tp8_cache_compile_40s wan2.2_t2v 8 40 "--cache --compile" "$T2V_PROMPT" t2v
run_bench cd_tp8_cache_compile_40s wan2.2_i2v 8 40 "--cache --compile" "$I2V_PROMPT" i2v
run_bench cd_tp8_cache_5s wan2.2_t2v 8 5 "--cache" "$T2V_PROMPT" t2v
run_bench cd_tp8_cache_5s wan2.2_i2v 8 5 "--cache" "$I2V_PROMPT" i2v
run_bench cd_tp8_cache_compile_5s wan2.2_t2v 8 5 "--cache --compile" "$T2V_PROMPT" t2v
run_bench cd_tp8_cache_compile_5s wan2.2_i2v 8 5 "--cache --compile" "$I2V_PROMPT" i2v

echo ""
echo "=== ALL 10 BENCHMARKS COMPLETE at $(date) ==="
for dir in /scratch/results/cd_tp*/; do
  name=$(basename $dir)
  t2v=$(cat ${dir}/t2v_wall_time_ms.txt 2>/dev/null || echo "?")
  i2v=$(cat ${dir}/i2v_wall_time_ms.txt 2>/dev/null || echo "?")
  echo "${name}: T2V=${t2v}ms I2V=${i2v}ms"
done
