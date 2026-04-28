#!/bin/bash
# Launch cache-dit optimized benchmarks on a single G4 VM
# Uses cache-dit library for DiT block caching + Ulysses distributed inference
set -x

echo "=== Installing cache-dit and dependencies ==="
pip install -U uv 2>&1 | tail -1
uv pip install transformers accelerate opencv-python-headless einops imageio-ffmpeg ftfy 2>&1 | tail -3
uv pip install git+https://github.com/huggingface/diffusers.git 2>&1 | tail -3
uv pip install git+https://github.com/vipshop/cache-dit.git 2>&1 | tail -3
pip install sageattention 2>&1 | tail -1
echo "=== Dependencies installed ==="

# Clone cache-dit examples
cd /scratch
if [ ! -d cache-dit ]; then
  git clone https://github.com/vipshop/cache-dit.git
fi
cd cache-dit/examples

export NCCL_P2P_LEVEL=5

mkdir -p /scratch/results

echo ""
echo "============================================="
echo "  cache-dit Wan2.2-A14B Benchmarks"
echo "============================================="

# 1. Single GPU with cache-dit (DBCache + TaylorSeer)
echo "=== [1/6] cache-dit 1 GPU ==="
mkdir -p /scratch/results/cachedit_1gpu
START_TS=$(date +%s%N)
python3 -m cache_dit.generate wan2.2_t2v \
  --cache \
  --num_inference_steps 40 \
  --height 720 --width 1280 \
  --num_frames 81 \
  --seed 42 \
  --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard." \
  2>&1 | tee /scratch/results/cachedit_1gpu/t2v_output.log
END_TS=$(date +%s%N)
echo $(( (END_TS - START_TS) / 1000000 )) > /scratch/results/cachedit_1gpu/t2v_wall_time_ms.txt

# 2. 4 GPU with Ulysses + cache-dit
echo "=== [2/6] cache-dit 4 GPU Ulysses ==="
mkdir -p /scratch/results/cachedit_4gpu_ulysses
START_TS=$(date +%s%N)
torchrun --nproc_per_node=4 -m cache_dit.generate wan2.2_t2v \
  --cache \
  --ulysses 4 \
  --num_inference_steps 40 \
  --height 720 --width 1280 \
  --num_frames 81 \
  --seed 42 \
  --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard." \
  2>&1 | tee /scratch/results/cachedit_4gpu_ulysses/t2v_output.log
END_TS=$(date +%s%N)
echo $(( (END_TS - START_TS) / 1000000 )) > /scratch/results/cachedit_4gpu_ulysses/t2v_wall_time_ms.txt

# 3. 8 GPU with Ulysses + cache-dit
echo "=== [3/6] cache-dit 8 GPU Ulysses ==="
mkdir -p /scratch/results/cachedit_8gpu_ulysses
START_TS=$(date +%s%N)
torchrun --nproc_per_node=8 -m cache_dit.generate wan2.2_t2v \
  --cache \
  --ulysses 8 \
  --num_inference_steps 40 \
  --height 720 --width 1280 \
  --num_frames 81 \
  --seed 42 \
  --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard." \
  2>&1 | tee /scratch/results/cachedit_8gpu_ulysses/t2v_output.log
END_TS=$(date +%s%N)
echo $(( (END_TS - START_TS) / 1000000 )) > /scratch/results/cachedit_8gpu_ulysses/t2v_wall_time_ms.txt

# 4. 8 GPU with Ulysses + cache-dit + torch compile
echo "=== [4/6] cache-dit 8 GPU + torch compile ==="
mkdir -p /scratch/results/cachedit_8gpu_compile
START_TS=$(date +%s%N)
torchrun --nproc_per_node=8 -m cache_dit.generate wan2.2_t2v \
  --cache \
  --ulysses 8 \
  --compile \
  --num_inference_steps 40 \
  --height 720 --width 1280 \
  --num_frames 81 \
  --seed 42 \
  --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard." \
  2>&1 | tee /scratch/results/cachedit_8gpu_compile/t2v_output.log
END_TS=$(date +%s%N)
echo $(( (END_TS - START_TS) / 1000000 )) > /scratch/results/cachedit_8gpu_compile/t2v_wall_time_ms.txt

# 5. 8 GPU with 5-step sampling (ultra-fast)
echo "=== [5/6] cache-dit 8 GPU 5 steps ==="
mkdir -p /scratch/results/cachedit_8gpu_5steps
START_TS=$(date +%s%N)
torchrun --nproc_per_node=8 -m cache_dit.generate wan2.2_t2v \
  --cache \
  --ulysses 8 \
  --num_inference_steps 5 \
  --height 720 --width 1280 \
  --num_frames 81 \
  --seed 42 \
  --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard." \
  2>&1 | tee /scratch/results/cachedit_8gpu_5steps/t2v_output.log
END_TS=$(date +%s%N)
echo $(( (END_TS - START_TS) / 1000000 )) > /scratch/results/cachedit_8gpu_5steps/t2v_wall_time_ms.txt

# 6. Baseline without cache-dit (for fair comparison on same VM)
echo "=== [6/6] Baseline no cache (8 GPU Ulysses) ==="
mkdir -p /scratch/results/cachedit_8gpu_nocache
START_TS=$(date +%s%N)
torchrun --nproc_per_node=8 -m cache_dit.generate wan2.2_t2v \
  --ulysses 8 \
  --num_inference_steps 40 \
  --height 720 --width 1280 \
  --num_frames 81 \
  --seed 42 \
  --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard." \
  2>&1 | tee /scratch/results/cachedit_8gpu_nocache/t2v_output.log
END_TS=$(date +%s%N)
echo $(( (END_TS - START_TS) / 1000000 )) > /scratch/results/cachedit_8gpu_nocache/t2v_wall_time_ms.txt

echo ""
echo "=== ALL CACHE-DIT BENCHMARKS COMPLETE at $(date) ==="
