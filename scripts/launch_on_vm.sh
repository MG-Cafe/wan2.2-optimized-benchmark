#!/bin/bash
# Launch script - SCP this to VM then run it
set -x

echo "=== Starting benchmark pipeline at $(date) ==="

# Pull docker image
echo "Pulling SGLang Docker image..."
sudo docker pull lmsysorg/sglang:latest 2>&1 | tail -5
echo "Docker pull complete at $(date)"

# Create the benchmark script
cat > /scratch/run_all_benchmarks.sh << 'BENCHSCRIPT'
#!/bin/bash
set -x

run_bench() {
  local BENCH_ID=$1
  local FLAGS="$2"
  local EXTRA_ENV="$3"

  for MODEL_TYPE in t2v i2v; do
    echo ""
    echo "###############################################"
    echo "# ${BENCH_ID} - ${MODEL_TYPE}"
    echo "# Started: $(date)"
    echo "###############################################"

    export BENCHMARK_ID="${BENCH_ID}"
    export MODEL_TYPE="${MODEL_TYPE}"
    export SGLANG_FLAGS="${FLAGS}"
    export NUM_FRAMES=81
    export SEED=42
    export SGLANG_DIFFUSION_STAGE_LOGGING=true

    if [ -n "${EXTRA_ENV}" ]; then
      for kv in ${EXTRA_ENV}; do
        export ${kv}
      done
    fi

    bash /scratch/run_benchmark.sh || echo "BENCHMARK ${BENCH_ID} ${MODEL_TYPE} FAILED"

    if [ -n "${EXTRA_ENV}" ]; then
      for kv in ${EXTRA_ENV}; do
        key=$(echo ${kv} | cut -d= -f1)
        unset ${key} 2>/dev/null || true
      done
    fi
  done
}

# Baselines
run_bench "baseline_tp4" "--num-gpus 4 --tp-size 4" ""
run_bench "baseline_tp8" "--num-gpus 8 --tp-size 8" ""

# Sequence Parallelism
run_bench "sp_4gpu_u2r2" "--num-gpus 4 --sp-degree 4 --ulysses-degree 2 --ring-degree 2" ""
run_bench "sp_8gpu_u4r2" "--num-gpus 8 --sp-degree 8 --ulysses-degree 4 --ring-degree 2" ""
run_bench "sp_8gpu_u2r4" "--num-gpus 8 --sp-degree 8 --ulysses-degree 2 --ring-degree 4" ""

# SageAttention
run_bench "sage_attn_tp4" "--num-gpus 4 --tp-size 4 --attention-backend sage_attn" ""
run_bench "sage_attn_tp8" "--num-gpus 8 --tp-size 8 --attention-backend sage_attn" ""

# Combined: SageAttn + SP
run_bench "sage_sp_4gpu" "--num-gpus 4 --sp-degree 4 --ulysses-degree 2 --ring-degree 2 --attention-backend sage_attn" ""
run_bench "sage_sp_8gpu" "--num-gpus 8 --sp-degree 8 --ulysses-degree 4 --ring-degree 2 --attention-backend sage_attn" ""

# P2P Communication
run_bench "p2p_tp4" "--num-gpus 4 --tp-size 4" "NCCL_P2P_LEVEL=5"
run_bench "p2p_tp8" "--num-gpus 8 --tp-size 8" "NCCL_P2P_LEVEL=5"

# All optimizations combined
run_bench "p2p_sp_sage_8gpu" "--num-gpus 8 --sp-degree 8 --ulysses-degree 4 --ring-degree 2 --attention-backend sage_attn" "NCCL_P2P_LEVEL=5"

# Cache-DiT (1-GPU with VAE offload)
run_bench "cache_dit_1gpu" "--num-gpus 1 --vae-cpu-offload true" "SGLANG_CACHE_DIT_ENABLED=true SGLANG_CACHE_DIT_TAYLORSEER=true SGLANG_CACHE_DIT_TS_ORDER=1"

echo "=== ALL BENCHMARKS COMPLETE at $(date) ==="
BENCHSCRIPT
chmod +x /scratch/run_all_benchmarks.sh

# Launch benchmarks in Docker container
echo "Starting Docker container with benchmarks..."
sudo docker run -d --name benchmarks --gpus all \
  -v /scratch:/scratch -v /scratch/cache:/root/.cache --ipc=host \
  --shm-size=32g \
  lmsysorg/sglang:latest \
  /bin/bash -c "pip install sageattention 2>&1 | tail -3; bash /scratch/run_all_benchmarks.sh 2>&1 | tee /scratch/all_benchmarks.log"

echo "Benchmarks container started at $(date)"
sudo docker ps
echo "=== Pipeline launch complete ==="
