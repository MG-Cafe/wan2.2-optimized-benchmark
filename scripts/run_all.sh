#!/bin/bash
#
# Wan2.2 Optimized Benchmark - Complete End-to-End Runner
#
# This script provisions G4 VMs, runs all single-host and multi-host benchmarks
# with optimizations (SP, SageAttention, P2P, Cache-DiT), collects results.
#
# Usage:
#   export PROJECT_ID="your-project-id"
#   export ZONE="europe-west4-b"   # or any zone with G4 capacity
#   bash scripts/run_all.sh
#
# Prerequisites:
#   - gcloud CLI authenticated
#   - GPU quota for g4-standard-384 in chosen zone (need 2 VMs = 16 GPUs)
#   - IAP API enabled
#
set -euo pipefail

# ================================================================
# Configuration
# ================================================================
PROJECT_ID="${PROJECT_ID:?ERROR: Set PROJECT_ID environment variable}"
ZONE="${ZONE:-europe-west4-b}"
MACHINE_TYPE="g4-standard-384"
IMAGE_PROJECT="ubuntu-os-accelerator-images"
IMAGE_FAMILY="ubuntu-accelerator-2404-amd64-with-nvidia-570"
BOOT_DISK_SIZE="500GB"
DOCKER_IMAGE="lmsysorg/sglang:latest"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
VM1="bench-g4-opt1-${RUN_ID}"
VM2="bench-g4-opt2-${RUN_ID}"

SSH="gcloud compute ssh --project=${PROJECT_ID} --zone=${ZONE} --tunnel-through-iap"
SCP="gcloud compute scp --project=${PROJECT_ID} --zone=${ZONE} --tunnel-through-iap"

LOCAL_RESULTS="$(pwd)/results/run_${RUN_ID}"
mkdir -p "${LOCAL_RESULTS}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

echo "==============================================================="
echo "  Wan2.2 Optimized Benchmark Suite"
echo "==============================================================="
echo "Project:  ${PROJECT_ID}"
echo "Zone:     ${ZONE}"
echo "Machine:  ${MACHINE_TYPE}"
echo "Run ID:   ${RUN_ID}"
echo "VMs:      ${VM1}, ${VM2}"
echo "Results:  ${LOCAL_RESULTS}"
echo "==============================================================="
echo ""

# ================================================================
# Step 1: Create VMs
# ================================================================
log "[Step 1/9] Creating 2 G4 VMs..."

for VM in ${VM1} ${VM2}; do
  log "  Creating ${VM}..."
  gcloud compute instances create ${VM} \
    --machine-type=${MACHINE_TYPE} \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --image-project=${IMAGE_PROJECT} \
    --image-family=${IMAGE_FAMILY} \
    --maintenance-policy=TERMINATE \
    --boot-disk-size=${BOOT_DISK_SIZE} \
    --quiet 2>&1 | tee -a "${LOCAL_RESULTS}/vm_creation.log"
done

log "  Waiting 90s for VMs to boot..."
sleep 90

# ================================================================
# Step 2: Setup both VMs
# ================================================================
log "[Step 2/9] Setting up Docker + NVIDIA toolkit on both VMs..."

for VM in ${VM1} ${VM2}; do
  log "  Setting up ${VM}..."
  ${SCP} scripts/01_vm_setup.sh ${VM}:/tmp/vm_setup.sh 2>/dev/null
  ${SSH} ${VM} --command="sudo bash /tmp/vm_setup.sh" 2>&1 | tee -a "${LOCAL_RESULTS}/setup_${VM}.log"
done

# ================================================================
# Step 3: Pull Docker image
# ================================================================
log "[Step 3/9] Pulling SGLang Docker image on both VMs (~5 min)..."

for VM in ${VM1} ${VM2}; do
  ${SSH} ${VM} --command="sudo docker pull ${DOCKER_IMAGE}" 2>&1 | tee -a "${LOCAL_RESULTS}/docker_pull.log" &
done
wait
log "  Docker image pulled on both VMs."

# ================================================================
# Step 4: Copy benchmark scripts to VMs
# ================================================================
log "[Step 4/9] Copying benchmark scripts to VMs..."

for VM in ${VM1} ${VM2}; do
  ${SCP} scripts/02_run_sglang_benchmark.sh ${VM}:/scratch/run_benchmark.sh 2>/dev/null
  ${SCP} scripts/03_run_multihost_benchmark.sh ${VM}:/scratch/run_multihost.sh 2>/dev/null
  ${SSH} ${VM} --command="chmod +x /scratch/run_benchmark.sh /scratch/run_multihost.sh" 2>/dev/null
done

# ================================================================
# Step 5: Run single-host benchmarks on VM1
# ================================================================
log "[Step 5/9] Running single-host benchmarks on ${VM1}..."

# Define all single-host benchmark configurations
# Format: BENCHMARK_ID|SGLANG_FLAGS|ENV_VARS
SINGLE_HOST_BENCHMARKS=(
  # Baselines
  "baseline_tp4|--num-gpus 4 --tp-size 4|"
  "baseline_tp8|--num-gpus 8 --tp-size 8|"
  # Sequence Parallelism
  "sp_4gpu_u2r2|--num-gpus 4 --sp-degree 4 --ulysses-degree 2 --ring-degree 2|"
  "sp_8gpu_u4r2|--num-gpus 8 --sp-degree 8 --ulysses-degree 4 --ring-degree 2|"
  "sp_8gpu_u2r4|--num-gpus 8 --sp-degree 8 --ulysses-degree 2 --ring-degree 4|"
  # SageAttention
  "sage_attn_tp4|--num-gpus 4 --tp-size 4 --attention-backend sage_attn|"
  "sage_attn_tp8|--num-gpus 8 --tp-size 8 --attention-backend sage_attn|"
  # Combined: SageAttn + SP
  "sage_sp_4gpu|--num-gpus 4 --sp-degree 4 --ulysses-degree 2 --ring-degree 2 --attention-backend sage_attn|"
  "sage_sp_8gpu|--num-gpus 8 --sp-degree 8 --ulysses-degree 4 --ring-degree 2 --attention-backend sage_attn|"
  # P2P Communication
  "p2p_tp4|--num-gpus 4 --tp-size 4|NCCL_P2P_LEVEL=5"
  "p2p_tp8|--num-gpus 8 --tp-size 8|NCCL_P2P_LEVEL=5"
  # All optimizations combined
  "p2p_sp_sage_8gpu|--num-gpus 8 --sp-degree 8 --ulysses-degree 4 --ring-degree 2 --attention-backend sage_attn|NCCL_P2P_LEVEL=5"
  # Cache-DiT (1-GPU with VAE offload)
  "cache_dit_1gpu|--num-gpus 1 --vae-cpu-offload true|SGLANG_CACHE_DIT_ENABLED=true SGLANG_CACHE_DIT_TAYLORSEER=true SGLANG_CACHE_DIT_TS_ORDER=1"
)

# Build the master benchmark script that runs all configs sequentially
${SSH} ${VM1} --command="cat > /scratch/run_all_benchmarks.sh << 'MASTERSCRIPT'
#!/bin/bash
set -x

run_bench() {
  local BENCH_ID=\$1
  local FLAGS=\$2
  local EXTRA_ENV=\$3

  for MODEL_TYPE in t2v i2v; do
    echo \"\"
    echo \"###############################################\"
    echo \"# \${BENCH_ID} - \${MODEL_TYPE}\"
    echo \"###############################################\"

    # Set environment variables
    export BENCHMARK_ID=\"\${BENCH_ID}\"
    export MODEL_TYPE=\"\${MODEL_TYPE}\"
    export SGLANG_FLAGS=\"\${FLAGS}\"
    export NUM_FRAMES=81
    export SEED=42
    export SGLANG_DIFFUSION_STAGE_LOGGING=true

    # Set any extra env vars (P2P, Cache-DiT, etc.)
    if [ -n \"\${EXTRA_ENV}\" ]; then
      for kv in \${EXTRA_ENV}; do
        export \${kv}
      done
    fi

    bash /scratch/run_benchmark.sh || echo \"BENCHMARK \${BENCH_ID} \${MODEL_TYPE} FAILED (continuing)\"

    # Unset extra env vars to avoid contamination
    if [ -n \"\${EXTRA_ENV}\" ]; then
      for kv in \${EXTRA_ENV}; do
        key=\$(echo \${kv} | cut -d= -f1)
        unset \${key} 2>/dev/null || true
      done
    fi
  done
}

MASTERSCRIPT
chmod +x /scratch/run_all_benchmarks.sh
"

# Append each benchmark to the master script
for bench in "${SINGLE_HOST_BENCHMARKS[@]}"; do
  IFS='|' read -r BENCH_ID FLAGS ENV_VARS <<< "$bench"
  ${SSH} ${VM1} --command="echo 'run_bench \"${BENCH_ID}\" \"${FLAGS}\" \"${ENV_VARS}\"' >> /scratch/run_all_benchmarks.sh"
done

${SSH} ${VM1} --command="echo 'echo === ALL SINGLE-HOST BENCHMARKS COMPLETE ===' >> /scratch/run_all_benchmarks.sh"

# Launch inside Docker container
${SSH} ${VM1} --command="
sudo docker run -d --name benchmarks --gpus all \
  -v /scratch:/scratch -v /scratch/cache:/root/.cache --ipc=host \
  --shm-size=32g \
  ${DOCKER_IMAGE} \
  /bin/bash -c 'pip install sageattention 2>&1 | tail -3; bash /scratch/run_all_benchmarks.sh 2>&1 | tee /scratch/all_benchmarks.log'
echo 'Single-host benchmarks started in container on ${VM1}.'
" 2>&1 | tee -a "${LOCAL_RESULTS}/benchmark_start.log"

# ================================================================
# Step 6: Run single-host benchmarks on VM2 (parallel - different configs)
# ================================================================
log "[Step 6/9] VM2 reserved for multi-host benchmarks (will start after VM1 completes)."

# ================================================================
# Step 7: Wait for single-host benchmarks to complete
# ================================================================
log "[Step 7/9] Waiting for single-host benchmarks on ${VM1} (est. 4-6 hours)..."
log "  Monitor: ${SSH} ${VM1} --command='sudo docker logs --tail 10 benchmarks 2>&1'"

while true; do
  STATUS=$(${SSH} ${VM1} --command="sudo docker ps -a --format '{{.Status}}' --filter name=benchmarks" 2>/dev/null | grep -c "Exited" || true)
  if [ "$STATUS" -gt 0 ]; then
    log "  Single-host benchmarks DONE on ${VM1}!"
    break
  fi
  LAST_LINE=$(${SSH} ${VM1} --command="sudo docker logs --tail 1 benchmarks 2>&1" 2>/dev/null || echo "...")
  log "  Still running... Last: ${LAST_LINE}"
  sleep 300
done

# Collect single-host results
log "  Collecting single-host results..."
${SSH} ${VM1} --command="sudo docker logs benchmarks 2>&1" > "${LOCAL_RESULTS}/vm1_full.log" 2>&1
${SCP} --recurse ${VM1}:/scratch/results/ "${LOCAL_RESULTS}/vm1_results/" 2>/dev/null || true

# ================================================================
# Step 8: Run multi-host benchmarks (2 VMs)
# ================================================================
log "[Step 8/9] Running multi-host benchmarks across ${VM1} and ${VM2}..."

# Get internal IPs
VM1_IP=$(gcloud compute instances describe ${VM1} --project=${PROJECT_ID} --zone=${ZONE} --format='get(networkInterfaces[0].networkIP)')
VM2_IP=$(gcloud compute instances describe ${VM2} --project=${PROJECT_ID} --zone=${ZONE} --format='get(networkInterfaces[0].networkIP)')
log "  VM1 IP: ${VM1_IP}, VM2 IP: ${VM2_IP}"

# Run multi-host with ulysses_size=16 (T2V)
log "  Starting multi-host T2V with ulysses_size=16..."

# Start on VM2 (worker) first
${SSH} ${VM2} --command="
sudo docker run -d --name multihost-worker --gpus all \
  -v /scratch:/scratch -v /scratch/cache:/root/.cache --ipc=host \
  --shm-size=32g --network=host \
  -e MASTER_ADDR=${VM1_IP} -e NODE_RANK=1 -e ULYSSES_SIZE=16 -e TASK=t2v-A14B \
  -e NCCL_P2P_LEVEL=5 -e NCCL_SOCKET_IFNAME=eth0 -e NCCL_IB_DISABLE=1 \
  ${DOCKER_IMAGE} \
  /bin/bash -c 'pip install easydict 2>/dev/null; bash /scratch/run_multihost.sh 2>&1 | tee /scratch/multihost_worker.log'
" 2>&1 | tee -a "${LOCAL_RESULTS}/multihost_start.log"

# Start on VM1 (master)
${SSH} ${VM1} --command="
sudo docker run -d --name multihost-master --gpus all \
  -v /scratch:/scratch -v /scratch/cache:/root/.cache --ipc=host \
  --shm-size=32g --network=host \
  -e MASTER_ADDR=${VM1_IP} -e NODE_RANK=0 -e ULYSSES_SIZE=16 -e TASK=t2v-A14B \
  -e NCCL_P2P_LEVEL=5 -e NCCL_SOCKET_IFNAME=eth0 -e NCCL_IB_DISABLE=1 \
  ${DOCKER_IMAGE} \
  /bin/bash -c 'pip install easydict 2>/dev/null; bash /scratch/run_multihost.sh 2>&1 | tee /scratch/multihost_master.log'
" 2>&1 | tee -a "${LOCAL_RESULTS}/multihost_start.log"

# Wait for multi-host to complete
log "  Waiting for multi-host benchmarks..."
while true; do
  MASTER_STATUS=$(${SSH} ${VM1} --command="sudo docker ps -a --format '{{.Status}}' --filter name=multihost-master" 2>/dev/null | grep -c "Exited" || true)
  if [ "$MASTER_STATUS" -gt 0 ]; then
    log "  Multi-host T2V benchmark DONE!"
    break
  fi
  sleep 120
done

# Collect multi-host results
${SSH} ${VM1} --command="sudo docker logs multihost-master 2>&1" > "${LOCAL_RESULTS}/multihost_master.log" 2>&1
${SSH} ${VM2} --command="sudo docker logs multihost-worker 2>&1" > "${LOCAL_RESULTS}/multihost_worker.log" 2>&1

# ================================================================
# Step 9: Collect all results and cleanup
# ================================================================
log "[Step 9/9] Collecting final results..."

${SCP} --recurse ${VM1}:/scratch/results/ "${LOCAL_RESULTS}/vm1_results/" 2>/dev/null || true
${SCP} --recurse ${VM2}:/scratch/results/ "${LOCAL_RESULTS}/vm2_results/" 2>/dev/null || true

echo ""
echo "==============================================================="
echo "  ALL BENCHMARKS COMPLETE"
echo "==============================================================="
echo "Results: ${LOCAL_RESULTS}/"
echo ""
echo "To delete VMs and stop charges:"
echo "  gcloud compute instances delete ${VM1} ${VM2} \\"
echo "    --zone=${ZONE} --project=${PROJECT_ID} --quiet --delete-disks=all"
echo ""
echo "To generate plots:"
echo "  python3 scripts/parse_results.py ${LOCAL_RESULTS}"
echo "  python3 generate_plots.py"
echo "==============================================================="
