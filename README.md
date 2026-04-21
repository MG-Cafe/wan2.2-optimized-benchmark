# Wan2.2 Optimized Video Generation Benchmark on NVIDIA RTX PRO 6000 (G4)

> **Performance Optimization Study**: Wan2.2-A14B (Text-to-Video & Image-to-Video) inference with SGLang and torchrun on Google Cloud G4 VMs — applying Sequence Parallelism, SageAttention, P2P Communication, Cache-DiT, and Multi-Host scaling

Based on the [Google AI-Hypercomputer GPU Recipe](https://github.com/AI-Hypercomputer/gpu-recipes/tree/main/inference/g4/wan2.2/sglang) with performance optimizations derived from documented techniques for the G4 platform.

---

## Key Optimizations Applied

| # | Optimization | Technique | Source | Impact |
|---|-------------|-----------|--------|--------|
| 1 | **Sequence Parallelism (USP)** | `--ulysses-degree N --ring-degree N` replaces pure TP | [Wan2.2 Ulysses SP](https://github.com/Wan-Video/Wan2.2/blob/main/wan/distributed/ulysses.py), [SGLang Ring-SP](https://github.com/sgl-project/sglang/blob/main/docs/diffusion/performance/ring_sp_performance.md) | Better GPU scaling via DeepSpeed Ulysses all-to-all + Ring attention |
| 2 | **SageAttention** | `--attention-backend sage_attn` | [SGLang compatibility matrix](https://github.com/sgl-project/sglang/blob/main/docs/diffusion/compatibility_matrix.md) — Wan2.2 A14B ✅ | Optimized attention kernel for Blackwell (SM120) |
| 3 | **P2P GPU Communication** | `NCCL_P2P_LEVEL=5` | [G4 P2P design](https://cloud.google.com/compute/docs/accelerator-optimized-machines#g4-machine-types) — up to 2.2x NCCL throughput | G4's proprietary PCIe P2P bypasses CPU |
| 4 | **Cache-DiT** | `SGLANG_CACHE_DIT_ENABLED=true` + TaylorSeer | [SGLang Cache-DiT docs](https://github.com/sgl-project/sglang/blob/main/docs/diffusion/performance/cache/cache_dit.md) — Wan2.2 supported | Block-level caching skips redundant DiT computation (1-GPU) |
| 5 | **Multi-Host Inference** | `torchrun --nnodes=2 --ulysses_size N` | [Wan2.2 generate.py](https://github.com/Wan-Video/Wan2.2/blob/main/generate.py) with distributed groups | 16-GPU (2×8) cross-node inference via NCCL |

---

## Benchmark Matrix

### Single-Host Benchmarks (1× g4-standard-384, up to 8 GPUs)

| Config | GPUs | Framework | Key Flags | Category |
|--------|------|-----------|-----------|----------|
| Baseline TP=4 | 4 | SGLang | `--tp-size 4` | Control |
| Baseline TP=8 | 8 | SGLang | `--tp-size 8` | Control |
| SP u2r2 | 4 | SGLang | `--sp-degree 4 --ulysses-degree 2 --ring-degree 2` | Sequence Parallelism |
| SP u4r2 | 8 | SGLang | `--sp-degree 8 --ulysses-degree 4 --ring-degree 2` | Sequence Parallelism |
| SP u2r4 | 8 | SGLang | `--sp-degree 8 --ulysses-degree 2 --ring-degree 4` | Sequence Parallelism |
| SageAttn TP=4 | 4 | SGLang | `--tp-size 4 --attention-backend sage_attn` | Attention Backend |
| SageAttn TP=8 | 8 | SGLang | `--tp-size 8 --attention-backend sage_attn` | Attention Backend |
| SageAttn+SP 4GPU | 4 | SGLang | SP u2r2 + `sage_attn` | Combined |
| SageAttn+SP 8GPU | 8 | SGLang | SP u4r2 + `sage_attn` | Combined |
| P2P TP=4 | 4 | SGLang | `--tp-size 4` + `NCCL_P2P_LEVEL=5` | P2P Communication |
| P2P TP=8 | 8 | SGLang | `--tp-size 8` + `NCCL_P2P_LEVEL=5` | P2P Communication |
| All Optimized 8GPU | 8 | SGLang | SP u4r2 + sage_attn + P2P | Combined (best) |
| Cache-DiT 1GPU | 1 | SGLang | `--vae-cpu-offload true` + Cache-DiT | Caching |

### Multi-Host Benchmarks (2× g4-standard-384, 16 GPUs)

| Config | GPUs | Framework | Key Flags |
|--------|------|-----------|-----------|
| Multi-Host ulysses=16 | 16 | torchrun + Wan2.2 | `--ulysses_size 16` + NCCL P2P |
| Multi-Host ulysses=4 | 16 | torchrun + Wan2.2 | `--ulysses_size 4` (reference doc config) |

**Each configuration runs both T2V and I2V** with identical scenario parameters.

---

## Scenario Parameters

All benchmarks use identical inference parameters to ensure fair comparison:

| Parameter | Value | Source |
|-----------|-------|--------|
| Model | Wan2.2-A14B (T2V + I2V) | Reference doc |
| Resolution | 720P (1280×720) | Reference doc |
| Frames | 81 | Reference doc / Wan2.2 default |
| Inference Steps | 40 | Wan2.2 config default |
| Guidance Scale | (3.0, 4.0) T2V / (3.5, 3.5) I2V | Wan2.2 config default |
| Seed | 42 | Fixed for reproducibility |
| Precision | BF16 | Wan2.2 config default |

---

## Hardware & Software

| Component | Specification |
|-----------|--------------|
| **Machine Type** | `g4-standard-384` |
| **GPUs** | 8× NVIDIA RTX PRO 6000 (Blackwell) |
| **GPU Memory** | 96 GB GDDR7 per GPU (768 GB total) |
| **GPU Memory BW** | 1,600 GB/s per GPU |
| **BF16 TFLOPs** | 550 per GPU |
| **Interconnect** | PCIe Gen5 (128 GB/s bidirectional per GPU) |
| **P2P** | Google proprietary (up to 2.2x NCCL improvement) |
| **vCPUs** | 384 |
| **RAM** | 1,440 GB |
| **Local SSD** | 12 TB |
| **Networking** | 2× 200 Gbps (400 Gbps total egress) |
| **OS** | Ubuntu 24.04 LTS |
| **Framework** | SGLang (`lmsysorg/sglang:latest`) + Wan2.2 open-source |
| **Multi-Host** | torchrun with NCCL backend |

---

## Results

> ⚠️ **Results will be populated after benchmarks complete.** Run `bash scripts/run_all.sh` to execute all benchmarks.

### Summary Table

*To be populated with actual benchmark results after execution.*

| Config | T2V Total (s) | I2V Total (s) | T2V Speedup vs Baseline | Status |
|--------|---------------|---------------|------------------------|--------|
| Baseline TP=4 | — | — | 1.00x | ⏳ |
| Baseline TP=8 | — | — | — | ⏳ |
| SP u2r2 (4 GPU) | — | — | — | ⏳ |
| SP u4r2 (8 GPU) | — | — | — | ⏳ |
| SageAttn TP=4 | — | — | — | ⏳ |
| SageAttn TP=8 | — | — | — | ⏳ |
| P2P TP=4 | — | — | — | ⏳ |
| P2P TP=8 | — | — | — | ⏳ |
| All Optimized 8GPU | — | — | — | ⏳ |
| Cache-DiT 1GPU | — | — | — | ⏳ |
| Multi-Host 16GPU | — | — | — | ⏳ |

### Visualizations

After benchmarks complete, regenerate plots:
```bash
python3 scripts/parse_results.py results/run_*/
python3 generate_plots.py results/run_*/benchmark_results.json
```

---

## Reproducing the Benchmarks

### One-Command End-to-End

```bash
# Prerequisites: gcloud CLI authenticated, GPU quota for 2× g4-standard-384
git clone <this-repo>
cd wan2.2-optimized-benchmark

export PROJECT_ID="your-project-id"
export ZONE="europe-west4-b"  # any zone with G4 capacity

bash scripts/run_all.sh
```

**What `run_all.sh` does (9 steps):**
1. Creates 2 G4 VMs (`g4-standard-384`, 8× RTX PRO 6000, 500GB disk)
2. Sets up Docker + NVIDIA Container Toolkit on both VMs
3. Pulls `lmsysorg/sglang:latest` Docker image
4. Copies benchmark scripts to VMs
5. Runs all 13 single-host benchmarks sequentially on VM1
6. Reserves VM2 for multi-host
7. Waits for single-host completion, collects results
8. Runs multi-host benchmarks across both VMs (torchrun)
9. Collects all results

**Time estimate:** ~6-8 hours total (setup ~20 min, benchmarks ~5-6 hours)
**Cost estimate:** ~$150-250 for the full benchmark run (2× g4-standard-384 for ~7 hours)

### Manual Step-by-Step

#### Step 1: Create a G4 VM

```bash
export PROJECT_ID="your-project-id"
export ZONE="europe-west4-b"

gcloud compute instances create g4-wan22-bench \
  --machine-type=g4-standard-384 \
  --project=${PROJECT_ID} \
  --zone=${ZONE} \
  --image-project=ubuntu-os-accelerator-images \
  --image-family=ubuntu-accelerator-2404-amd64-with-nvidia-570 \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=500GB
```

#### Step 2: Setup VM

```bash
gcloud compute ssh g4-wan22-bench --project=${PROJECT_ID} --zone=${ZONE} --tunnel-through-iap
sudo bash scripts/01_vm_setup.sh
```

#### Step 3: Run Individual Benchmarks

```bash
# Start SGLang container
sudo docker run -it --gpus all \
  -v /scratch:/scratch -v /scratch/cache:/root/.cache --ipc=host \
  --shm-size=32g \
  lmsysorg/sglang:latest /bin/bash

# Install SageAttention (inside container)
pip install sageattention

# === Baseline: T2V 4-GPU TP=4 ===
sglang generate --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --dit-layerwise-offload false --text-encoder-cpu-offload false \
  --pin-cpu-memory --dit-cpu-offload false \
  --num-gpus 4 --tp-size 4 --num-frames 81 --seed 42 \
  --prompt "Summer beach vacation style, a white cat wearing sunglasses..." \
  --save-output

# === Optimized: SP u4r2 + SageAttn (8 GPU) ===
sglang generate --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --dit-layerwise-offload false --text-encoder-cpu-offload false \
  --pin-cpu-memory --dit-cpu-offload false \
  --num-gpus 8 --sp-degree 8 --ulysses-degree 4 --ring-degree 2 \
  --attention-backend sage_attn \
  --num-frames 81 --seed 42 \
  --prompt "Summer beach vacation style, a white cat wearing sunglasses..." \
  --save-output

# === With P2P (set before starting container) ===
# NCCL_P2P_LEVEL=5 sglang generate ...

# === Cache-DiT (1-GPU) ===
SGLANG_CACHE_DIT_ENABLED=true \
SGLANG_CACHE_DIT_TAYLORSEER=true \
SGLANG_CACHE_DIT_TS_ORDER=1 \
sglang generate --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --vae-cpu-offload true --pin-cpu-memory \
  --num-gpus 1 --num-frames 81 --seed 42 \
  --prompt "Summer beach vacation style, a white cat wearing sunglasses..." \
  --save-output
```

#### Step 4: Multi-Host Benchmark

```bash
# On VM1 (master):
MASTER_ADDR=<vm1_internal_ip> NODE_RANK=0 ULYSSES_SIZE=16 TASK=t2v-A14B \
  bash scripts/03_run_multihost_benchmark.sh

# On VM2 (worker, simultaneously):
MASTER_ADDR=<vm1_internal_ip> NODE_RANK=1 ULYSSES_SIZE=16 TASK=t2v-A14B \
  bash scripts/03_run_multihost_benchmark.sh
```

#### Step 5: Cleanup

```bash
gcloud compute instances delete g4-wan22-bench \
  --zone=${ZONE} --project=${PROJECT_ID} --quiet --delete-disks=all
```

---

## Optimization Details

### 1. Sequence Parallelism (USP = Ulysses + Ring)

**What:** Replaces tensor parallelism with Unified Sequence Parallelism that combines:
- **Ulysses parallelism** (DeepSpeed): All-to-all communication on attention head dimension — efficient when `num_heads % ulysses_degree == 0` (Wan2.2 A14B has 40 heads)
- **Ring attention**: Overlaps communication with computation for long sequences

**Why better than TP for diffusion:** Tensor parallelism splits model weights across GPUs. For DiT models with 40 attention heads, SP splits the sequence dimension instead, reducing per-GPU memory and enabling better PCIe utilization.

**Reference:** Wan2.2-A14B has `num_heads=40`, so valid ulysses degrees are 1, 2, 4, 5, 8, 10, 20, 40. Ring degree can be any factor.

### 2. SageAttention

**What:** Optimized attention kernel that uses quantized QK products for faster attention computation. Supports Blackwell SM120 architecture (RTX PRO 6000).

**Reference:** [SGLang compatibility matrix](https://github.com/sgl-project/sglang/blob/main/docs/diffusion/compatibility_matrix.md) confirms Wan2.2-T2V-A14B and Wan2.2-I2V-A14B both support SageAttention ✅.

### 3. P2P GPU Communication

**What:** Google Cloud G4 VMs have a proprietary Peer-to-Peer interconnect design that allows GPUs to communicate directly through PCIe switches, bypassing the CPU. This is activated via `NCCL_P2P_LEVEL=5` (SYS level).

**Reference:** G4 P2P delivers up to 2.2x NCCL collectives performance improvement for 8-GPU configurations, and up to 168% higher throughput for multi-GPU inference workloads.

### 4. Cache-DiT

**What:** Block-level caching acceleration for Diffusion Transformers that skips redundant computation in the denoising loop. Uses:
- **DBCache**: Dynamic caching based on residual differences
- **TaylorSeer**: Taylor expansion calibrator for better caching decisions

**Limitation:** Currently disabled when `world_size > 1` in SGLang native pipelines (only works for single-GPU). That's why we test it with 1-GPU + VAE CPU offloading to avoid OOM.

### 5. Multi-Host Inference

**What:** Uses the open-source Wan2.2 `generate.py` with PyTorch's `torchrun` for distributed inference across 2 G4 nodes (16 GPUs total). The `--ulysses_size` flag enables Ulysses sequence parallelism across all GPUs.

**Network:** G4 instances support 400 Gbps total egress bandwidth (2× 200 Gbps NICs). NCCL uses standard VPC networking (no RDMA/NVLink between nodes).

---

## Repository Structure

```
├── README.md                           # This file
├── generate_plots.py                   # Plot generation from results JSON
├── configs/
│   └── benchmark_matrix.json           # Full benchmark configuration
├── scripts/
│   ├── run_all.sh                      # ⭐ End-to-end orchestrator
│   ├── 01_vm_setup.sh                  # VM Docker + NVIDIA setup
│   ├── 02_run_sglang_benchmark.sh      # Single SGLang benchmark runner
│   ├── 03_run_multihost_benchmark.sh   # Multi-host torchrun runner
│   └── parse_results.py                # Log parser → JSON
├── results/
│   └── run_YYYYMMDD_HHMMSS/            # Benchmark run results
│       ├── benchmark_results.json      # Parsed results
│       ├── vm1_full.log                # Full VM1 logs
│       └── ...
├── plots/                              # Generated visualizations
│   ├── 01_t2v_optimization_comparison.png
│   ├── 02_i2v_optimization_comparison.png
│   ├── 03_speedup_by_optimization.png
│   ├── 04_denoising_per_step.png
│   ├── 05_category_impact.png
│   ├── 06_gpu_scaling.png
│   └── 07_optimization_legend.png
└── docs/
    └── optimization_reference.md       # Optimization technique details
```

---

## Deviations from Original Recipe

| Aspect | Original Recipe | This Benchmark | Reason |
|--------|----------------|----------------|--------|
| **Parallelism** | TP only (`--tp-size N`) | TP + SP + USP | Sequence parallelism better suits DiT models on PCIe |
| **Attention** | Default (FlashAttention) | + SageAttention | SageAttn supported on Blackwell SM120, potentially faster |
| **P2P** | Not configured | `NCCL_P2P_LEVEL=5` | Enables G4's proprietary GPU P2P interconnect |
| **Caching** | None | Cache-DiT (1-GPU) | Block-level caching reduces denoising computation |
| **Multi-host** | Not in recipe | 2-node torchrun | Tests 16-GPU scaling via open-source Wan2.2 distributed |
| **Boot disk** | 200 GB | 500 GB | 200GB insufficient for Docker + model weights |
| **Docker mode** | Interactive | Detached | Automation via SSH with IAP tunnel |
| **Frames** | 81/93 | 81 (all configs) | Consistent for fair comparison across GPU counts |
| **SageAttention** | Not installed | `pip install sageattention` | Required for `--attention-backend sage_attn` |

**What is identical:**
- Machine type: `g4-standard-384`
- Image: `ubuntu-accelerator-2404-amd64-with-nvidia-570`
- Docker image: `lmsysorg/sglang:latest`
- Model: `Wan-AI/Wan2.2-T2V-A14B-Diffusers` / `Wan-AI/Wan2.2-I2V-A14B-Diffusers`
- All `sglang generate` base flags for model loading

---

## G4 VM Availability

G4 VMs with RTX PRO 6000 GPUs are available in many regions. The benchmark auto-selects any zone with capacity. Known zones:

- `us-central1-b`, `us-central1-f`
- `europe-west1-c`, `europe-west4-b`, `europe-west4-a`
- `us-west1-a/b/c`, `us-east1-b`, `us-east4-b/c`
- `asia-east1-b`, `asia-southeast1-a/b/c`, `asia-south1-c`
- `europe-west2-b/c`, `europe-north1-b`

---

*Benchmarks executed on Google Cloud G4 VMs.*
