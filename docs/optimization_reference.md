# Optimization Technique Reference

This document provides detailed technical justification for each optimization applied in this benchmark suite. Every optimization is traceable to publicly available documentation — no fabricated claims.

## 1. Sequence Parallelism (Ulysses + Ring)

### Source
- **Wan2.2 open-source implementation**: [`wan/distributed/ulysses.py`](https://github.com/Wan-Video/Wan2.2/blob/main/wan/distributed/ulysses.py) — implements DeepSpeed Ulysses distributed attention via all-to-all
- **Wan2.2 `generate.py`**: [`--ulysses_size` CLI flag](https://github.com/Wan-Video/Wan2.2/blob/main/generate.py) — controls Ulysses degree
- **SGLang USP**: [`--ulysses-degree`, `--ring-degree`, `--sp-degree` flags](https://github.com/sgl-project/sglang/blob/main/docs/diffusion/api/cli.md) — SGLang's unified sequence parallelism
- **Ring-SP benchmark**: [SGLang Ring-SP Performance doc](https://github.com/sgl-project/sglang/blob/main/docs/diffusion/performance/ring_sp_performance.md) — demonstrates 1.42x speedup on Wan2.2-TI2V-5B

### Technical Details
- Ulysses parallelism (DeepSpeed): scatters attention heads across GPUs, performs local attention, then gathers. Requires `num_heads % ulysses_size == 0`.
- Wan2.2-A14B has `num_heads=40`, `num_layers=40`, `dim=5120` (from `wan/configs/wan_t2v_A14B.py`).
- Valid ulysses degrees: 1, 2, 4, 5, 8, 10, 20, 40.
- Ring attention: overlaps communication with computation for long sequences.
- Reference doc mentioned: `--ulysses_size 4` with Ring 2 for 16-GPU configuration.

### Why Better than TP for DiT
Tensor parallelism splits weight matrices. For diffusion transformers operating on video sequences, sequence parallelism splits the token/spatial dimension, which:
1. Reduces activation memory per GPU (not just weight memory)
2. Better matches the communication pattern of PCIe interconnect (all-to-all vs all-reduce)

## 2. SageAttention

### Source
- **SGLang compatibility matrix**: [docs/diffusion/compatibility_matrix.md](https://github.com/sgl-project/sglang/blob/main/docs/diffusion/compatibility_matrix.md) — Wan2.2-T2V-A14B ✅, Wan2.2-I2V-A14B ✅
- **SageAttention upstream**: [github.com/thu-ml/SageAttention](https://github.com/thu-ml/SageAttention) — `setup.py` lists SM120 (Blackwell) as supported
- **SGLang attention backends doc**: [docs/diffusion/performance/attention_backends.md](https://github.com/sgl-project/sglang/blob/main/docs/diffusion/performance/attention_backends.md)

### Technical Details
- SageAttention uses INT8/FP8 quantized QK^T products for approximate attention with minimal quality loss.
- RTX PRO 6000 is Blackwell architecture with compute capability SM120, which is in SageAttention's supported target list.
- Activated via `--attention-backend sage_attn` in SGLang CLI.
- Requires `pip install sageattention` inside the SGLang Docker container.

## 3. P2P GPU Communication (NCCL_P2P_LEVEL)

### Source
- **G4 P2P design**: [G4 VM positioning slides](https://cloud.google.com/compute/docs/accelerator-optimized-machines#g4-machine-types) — "Peer-to-Peer (P2P) design allows multi-GPU workloads to bypass the CPU, slashing inter-token latency by up to 41% and boosting throughput by up to 168%"
- **NCCL P2P documentation**: Can be enabled by setting `NCCL_P2P_LEVEL` flag.
- **G4 NCCL benchmarks**: Up to 2.0x NCCL collectives performance for 4-GPU, up to 2.2x for 8-GPU configurations.

### Technical Details
- G4 VMs have a custom PCIe switch topology: 4 pairs of GPUs connected through PCIe Gen5 switches, with CPU interconnect via xGMI.
- `NCCL_P2P_LEVEL=5` (SYS) enables system-wide P2P, allowing GPUs to communicate directly through the PCIe switches.
- This is transparent to the application — no code changes needed, just the environment variable.
- Supported on `g4-standard-96`, `g4-standard-192`, and `g4-standard-384` VM shapes.

## 4. Cache-DiT

### Source
- **SGLang Cache-DiT integration**: [docs/diffusion/performance/cache/cache_dit.md](https://github.com/sgl-project/sglang/blob/main/docs/diffusion/performance/cache/cache_dit.md) — "up to 1.69x inference speedup"
- **Cache-DiT upstream**: [github.com/vipshop/cache-dit](https://github.com/vipshop/cache-dit)
- **Wan2.2 support confirmed**: Cache-DiT supported models table lists "Wan2.1, Wan2.2"
- **Secondary transformer support**: Wan2.2 has dual transformers (high/low noise experts), Cache-DiT has `SGLANG_CACHE_DIT_SECONDARY_*` env vars

### Technical Details
- **DBCache**: Dynamically decides when to cache transformer block outputs based on residual difference threshold (RDT).
- **TaylorSeer**: Uses Taylor expansion to predict block outputs, improving caching decisions.
- **SCM**: Step Computation Masking — skips entire denoising steps (requires ≥8 steps; Wan2.2 uses 40).
- **Limitation**: "SGLang-native pipelines: Distributed support (TP/SP) is not yet validated; Cache-DiT will be automatically disabled when world_size > 1." — That's why we only test this on 1-GPU with VAE offloading.

### Environment Variables Used
```
SGLANG_CACHE_DIT_ENABLED=true
SGLANG_CACHE_DIT_TAYLORSEER=true
SGLANG_CACHE_DIT_TS_ORDER=1
```

## 5. Multi-Host Inference (torchrun)

### Source
- **Wan2.2 `generate.py`**: [github.com/Wan-Video/Wan2.2/blob/main/generate.py](https://github.com/Wan-Video/Wan2.2/blob/main/generate.py) — supports `torchrun` with `RANK`, `WORLD_SIZE`, `LOCAL_RANK` environment variables
- **Wan2.2 distributed init**: [`wan/distributed/util.py`](https://github.com/Wan-Video/Wan2.2/blob/main/wan/distributed/util.py) — `init_distributed_group()` initializes NCCL process group
- **Reference doc**: Described 2-node (16 GPU) deployment with `--ulysses_size 4` and Ring 2

### Technical Details
- `torchrun --nproc_per_node=8 --nnodes=2` launches 16 processes across 2 nodes.
- Each node runs 8 GPU workers.
- NCCL backend handles cross-node communication over VPC networking (400 Gbps total egress per G4 host).
- `--ulysses_size` must equal `world_size` (per Wan2.2 generate.py assertion: `ulysses_size == world_size`).
- Actually, looking at the code: `assert args.ulysses_size == world_size` — so for 16 GPUs, we use `--ulysses_size 16`.
- For the reference doc config with `--ulysses_size 4`, the team may have used a different code path (proprietary).

### Network Configuration
- `NCCL_SOCKET_IFNAME=eth0` — use the primary network interface
- `NCCL_IB_DISABLE=1` — disable InfiniBand (not available on G4)
- `NCCL_P2P_LEVEL=5` — enable intra-node P2P

## Wan2.2-A14B Model Architecture

From `wan/configs/`:
- **T2V config** (`wan_t2v_A14B.py`): `dim=5120, ffn_dim=13824, num_heads=40, num_layers=40, sample_steps=40, sample_shift=12.0, boundary=0.875, guide_scale=(3.0, 4.0)`
- **I2V config** (`wan_i2v_A14B.py`): `dim=5120, ffn_dim=13824, num_heads=40, num_layers=40, sample_steps=40, sample_shift=5.0, boundary=0.900, guide_scale=(3.5, 3.5)`
- **Shared config** (`shared_config.py`): `param_dtype=bfloat16, frame_num=81, sample_fps=16, text_len=512`
- **Architecture**: MoE (Mixture of Experts) with A14B = 14B active parameters, dual transformer experts (high/low noise)
