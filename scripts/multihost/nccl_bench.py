"""
NCCL Inter-Node Bandwidth Benchmark
Tests all-reduce performance across multiple nodes with PyTorch distributed.
Used with torchrun --nnodes=2 --nproc_per_node=8
"""
import torch
import torch.distributed as dist
import os
import time

dist.init_process_group('nccl')
rank = dist.get_rank()
ws = dist.get_world_size()
dev = rank % 8
torch.cuda.set_device(dev)

if rank == 0:
    print(f'Initialized {ws} ranks across nodes')

# Warmup
for _ in range(5):
    t = torch.ones(1024*1024, device=f'cuda:{dev}')
    dist.all_reduce(t)
torch.cuda.synchronize()

# Benchmark at various message sizes
sizes_mb = [4, 16, 64, 256, 1024]
for mb in sizes_mb:
    n = mb * 1024 * 256  # number of float32 elements
    t = torch.ones(n, device=f'cuda:{dev}')
    torch.cuda.synchronize()
    dist.barrier()
    start = time.time()
    for _ in range(10):
        dist.all_reduce(t)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    bw = (n * 4 * 2 * 10) / elapsed / 1e9  # algorithm bandwidth in GB/s
    if rank == 0:
        print(f'  {mb}MB: algBW={bw:.2f} GB/s, time={elapsed/10*1000:.1f}ms/op')

if rank == 0:
    print('=== NCCL inter-node test DONE ===')
dist.destroy_process_group()
