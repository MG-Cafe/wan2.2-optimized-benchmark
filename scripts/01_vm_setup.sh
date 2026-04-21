#!/bin/bash
#
# VM Setup Script - Docker + NVIDIA Container Toolkit + SageAttention
# Runs on a fresh G4 VM to prepare for benchmarks
#
set -euo pipefail

echo "=== [1/5] Verifying GPUs ==="
nvidia-smi

echo "=== [2/5] Installing Docker ==="
sudo apt-get update -qq
sudo apt-get install -y -qq ca-certificates curl > /dev/null
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update -qq
sudo apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin > /dev/null
sudo systemctl start docker && sudo systemctl enable docker

echo "=== [3/5] Installing NVIDIA Container Toolkit ==="
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
sudo apt-get update -qq
sudo apt-get install -y -qq nvidia-container-toolkit > /dev/null
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo "=== [4/5] Preparing scratch directory ==="
sudo mkdir -p /scratch/cache /scratch/results /scratch/models
sudo chmod -R 777 /scratch

echo "=== [5/5] Resizing filesystem ==="
sudo growpart /dev/nvme0n1 1 2>/dev/null || true
sudo resize2fs /dev/nvme0n1p1 2>/dev/null || true
echo "Disk space:" && df -h / | tail -1

echo ""
echo "=== VM Setup Complete ==="
echo "GPU count: $(nvidia-smi -L | wc -l)"
echo "Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
echo "CUDA: $(nvidia-smi --query-gpu=cuda_version --format=csv,noheader 2>/dev/null | head -1 || echo 'N/A')"
