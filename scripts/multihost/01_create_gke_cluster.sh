#!/bin/bash
# Create GKE cluster with dual-NIC G4 nodes for multi-host Wan2.2 inference
# Requires: gcloud beta, GPU quota for g4-standard-384
set -e

export PROJECT_ID="${PROJECT_ID:-gpu-launchpad-playground}"
export ZONE="${ZONE:-us-west1-a}"
export CLUSTER_NAME="wan22-mh"
export VPC_NAME="wan22-vpc"
export SUBNET1="wan22-sub"
export SUBNET2="wan22-sub2"
export REGION="${ZONE%-*}"

echo "=== Step 1: Create custom VPC ==="
gcloud compute networks create ${VPC_NAME} \
  --project=${PROJECT_ID} --subnet-mode=custom --quiet

echo "=== Step 2: Create primary subnet (for pod networking) ==="
gcloud compute networks subnets create ${SUBNET1} \
  --project=${PROJECT_ID} --network=${VPC_NAME} --region=${REGION} \
  --range=10.100.0.0/16 \
  --secondary-range pods=10.104.0.0/14,services=10.108.0.0/20 \
  --quiet

echo "=== Step 3: Create secondary subnet (for second NIC) ==="
gcloud compute networks subnets create ${SUBNET2} \
  --project=${PROJECT_ID} --network=${VPC_NAME} --region=${REGION} \
  --range=10.110.0.0/16 \
  --quiet

echo "=== Step 4: Create firewall rules ==="
gcloud compute firewall-rules create ${VPC_NAME}-allow-internal \
  --project=${PROJECT_ID} --network=${VPC_NAME} \
  --allow=tcp,udp,icmp --source-ranges=10.0.0.0/8 --quiet

echo "=== Step 5: Create GKE cluster with Dataplane V2 + multi-networking ==="
gcloud container clusters create ${CLUSTER_NAME} \
  --project=${PROJECT_ID} --zone=${ZONE} \
  --network=${VPC_NAME} --subnetwork=${SUBNET1} \
  --cluster-secondary-range-name=pods --services-secondary-range-name=services \
  --machine-type=e2-medium --num-nodes=1 \
  --enable-ip-alias \
  --enable-dataplane-v2 \
  --enable-multi-networking \
  --no-enable-autoupgrade --no-enable-autorepair \
  --quiet

echo "=== Step 6: Add G4 node pool with dual NIC ==="
gcloud beta container node-pools create g4-pool \
  --project=${PROJECT_ID} --zone=${ZONE} --cluster=${CLUSTER_NAME} \
  --machine-type=g4-standard-384 \
  --accelerator=type=nvidia-rtx-pro-6000,count=8 \
  --num-nodes=2 \
  --placement-type=COMPACT \
  --spot \
  --disk-size=500 \
  --image-type=UBUNTU_CONTAINERD \
  --additional-node-network="${VPC_NAME},${SUBNET2}" \
  --no-enable-autoupgrade --no-enable-autorepair \
  --quiet

echo "=== Step 7: Get credentials ==="
gcloud container clusters get-credentials ${CLUSTER_NAME} \
  --zone=${ZONE} --project=${PROJECT_ID}

echo "=== Verify ==="
kubectl get nodes -o wide
echo ""
echo "GKE cluster ${CLUSTER_NAME} created with 2x G4 dual-NIC nodes!"
echo "Primary NIC: ens3 (${SUBNET1})"
echo "Secondary NIC: enp128s4 (${SUBNET2})"
