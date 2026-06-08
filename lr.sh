#!/bin/bash

# ==========================================
# NUMA TOPOLOGY DEFINITIONS
# ==========================================
# NUMA Node 1: Dedicated purely to the CPU-only baseline job
NUMA1_CPU_ONLY="16-31,48-63"

# NUMA Node 0: GPU-bound node, sliced for 1, 2, and 4 concurrent jobs
NUMA0_FULL="0-15,32-47"

NUMA0_HALF1="0-7,32-39"
NUMA0_HALF2="8-15,40-47"

NUMA0_Q1="0-3,32-35"
NUMA0_Q2="4-7,36-39"
NUMA0_Q3="8-11,40-43"
NUMA0_Q4="12-15,44-47"

echo "========================================="
echo " INITIALIZING EXCLUSIVE MPS ON NUMA 0"
echo "========================================="
export CUDA_VISIBLE_DEVICES=0
sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
nvidia-cuda-mps-control -d
sleep 2


echo "========================================="
echo " TEST 1: 1 CPU-ONLY JOB + 1 GPU/MPS JOB (100% ALLOC)"
echo "========================================="
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100

# CPU Job on NUMA 1: Hide GPU, force CPU device, bind memory to Node 1
CUDA_VISIBLE_DEVICES="" taskset -c $NUMA1_CPU_ONLY numactl --preferred=1 python bench.py --job-id 99 --device cpu &
PID_CPU=$!

# GPU Job on NUMA 0: Use full node, bind memory to Node 0
taskset -c $NUMA0_FULL numactl --preferred=0 python bench.py --job-id 1 &
PID_GPU1=$!

wait $PID_CPU $PID_GPU1
echo -e "\nTest 1 Complete.\n"


echo "========================================="
echo " TEST 2: 1 CPU-ONLY JOB + 2 GPU/MPS JOBS (50% ALLOC)"
echo "========================================="
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50

# CPU Job on NUMA 1
CUDA_VISIBLE_DEVICES="" taskset -c $NUMA1_CPU_ONLY numactl --preferred=1 python bench.py --job-id 99 --device cpu &
PID_CPU=$!

# GPU Jobs split across NUMA 0
taskset -c $NUMA0_HALF1 numactl --preferred=0 python bench.py --job-id 1 &
PID_GPU1=$!
taskset -c $NUMA0_HALF2 numactl --preferred=0 python bench.py --job-id 2 &
PID_GPU2=$!

wait $PID_CPU $PID_GPU1 $PID_GPU2
echo -e "\nTest 2 Complete.\n"


echo "========================================="
echo " TEST 3: 1 CPU-ONLY JOB + 4 GPU/MPS JOBS (25% ALLOC)"
echo "========================================="
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25

# CPU Job on NUMA 1
CUDA_VISIBLE_DEVICES="" taskset -c $NUMA1_CPU_ONLY numactl --preferred=1 python bench.py --job-id 99 --device cpu &
PID_CPU=$!

# GPU Jobs mapped to quadrants on NUMA 0
taskset -c $NUMA0_Q1 numactl --preferred=0 python bench.py --job-id 1 &
PID_GPU1=$!
taskset -c $NUMA0_Q2 numactl --preferred=0 python bench.py --job-id 2 &
PID_GPU2=$!
taskset -c $NUMA0_Q3 numactl --preferred=0 python bench.py --job-id 3 &
PID_GPU3=$!
taskset -c $NUMA0_Q4 numactl --preferred=0 python bench.py --job-id 4 &
PID_GPU4=$!

wait $PID_CPU $PID_GPU1 $PID_GPU2 $PID_GPU3 $PID_GPU4
echo -e "\nTest 3 Complete.\n"


echo "========================================="
echo " CLEANUP & REVERTING SYSTEM SETTINGS"
echo "========================================="
echo quit | nvidia-cuda-mps-control
sudo nvidia-smi -i 0 -c DEFAULT

echo -e "\nAll multi-node benchmarks complete.\n"