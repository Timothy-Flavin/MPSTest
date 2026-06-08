#!/bin/bash

# ==========================================
# ACTUAL TOPOLOGY DEFINITIONS (GPU on NUMA 1)
# ==========================================
# NUMA Node 0: Isolated for CPU-only anchor workloads
NUMA0_CPU_ONLY="0-15,32-47"

# NUMA Node 1: Slices for the GPU/MPS pipeline
NUMA1_FULL="16-31,48-63"

NUMA1_HALF1="16-23,48-55"
NUMA1_HALF2="24-31,56-63"

NUMA1_Q1="16-19,48-51"
NUMA1_Q2="20-23,52-55"
NUMA1_Q3="24-27,56-59"
NUMA1_Q4="28-31,60-63"

echo "========================================="
echo " INITIALIZING EXCLUSIVE MPS ON NUMA 1"
echo "========================================="
export CUDA_VISIBLE_DEVICES=0
sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
nvidia-cuda-mps-control -d
sleep 2


echo "========================================="
echo " TEST 1: 1 CPU-ONLY (NUMA 0) + 1 GPU/MPS (NUMA 1 @ 100%)"
echo "========================================="
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100

# CPU-only anchor hides GPU, targets Node 0 local memory
CUDA_VISIBLE_DEVICES="" taskset -c $NUMA0_CPU_ONLY numactl --preferred=0 python bench.py --job-id 99 --device cpu &
PID_CPU=$!

# GPU baseline targets full Node 1 local memory
taskset -c $NUMA1_FULL numactl --preferred=1 python bench.py --job-id 1 &
PID_GPU1=$!

wait $PID_CPU $PID_GPU1
echo -e "\nTest 1 Complete.\n"


echo "========================================="
echo " TEST 2: 1 CPU-ONLY (NUMA 0) + 2 GPU/MPS (NUMA 1 @ 50%)"
echo "========================================="
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50

CUDA_VISIBLE_DEVICES="" taskset -c $NUMA0_CPU_ONLY numactl --preferred=0 python bench.py --job-id 99 --device cpu &
PID_CPU=$!

taskset -c $NUMA1_HALF1 numactl --preferred=1 python bench.py --job-id 1 &
PID_GPU1=$!
taskset -c $NUMA1_HALF2 numactl --preferred=1 python bench.py --job-id 2 &
PID_GPU2=$!

wait $PID_CPU $PID_GPU1 $PID_GPU2
echo -e "\nTest 2 Complete.\n"


echo "========================================="
echo " TEST 3: 1 CPU-ONLY (NUMA 0) + 4 GPU/MPS (NUMA 1 @ 25%)"
echo "========================================="
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25

CUDA_VISIBLE_DEVICES="" taskset -c $NUMA0_CPU_ONLY numactl --preferred=0 python bench.py --job-id 99 --device cpu &
PID_CPU=$!

taskset -c $NUMA1_Q1 numactl --preferred=1 python bench.py --job-id 1 &
PID_GPU1=$!
taskset -c $NUMA1_Q2 numactl --preferred=1 python bench.py --job-id 2 &
PID_GPU2=$!
taskset -c $NUMA1_Q3 numactl --preferred=1 python bench.py --job-id 3 &
PID_GPU3=$!
taskset -c $NUMA1_Q4 numactl --preferred=1 python bench.py --job-id 4 &
PID_GPU4=$!

wait $PID_CPU $PID_GPU1 $PID_GPU2 $PID_GPU3 $PID_GPU4
echo -e "\nTest 3 Complete.\n"


echo "========================================="
echo " CLEANUP & REVERTING SYSTEM SETTINGS"
echo "========================================="
# Shut down daemon and restore compute mode to default shared access
echo quit | nvidia-cuda-mps-control
sudo nvidia-smi -i 0 -c DEFAULT

echo -e "\nAll hardware boundaries restored to default state.\n"
