#!/bin/bash

# Define core affinity arrays based on NUMA Node 0 core map
JOB1_CORES="0-3,32-35"
JOB2_CORES="4-7,36-39"
JOB3_CORES="8-11,40-43"
JOB4_CORES="12-15,44-47"

echo "========================================="
echo " INITIALIZING EXCLUSIVE MPS ON NUMA 0"
echo "========================================="
export CUDA_VISIBLE_DEVICES=0

# Set exclusive process mode to block un-mapped scripts
sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS

# Launch the MPS control daemon
nvidia-cuda-mps-control -d
sleep 2

# Cap active threads to 25% for 4 balanced jobs on the Ada 6000
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25

echo "========================================="
echo " RUNNING QUAD WITH CUDA MPS"
echo "========================================="

# numactl --preferred=0 restricts RAM allocation to local socket memory pools
taskset -c $JOB1_CORES numactl --preferred=0 python bench.py --job-id 1 &
PID1=$!

taskset -c $JOB2_CORES numactl --preferred=0 python bench.py --job-id 2 &
PID2=$!

taskset -c $JOB3_CORES numactl --preferred=0 python bench.py --job-id 3 &
PID3=$!

taskset -c $JOB4_CORES numactl --preferred=0 python bench.py --job-id 4 &
PID4=$!

# Monitor background execution blocks
wait $PID1 $PID2 $PID3 $PID4

echo "========================================="
echo " CLEANUP & REVERTING SYSTEM SETTINGS"
echo "========================================="
echo quit | nvidia-cuda-mps-control
sudo nvidia-smi -i 0 -c DEFAULT

echo -e "\nWorkstation benchmark complete.\n"
