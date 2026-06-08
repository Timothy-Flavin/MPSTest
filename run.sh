#!/bin/bash

# Define core affinity arrays for the 16-core / 32-thread 9950x
# CCD0 splits:
JOB1_CORES="0-3,16-19"
JOB2_CORES="4-7,20-23"
# CCD1 splits:
JOB3_CORES="8-11,24-27"
JOB4_CORES="12-15,28-31"

# Combined sets for the 2-job baselines
DUAL_JOB1="0-7,16-23"
DUAL_JOB2="8-15,24-31"

echo "========================================="
echo " 1. RUNNING SINGLE ENVIRONMENT BASELINE"
echo "========================================="
taskset -c $JOB1_CORES python bench.py --job-id 1 &
PID1=$!

wait $PID1 
echo -e "\nSingle baseline complete.\n"


echo "========================================="
echo " 2. RUNNING DUAL BASELINE (NO MPS)"
echo "========================================="
taskset -c $DUAL_JOB1 python bench.py --job-id 1 &
PID1=$!
taskset -c $DUAL_JOB2 python bench.py --job-id 2 &
PID2=$!

wait $PID1 $PID2
echo -e "\nDual baseline complete.\n"


echo "========================================="
echo " 3. RUNNING DUAL WITH CUDA MPS (50% ALLOCATION)"
echo "========================================="
export CUDA_VISIBLE_DEVICES=0
sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
nvidia-cuda-mps-control -d
sleep 2

export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50

taskset -c $DUAL_JOB1 python bench.py --job-id 1 &
PID1=$!
taskset -c $DUAL_JOB2 python bench.py --job-id 2 &
PID2=$!

wait $PID1 $PID2

# Tear down 50% MPS session to prepare for 25% allocations
echo quit | nvidia-cuda-mps-control
sleep 1


echo "========================================="
echo " 4. RUNNING QUAD WITH CUDA MPS (25% ALLOCATION)"
echo "========================================="
# Restart daemon to apply new environmental variable topology cleanly
nvidia-cuda-mps-control -d
sleep 2

export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25

taskset -c $JOB1_CORES python bench.py --job-id 1 &
PID1=$!
taskset -c $JOB2_CORES python bench.py --job-id 2 &
PID2=$!
taskset -c $JOB3_CORES python bench.py --job-id 3 &
PID3=$!
taskset -c $JOB4_CORES python bench.py --job-id 4 &
PID4=$!

wait $PID1 $PID2 $PID3 $PID4


echo "========================================="
echo " CLEANUP & REVERTING SYSTEM SETTINGS"
echo "========================================="
echo quit | nvidia-cuda-mps-control
sudo nvidia-smi -i 0 -c DEFAULT

echo -e "\nAll benchmarks complete.\n"