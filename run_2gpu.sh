#!/bin/bash
# Multi-GPU launcher for generate_pseudo_bbox.py
# Usage: bash run_2gpu.sh

mkdir -p output/generate_pseudo_label_gpu0/SUNRGBD
mkdir -p output/generate_pseudo_label_gpu1/SUNRGBD

# GPU 0: images 0-2300
CUDA_VISIBLE_DEVICES=0 START_IDX=0 END_IDX=2300 \
  nohup python tools/generate_pseudo_bbox.py \
    --config-file configs/Base_Omni3D_SUN.yaml \
    OUTPUT_DIR output/generate_pseudo_label_gpu0/SUNRGBD \
    YAW_METHOD labelany3d > gpu0.log 2>&1 &

echo "GPU 0 started (PID: $!)"

# GPU 1: images 2300-end
CUDA_VISIBLE_DEVICES=1 START_IDX=2300 END_IDX=0 \
  nohup python tools/generate_pseudo_bbox.py \
    --config-file configs/Base_Omni3D_SUN.yaml \
    OUTPUT_DIR output/generate_pseudo_label_gpu1/SUNRGBD \
    YAW_METHOD labelany3d > gpu1.log 2>&1 &

echo "GPU 1 started (PID: $!)"
echo "Done. Monitor with: tail -f gpu0.log / tail -f gpu1.log"
