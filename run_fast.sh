#!/bin/bash
# Fast run with mast3r_hybrid (no TRELLIS needed, ~1.5s per object)

CUDA_VISIBLE_DEVICES=0 \
python tools/generate_pseudo_bbox.py \
  --config-file configs/Base_Omni3D_SUN.yaml \
  OUTPUT_DIR output/generate_pseudo_label_fast/SUNRGBD \
  YAW_METHOD mast3r_hybrid 2>&1 | tee run_fast.log
