#!/bin/bash
# Smart hybrid: PCA for simple objects, MASt3R+TRELLIS for complex objects
# Usage: bash run_pca_trellis_hybrid.sh

mkdir -p output/pca_trellis_hybrid/SUNRGBD

# Single GPU version
YAW_METHOD=pca_trellis_hybrid \
CUDA_VISIBLE_DEVICES=0 \
python tools/generate_pseudo_bbox.py \
  --config-file configs/Base_Omni3D_SUN.yaml \
  OUTPUT_DIR output/pca_trellis_hybrid/SUNRGBD 2>&1 | tee run_pca_trellis_hybrid.log

echo "Done! Check run_pca_trellis_hybrid.log"
