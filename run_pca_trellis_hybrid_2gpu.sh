#!/bin/bash
# Smart hybrid with 2 GPUs: PCA for simple, MASt3R+TRELLIS for complex
# Usage: bash run_pca_trellis_hybrid_2gpu.sh

# Source conda
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate ovm3d-1

mkdir -p output/pca_trellis_hybrid_gpu0/SUNRGBD
mkdir -p output/pca_trellis_hybrid_gpu1/SUNRGBD

# GPU 0: images 0-2300
CUDA_VISIBLE_DEVICES=0 YAW_METHOD=pca_trellis_hybrid START_IDX=0 END_IDX=2300 \
  nohup python tools/generate_pseudo_bbox.py \
    --config-file configs/Base_Omni3D_SUN.yaml \
    OUTPUT_DIR output/pca_trellis_hybrid_gpu0/SUNRGBD > gpu0_hybrid.log 2>&1 &

echo "GPU 0 started (PID: $!)"

# GPU 1: images 2300-end
CUDA_VISIBLE_DEVICES=1 YAW_METHOD=pca_trellis_hybrid START_IDX=2300 END_IDX=0 \
  nohup python tools/generate_pseudo_bbox.py \
    --config-file configs/Base_Omni3D_SUN.yaml \
    OUTPUT_DIR output/pca_trellis_hybrid_gpu1/SUNRGBD > gpu1_hybrid.log 2>&1 &

echo "GPU 1 started (PID: $!)"
echo ""
echo "Done! Monitor with:"
echo "  tail -f gpu0_hybrid.log"
echo "  tail -f gpu1_hybrid.log"
