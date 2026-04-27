#!/bin/bash

# Usage: bash scripts/generate_pseudo_label.sh <DATASET> [SKIP_STEPS] [DEPTH_METHOD]
# DEPTH_METHOD: unidepth (default) | moge_depthpro

DATASET=$1
SKIP_STEPS=${2:-""}
DEPTH_METHOD=${3:-"unidepth"}

if [ -z "$DATASET" ]; then
    echo "Usage: bash scripts/generate_pseudo_label.sh <DATASET> [SKIP_STEPS] [DEPTH_METHOD]"
    echo "  e.g.: bash scripts/generate_pseudo_label.sh SUNRGBD"
    echo "  e.g.: bash scripts/generate_pseudo_label.sh SUNRGBD 1,2"
    echo "  e.g.: bash scripts/generate_pseudo_label.sh SUNRGBD '' moge_depthpro"
    echo "  DEPTH_METHOD: unidepth (default) | moge_depthpro"
    exit 1
fi

# Map display names to config names
case $DATASET in
    "SUNRGBD") CONFIG_NAME="SUN" ;;
    *) CONFIG_NAME=$DATASET ;;
esac

# Helper: check if a step should be skipped
should_skip() {
    echo "$SKIP_STEPS" | tr ',' '\n' | grep -qx "$1"
}

# Activate conda environment
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate ovm3d-1

# Step 1: Predict depth
if ! should_skip 1; then
    if [ "$DEPTH_METHOD" = "moge_depthpro" ]; then
        echo "=== Step 1: Depth prediction (MoGe+DepthPro fusion) ==="
        CUDA_VISIBLE_DEVICES=0 python third_party/fused_depth/run_fused_depth.py --dataset $DATASET
    else
        echo "=== Step 1: Depth prediction (UniDepth) ==="
        CUDA_VISIBLE_DEVICES=0 python third_party/UniDepth/run_unidepth.py --dataset $DATASET
    fi
else
    echo "=== Step 1: Skipped ==="
fi

# Step 2: Segment novel objects using Grounded-SAM
if ! should_skip 2; then
    echo "=== Step 2: Segmentation (Grounded-SAM) ==="
    CUDA_VISIBLE_DEVICES=0 python third_party/Grounded-Segment-Anything/grounded_sam_detect.py --dataset $DATASET
    CUDA_VISIBLE_DEVICES=0 python third_party/Grounded-Segment-Anything/grounded_sam_detect_ground.py --dataset $DATASET
else
    echo "=== Step 2: Skipped ==="
fi

# Step 3: Generate pseudo 3D bounding boxes
if ! should_skip 3; then
    echo "=== Step 3: Generate 3D pseudo bboxes (pca) ==="
    YAW_METHOD=pca python tools/generate_pseudo_bbox.py \
      --config-file configs/Base_Omni3D_${CONFIG_NAME}.yaml \
      OUTPUT_DIR output/generate_pseudo_label/$DATASET
else
    echo "=== Step 3: Skipped ==="
fi

# Step 4: Convert to COCO dataset format
if ! should_skip 4; then
    echo "=== Step 4: Convert to COCO format ==="
    python tools/transform_to_coco.py --dataset_name $DATASET
else
    echo "=== Step 4: Skipped ==="
fi

echo "Done."
