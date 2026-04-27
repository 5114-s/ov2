#!/bin/bash

# Usage: bash scripts/generate_pseudo_label.sh <DATASET> [SKIP_STEPS]
# SKIP_STEPS: comma-separated list of steps to skip, e.g. "1,2" skips steps 1 and 2
#   Step 1: Depth prediction (UniDepth)
#   Step 2: Segmentation (Grounded-SAM)
#   Step 3: Generate 3D pseudo bboxes
#   Step 4: Convert to COCO format

DATASET=$1
SKIP_STEPS=${2:-""}

if [ -z "$DATASET" ]; then
    echo "Usage: bash scripts/generate_pseudo_label.sh <DATASET> [SKIP_STEPS]"
    echo "  e.g.: bash scripts/generate_pseudo_label.sh SUNRGBD"
    echo "  e.g.: bash scripts/generate_pseudo_label.sh SUNRGBD 1,2"
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

# Step 1: Predict depth using UniDepth
if ! should_skip 1; then
    echo "=== Step 1: Depth prediction (UniDepth) ==="
    CUDA_VISIBLE_DEVICES=0 python third_party/UniDepth/run_unidepth.py --dataset $DATASET
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
    echo "=== Step 3: Generate 3D pseudo bboxes (hybrid) ==="
    YAW_METHOD=hybrid python tools/generate_pseudo_bbox.py \
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
