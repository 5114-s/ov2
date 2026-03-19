# 1. 显存优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 2. 恢复训练 (注意多了 --resume 参数)
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
  --config-file configs/Base_Omni3D_KITTI.yaml \
  --num-gpus 1 \
  --resume \
  OUTPUT_DIR output/training/KITTI