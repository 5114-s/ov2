#!/usr/bin/env python3
"""
Multi-GPU launcher for generate_pseudo_bbox.py
Splits dataset across multiple GPUs for parallel processing
"""
import os
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-gpus', type=int, default=2, help='Number of GPUs to use')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID for this process')
    parser.add_argument('--config-file', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--input-folder', type=str, default='pseudo_label/SUNRGBD/train')
    parser.add_argument('--yaw-method', type=str, default='labelany3d')
    args = parser.parse_args()

    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Calculate image range for this GPU
    total_images = 4639  # Approximate
    images_per_gpu = total_images // args.num_gpus
    start_idx = args.gpu_id * images_per_gpu
    end_idx = start_idx + images_per_gpu if args.gpu_id < args.num_gpus - 1 else total_images

    print(f"[GPU {args.gpu_id}] Processing images {start_idx} to {end_idx}")

    # Run the generate_pseudo_bbox script with modified dataset
    # We'll modify the dataset by setting environment variable for start/end indices
    os.environ['PROCESS_START_IDX'] = str(start_idx)
    os.environ['PROCESS_END_IDX'] = str(end_idx)

    cmd = [
        sys.executable,
        'tools/generate_pseudo_bbox.py',
        '--config-file', args.config_file,
        'OUTPUT_DIR', args.output_dir,
        'YAW_METHOD', args.yaw_method,
    ]

    os.execvp(cmd[0], cmd)

if __name__ == '__main__':
    main()
