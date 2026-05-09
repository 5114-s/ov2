#!/usr/bin/env python3
"""
Multi-GPU runner: splits dataset by image_id and processes in parallel
Usage: python run_multi_gpu_parallel.py
"""
import os
import subprocess
import time

# GPU configuration
GPUS = [0, 1]
IMAGE_SPLITS = {
    0: (162893, 180000),  # GPU0: image_id 162893-180000
    1: (180000, 210000),  # GPU1: image_id 180000-210000
}

def main():
    processes = []

    for gpu_id in GPUS:
        start_id, end_id = IMAGE_SPLITS[gpu_id]
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        env['YAW_METHOD'] = 'labelany3d'
        env['PROCESS_START_ID'] = str(start_id)
        env['PROCESS_END_ID'] = str(end_id)
        env['OUTPUT_DIR'] = f'output/generate_pseudo_label_gpu{gpu_id}/SUNRGBD'
        env['SKIP_STEP7_8'] = '1'  # Skip slow steps

        cmd = [
            'python', 'tools/generate_pseudo_bbox.py',
            '--config-file', 'configs/Base_Omni3D_SUN.yaml',
            'OUTPUT_DIR', env['OUTPUT_DIR'],
        ]

        print(f"[GPU {gpu_id}] Starting: image_id {start_id} to {end_id}")
        print(f"[GPU {gpu_id}] Command: {' '.join(cmd)}")

        log_file = open(f'multi_gpu_gpu{gpu_id}.log', 'w')
        p = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((gpu_id, p, log_file))

    print("\nAll processes started. Waiting for completion...")
    print("To monitor: tail -f multi_gpu_gpu0.log / multi_gpu_gpu1.log")

    # Wait for all processes
    for gpu_id, p, log_file in processes:
        p.wait()
        log_file.close()
        print(f"[GPU {gpu_id}] Process completed")

    print("\nAll done!")

if __name__ == '__main__':
    main()
