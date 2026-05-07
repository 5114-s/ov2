#!/usr/bin/env python3
"""
Auto-run pipeline after depth estimation completes.
Monitors Step 1 progress and automatically runs Step 2, 3, 4 when done.
"""

import os
import sys
import time
import subprocess
import signal

# Configuration
ROOT = "/data/ZhaoX/OVM3D-Det"
PSEUDO_LABEL_DIR = os.path.join(ROOT, "pseudo_label", "SUNRGBD")
DEPTH_SCRIPT = os.path.join(ROOT, "third_party/metric-anything/models/student_depthmap/run_metric_anything_mp.py")

# Expected file counts
TRAIN_COUNT = 4929
VAL_COUNT = 356
EXPECTED_TOTAL = TRAIN_COUNT + VAL_COUNT  # 5285

def count_depth_files():
    """Count generated depth files"""
    train_count = 0
    val_count = 0
    
    train_dir = os.path.join(PSEUDO_LABEL_DIR, "train", "depth")
    val_dir = os.path.join(PSEUDO_LABEL_DIR, "val", "depth")
    
    if os.path.exists(train_dir):
        train_count = len([f for f in os.listdir(train_dir) if f.endswith('.npy')])
    if os.path.exists(val_dir):
        val_count = len([f for f in os.listdir(val_dir) if f.endswith('.npy')])
    
    return train_count, val_count

def run_command(cmd, description, wait=True):
    """Run a shell command"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    full_cmd = f"cd {ROOT} && eval \"$(~/miniconda3/bin/conda shell.bash hook)\" && conda activate ovm3d-1 && {cmd}"
    
    if wait:
        result = subprocess.run(full_cmd, shell=True, capture_output=False)
        return result.returncode == 0
    else:
        subprocess.Popen(full_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True

def monitor_and_run():
    """Main monitoring loop"""
    print("="*60)
    print("Auto Pipeline Monitor Started")
    print("="*60)
    
    # Check if depth processes are running
    check_cmd = "ps aux | grep run_metric_anything | grep -v grep | grep python"
    result = subprocess.run(check_cmd, shell=True, capture_output=True)
    
    if result.stdout.decode().strip():
        print("\n[Step 1] Depth estimation is running...")
    else:
        print("\n[Warning] No depth estimation process found!")
        print("Starting depth estimation manually...")
        
        # Start both GPUs
        run_command(f"cd {DEPTH_SCRIPT.replace('/run_metric_anything_mp.py', '')} && nohup python run_metric_anything_mp.py --root {ROOT} --dataset SUNRGBD --gpu 0 --start 0 --end 2464 > gpu0_run.log 2>&1 &", "GPU 0 depth estimation", wait=False)
        time.sleep(2)
        run_command(f"cd {DEPTH_SCRIPT.replace('/run_metric_anything_mp.py', '')} && nohup python run_metric_anything_mp.py --root {ROOT} --dataset SUNRGBD --gpu 1 --start 2464 --end 4929 > gpu1_run.log 2>&1 &", "GPU 1 depth estimation", wait=False)
    
    # Monitor loop
    last_report = 0
    while True:
        train_count, val_count = count_depth_files()
        total = train_count + val_count
        
        # Report every 30 seconds
        if time.time() - last_report > 30:
            progress = total / EXPECTED_TOTAL * 100
            print(f"\n[{time.strftime('%H:%M:%S')}] Depth files: train={train_count}/{TRAIN_COUNT}, val={val_count}/{VAL_COUNT}, total={total}/{EXPECTED_TOTAL} ({progress:.1f}%)")
            last_report = time.time()
        
        # Check if both GPUs are still running
        result = subprocess.run(check_cmd, shell=True, capture_output=True)
        gpu_running = bool(result.stdout.decode().strip())
        
        # Check if complete or if we should move on (at least train set is complete)
        if total >= EXPECTED_TOTAL and not gpu_running:
            print(f"\n[Step 1] COMPLETE! Generated {total} depth files.")
            break
        elif train_count >= TRAIN_COUNT and not gpu_running:
            print(f"\n[Step 1] Train set complete! Generated {train_count} train depth files.")
            print("Note: Val set may not be processed. Continuing anyway...")
            break
        
        time.sleep(10)
    
    # Step 2: Grounded-SAM Segmentation
    print("\n" + "="*60)
    print("Starting Step 2: Grounded-SAM Segmentation")
    print("="*60)
    
    run_command(
        "CUDA_VISIBLE_DEVICES=0 python third_party/Grounded-Segment-Anything/grounded_sam_detect.py --dataset SUNRGBD",
        "Step 2a: Grounded-SAM detection"
    )
    
    run_command(
        "CUDA_VISIBLE_DEVICES=0 python third_party/Grounded-Segment-Anything/grounded_sam_detect_ground.py --dataset SUNRGBD",
        "Step 2b: Grounded-SAM ground detection"
    )
    
    # Step 3: Generate 3D pseudo bboxes
    print("\n" + "="*60)
    print("Starting Step 3: Generate 3D pseudo bboxes")
    print("="*60)
    
    run_command(
        "YAW_METHOD=pca python tools/generate_pseudo_bbox.py --config-file configs/Base_Omni3D_SUN.yaml OUTPUT_DIR output/generate_pseudo_label/SUNRGBD",
        "Step 3: Generate 3D pseudo bboxes (PCA)"
    )
    
    # Step 4: Convert to COCO format
    print("\n" + "="*60)
    print("Starting Step 4: Convert to COCO format")
    print("="*60)
    
    run_command(
        "python tools/transform_to_coco.py --dataset_name SUNRGBD",
        "Step 4: Convert to COCO format"
    )
    
    print("\n" + "="*60)
    print("ALL STEPS COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    try:
        monitor_and_run()
    except KeyboardInterrupt:
        print("\n\nMonitor interrupted by user.")
        sys.exit(0)
