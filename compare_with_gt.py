#!/usr/bin/env python3
"""Compare pseudo_label depth with ground truth depth from SUNRGBD"""

import numpy as np
from pathlib import Path
import os

# Find files that exist in all three directories
pseudo_730 = Path("/data/ZhaoX/OVM3D-Det/pseudo_label/SUNRGBD/train/depth")
pseudo_uni = Path("/data/ZhaoX/OVM3D-Det/pseudo_label-2/SUNRGBD/train/depth")
gt_dir = Path("/data/ZhaoX/OVM3D-Det/SUNRGBD/train/depth")

# Check what files exist
files_730 = set(f.stem for f in pseudo_730.glob("*.npy")) if pseudo_730.exists() else set()
files_uni = set(f.stem for f in pseudo_uni.glob("*.npy")) if pseudo_uni.exists() else set()
files_gt = set(f.stem for f in gt_dir.glob("*.npy")) if gt_dir.exists() else set()

print(f"pseudo_label (730): {len(files_730)} files")
print(f"pseudo_label-2 (UniDepth): {len(files_uni)} files")
print(f"GT depth: {len(files_gt)} files")
print()

# Find common files
common_730_gt = sorted(files_730 & files_gt)
common_uni_gt = sorted(files_uni & files_gt)
common_all = sorted(files_730 & files_uni & files_gt)

print(f"Common files (730 + GT): {len(common_730_gt)}")
print(f"Common files (UniDepth + GT): {len(common_uni_gt)}")
print(f"Common files (all three): {len(common_all)}")
print()

if len(common_all) == 0:
    print("No common files found!")
    exit(1)

# Sample comparison
sample_files = common_all[:10]

print("=== 深度对比 (与 GT 相比) ===\n")
for fname in sample_files:
    depth_730 = np.load(pseudo_730 / f"{fname}.npy")
    depth_uni = np.load(pseudo_uni / f"{fname}.npy")
    depth_gt = np.load(gt_dir / f"{fname}.npy")

    # Filter out invalid GT values (GT depth might have 0 or very small values)
    valid_mask = depth_gt > 0.1  # Filter invalid pixels

    if valid_mask.sum() == 0:
        continue

    depth_730_valid = depth_730[valid_mask]
    depth_uni_valid = depth_uni[valid_mask]
    depth_gt_valid = depth_gt[valid_mask]

    # Calculate metrics
    mae_730 = np.abs(depth_730_valid - depth_gt_valid).mean()
    mae_uni = np.abs(depth_uni_valid - depth_gt_valid).mean()

    rmse_730 = np.sqrt(((depth_730_valid - depth_gt_valid)**2).mean())
    rmse_uni = np.sqrt(((depth_uni_valid - depth_gt_valid)**2).mean())

    rel_730 = np.abs(depth_730_valid - depth_gt_valid) / depth_gt_valid
    rel_uni = np.abs(depth_uni_valid - depth_gt_valid) / depth_gt_valid
    rel_730_mean = rel_730.mean()
    rel_uni_mean = rel_uni.mean()

    print(f"文件: {fname}.npy")
    print(f"  GT 深度: min={depth_gt_valid.min():.3f}, max={depth_gt_valid.max():.3f}, mean={depth_gt_valid.mean():.3f}")
    print(f"  730 深度: mean={depth_730_valid.mean():.3f}, MAE={mae_730:.3f}, RMSE={rmse_730:.3f}, REL={rel_730_mean:.3f}")
    print(f"  UniDepth: mean={depth_uni_valid.mean():.3f}, MAE={mae_uni:.3f}, RMSE={rmse_uni:.3f}, REL={rel_uni_mean:.3f}")

    # Which is better?
    if mae_730 < mae_uni:
        print(f"  => 730 更接近 GT (MAE 差异: {mae_uni - mae_730:.3f})")
    elif mae_uni < mae_730:
        print(f"  => UniDepth 更接近 GT (MAE 差异: {mae_730 - mae_uni:.3f})")
    else:
        print(f"  => 两者相近")
    print()

# Aggregate statistics
print("=== 总体统计 (前100个文件) ===\n")
maes_730 = []
maes_uni = []
rmses_730 = []
rmses_uni = []

for fname in common_all[:100]:
    try:
        depth_730 = np.load(pseudo_730 / f"{fname}.npy")
        depth_uni = np.load(pseudo_uni / f"{fname}.npy")
        depth_gt = np.load(gt_dir / f"{fname}.npy")

        valid_mask = depth_gt > 0.1
        if valid_mask.sum() < 100:
            continue

        depth_730_valid = depth_730[valid_mask]
        depth_uni_valid = depth_uni[valid_mask]
        depth_gt_valid = depth_gt[valid_mask]

        mae_730 = np.abs(depth_730_valid - depth_gt_valid).mean()
        mae_uni = np.abs(depth_uni_valid - depth_gt_valid).mean()
        rmse_730 = np.sqrt(((depth_730_valid - depth_gt_valid)**2).mean())
        rmse_uni = np.sqrt(((depth_uni_valid - depth_gt_valid)**2).mean())

        maes_730.append(mae_730)
        maes_uni.append(mae_uni)
        rmses_730.append(rmse_730)
        rmses_uni.append(rmse_uni)
    except Exception as e:
        continue

if maes_730:
    print(f"文件数: {len(maes_730)}")
    print(f"\n730 focal:")
    print(f"  平均 MAE: {np.mean(maes_730):.4f}")
    print(f"  平均 RMSE: {np.mean(rmses_730):.4f}")
    print(f"\nUniDepth:")
    print(f"  平均 MAE: {np.mean(maes_uni):.4f}")
    print(f"  平均 RMSE: {np.mean(rmses_uni):.4f}")

    diff_mae = np.mean(maes_730) - np.mean(maes_uni)
    if diff_mae < 0:
        print(f"\n结论: 730 focal 更接近 GT (MAE 差异: {-diff_mae:.4f})")
    else:
        print(f"\n结论: UniDepth 更接近 GT (MAE 差异: {diff_mae:.4f})")
