#!/usr/bin/env python3
"""
Apply scale compensation to MetricAnything depth maps to match UniDepth scale.

This fixes the focal-length distribution mismatch between SUNRGBD and MetricAnything's training data.
"""

import json
import numpy as np
import os
import argparse
from tqdm import tqdm

# Scale factors: multiply MetricAnything depth by these values to match UniDepth
SCALE_FACTORS = {
    "kv1": 1.0425,
    "kv2": 1.7585,
    "realsense": 0.9526,
    "xtion": 0.9269,
}


def get_camera_from_id(image_id: int, id_to_camera: dict) -> str:
    """Get camera type from image ID."""
    return id_to_camera.get(image_id, "unknown")


def apply_compensation(
    input_dir: str,
    output_dir: str,
    json_path: str,
    dry_run: bool = False,
) -> None:
    """
    Apply scale compensation to depth maps.

    Args:
        input_dir: Directory containing original MetricAnything depth maps
        output_dir: Directory to save compensated depth maps
        json_path: Path to Omni3D JSON file with camera info
        dry_run: If True, only compute statistics without saving
    """
    # Load image ID to camera mapping
    with open(json_path, 'r') as f:
        data = json.load(f)
    id_to_camera = {img['id']: img['file_path'].split('/')[1] for img in data['images']}

    # Find all depth files
    depth_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
    print(f"Found {len(depth_files)} depth files")

    # Statistics
    stats = {cam: {'count': 0, 'mean_ratio': []} for cam in SCALE_FACTORS}
    stats['unknown'] = {'count': 0, 'mean_ratio': []}

    if dry_run:
        # Just compute and display statistics
        for fname in tqdm(depth_files, desc="Analyzing"):
            image_id = int(fname.replace('.npy', ''))
            camera = get_camera_from_id(image_id, id_to_camera)

            if camera in SCALE_FACTORS:
                stats[camera]['count'] += 1
            else:
                stats['unknown']['count'] += 1

        print("\nStatistics:")
        for cam, s in stats.items():
            if s['count'] > 0:
                print(f"  {cam}: {s['count']} images")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each depth file
    for fname in tqdm(depth_files, desc="Compensating"):
        image_id = int(fname.replace('.npy', ''))
        camera = get_camera_from_id(image_id, id_to_camera)

        if camera not in SCALE_FACTORS:
            print(f"Warning: Unknown camera for {fname}, skipping")
            continue

        # Load depth
        depth = np.load(os.path.join(input_dir, fname))

        # Apply compensation
        scale = SCALE_FACTORS[camera]
        depth_compensated = depth * scale

        # Save
        output_path = os.path.join(output_dir, fname)
        np.save(output_path, depth_compensated)

        stats[camera]['count'] += 1

    print("\nCompensation complete!")
    print(f"Output directory: {output_dir}")
    print("\nProcessed:")
    for cam, s in stats.items():
        if s['count'] > 0:
            print(f"  {cam}: {s['count']} images (scale={SCALE_FACTORS.get(cam, 'N/A')})")


def main():
    parser = argparse.ArgumentParser(description="Apply scale compensation to MetricAnything depth maps")
    parser.add_argument('--input-dir', type=str,
                        default='/data/ZhaoX/OVM3D-Det/pseudo_label/SUNRGBD/train/depth',
                        help='Input directory with original depth maps')
    parser.add_argument('--output-dir', type=str,
                        default='/data/ZhaoX/OVM3D-Det/pseudo_label_compensated/SUNRGBD/train/depth',
                        help='Output directory for compensated depth maps')
    parser.add_argument('--json-path', type=str,
                        default='/data/ZhaoX/OVM3D-Det/datasets/Omni3D_pl-1/SUNRGBD_train.json',
                        help='Path to Omni3D JSON file')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only analyze without saving')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'val'],
                        help='Dataset mode (train or val)')

    args = parser.parse_args()

    # Update paths based on mode
    if args.mode == 'val':
        args.input_dir = args.input_dir.replace('/train/', '/val/')
        args.output_dir = args.output_dir.replace('/train/', '/val/')
        args.json_path = args.json_path.replace('_train.json', '_val.json')

    print("=" * 60)
    print("MetricAnything Depth Scale Compensation")
    print("=" * 60)
    print(f"\nScale factors:")
    for cam, factor in SCALE_FACTORS.items():
        print(f"  {cam}: {factor:.4f}")
    print()

    apply_compensation(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        json_path=args.json_path,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
