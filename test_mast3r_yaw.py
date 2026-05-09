#!/usr/bin/env python3
"""
Test script for MASt3R yaw estimation integration.

Usage:
    python test_mast3r_yaw.py --dataset SUNRGBD --subset train
"""

import argparse
import os
import sys
import torch
import numpy as np

# Add project to path
sys.path.insert(0, '/data/ZhaoX/OVM3D-Det')

from cubercnn.generate_label.process_indoor import process_indoor
from cubercnn.data.dataset import DatasetFromFile


def main():
    parser = argparse.ArgumentParser(description="Test MASt3R yaw estimation")
    parser.add_argument("--dataset", type=str, default="SUNRGBD", help="Dataset name")
    parser.add_argument("--subset", type=str, default="train", help="train/val")
    parser.add_argument("--yaw_method", type=str, default="mast3r_hybrid", 
                       choices=["pca", "lshape_mabr", "hybrid", "mast3r", "mast3r_hybrid"],
                       help="Yaw estimation method")
    parser.add_argument("--image_folder", type=str, default=None,
                       help="Folder containing RGB images")
    parser.add_argument("--max_images", type=int, default=10,
                       help="Max number of images to process")
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Testing MASt3R yaw estimation")
    print(f"Dataset: {args.dataset}")
    print(f"Subset: {args.subset}")
    print(f"Yaw method: {args.yaw_method}")
    print("=" * 60)
    
    # Set paths
    dataset_name = f"{args.dataset}_{args.subset}"
    base_folder = f"/data/ZhaoX/OVM3D-Det/pseudo_label/{args.dataset}/{args.subset}"
    
    # Default image folder
    if args.image_folder is None:
        if args.dataset == "SUNRGBD":
            # SUNRGBD images are in the Omni3D folder
            args.image_folder = f"/data/ZhaoX/OVM3D-Det/datasets/Omni3D/{args.dataset}/images"
    
    print(f"Input folder: {base_folder}")
    print(f"Image folder: {args.image_folder}")
    
    # Check if MASt3R is available
    try:
        from mast3r.model import AsymmetricMASt3R
        print("MASt3R is available ✓")
        mast3r_available = True
    except ImportError:
        print("WARNING: MASt3R is not installed!")
        print("Install with: git clone https://github.com/naver/mast3r")
        mast3r_available = False
        
        if args.yaw_method.startswith("mast3r"):
            print("Falling back to PCA for this test...")
            args.yaw_method = "pca"
    
    # Load priors
    from cubercnn.generate_label.priors import sunrgbd_priors
    cat_prior = sunrgbd_priors
    
    # Create dummy dataset for testing
    from detectron2.data import DatasetFromDict
    
    # Load dataset metadata
    dataset_dicts = []
    info_path = os.path.join(base_folder, "info.pth")
    if os.path.exists(info_path):
        info = torch.load(info_path)
        for im_id in list(info.keys())[:args.max_images]:
            if info[im_id]:
                dict_entry = {
                    "image_id": im_id,
                    "file_name": f"{im_id}.jpg",
                    "K": np.array([[525.0, 0, 319.5], [0, 525.0, 239.5], [0, 0, 1]])
                }
                dataset_dicts.append(dict_entry)
    
    print(f"Found {len(dataset_dicts)} images to process")
    
    if len(dataset_dicts) == 0:
        print("No images found! Please check paths.")
        return
    
    # Create simple dataset wrapper
    class SimpleDataset:
        def __init__(self, dicts):
            self._dataset = dicts
        
        def __len__(self):
            return len(self._dataset)
    
    dataset = SimpleDataset(dataset_dicts)
    
    # Process
    print(f"\nProcessing with yaw_method='{args.yaw_method}'...")
    
    process_indoor(
        dataset=dataset,
        cat_prior=cat_prior,
        input_folder=base_folder,
        output_folder=base_folder,
        yaw_method=args.yaw_method,
        image_folder=args.image_folder
    )
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
