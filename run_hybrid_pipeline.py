#!/usr/bin/env python3
"""
Two-stage hybrid pipeline for 3D bounding box generation:
Stage 1: PCA (fast) - process ALL objects
Stage 2: MASt3R - only re-process "complex" objects that need refinement
"""
import os
import sys
import torch
import numpy as np
from tqdm import tqdm

def estimate_bbox_pca(in_pc, prior, ground_equ=None):
    """Fast PCA-based yaw estimation"""
    if in_pc.shape[0] > 500:
        rand_ind = np.random.randint(0, in_pc.shape[0], 500)
        in_pc = in_pc[rand_ind]
    
    w, h, l = prior
    if ground_equ is not None:
        from cubercnn.generate_label.process_indoor import (
            rotation_matrix_from_vectors, point_to_plane_distance
        )
        dot_product = np.dot([0, -1, 0], ground_equ[:3])
        if dot_product <= 0:
            ground_equ = -ground_equ
        new_ground_equ = np.array([0, -1, 0, point_to_plane_distance(ground_equ, 0, 0, 0)])
        rotation_matrix = rotation_matrix_from_vectors([0, -1, 0], ground_equ[:3])
    else:
        rotation_matrix = np.eye(3)
    
    # PCA for yaw
    in_pc_centered = in_pc - in_pc.mean(axis=0)
    cov = np.cov(in_pc_centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # Main direction
    main_dir = eigenvectors[:, 0]
    angle = np.arctan2(main_dir[0], main_dir[2])
    
    # Create rotation matrix aligned with ground
    Rz = np.array([[np.cos(angle), 0, np.sin(angle)],
                   [0, 1, 0],
                   [-np.sin(angle), 0, np.cos(angle)]])
    R = rotation_matrix @ Rz
    
    # Dimensions
    pc_max = in_pc.max(axis=0)
    pc_min = in_pc.min(axis=0)
    center = (pc_max + pc_min) / 2
    
    return center, np.array([w, h, l]), R


def is_complex_object(in_pc, threshold=0.85):
    """
    Determine if an object needs MASt3R refinement based on point cloud geometry.
    Uses PCA eigenvalues to measure planarity vs complexity.
    
    Returns True if object is complex (needs MASt3R)
    Returns False if object is simple/planar (PCA is sufficient)
    """
    if in_pc.shape[0] < 50:
        return False  # Too small to analyze
    
    # Subsample if needed
    if in_pc.shape[0] > 500:
        rand_ind = np.random.randint(0, in_pc.shape[0], 500)
        in_pc = in_pc[rand_ind]
    
    # PCA analysis
    in_pc_centered = in_pc - in_pc.mean(axis=0)
    try:
        cov = np.cov(in_pc_centered.T)
        eigenvalues, _ = np.linalg.eig(cov)
        eigenvalues = np.abs(eigenvalues)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        # Normalize
        total = eigenvalues.sum()
        if total < 1e-10:
            return False
        ratios = eigenvalues / total
        
        # Planar object: one eigenvalue is much larger than others
        # e.g., wall, floor, ceiling: λ1 >> λ2 ≈ λ3
        planarity_ratio = ratios[1] / ratios[0] if ratios[0] > 0 else 1
        
        # If λ2/λ1 < 0.1, it's very planar (like a wall)
        # If λ2/λ1 > 0.3, it's more complex (like furniture)
        is_planar = planarity_ratio < 0.1
        
        return not is_planar
        
    except:
        return False


def process_stage1_pca(cfg):
    """
    Stage 1: Fast PCA processing for ALL objects
    This should take ~1 hour for 4639 images
    """
    from detectron2.data import build_detection_test_loader
    from cubercnn import generate_label
    from cubercnn.generate_label.process_indoor import llm_generated_prior, process_instances, adaptive_erode_mask, process_ground
    import cv2
    
    print("="*60)
    print("STAGE 1: PCA Processing (Fast)")
    print("="*60)
    
    dataset_names = cfg.DATASETS.TRAIN
    input_folder = 'pseudo_label/SUNRGBD/train'
    image_folder = 'datasets/SUNRGBD'
    
    for dataset_name in dataset_names:
        dataset, mode = dataset_name.split('_')
        output_folder = os.path.join(cfg.OUTPUT_DIR, dataset, mode)
        
        data_loader = build_detection_test_loader(cfg, dataset_name)
        cat_prior = llm_generated_prior[dataset]
        
        info = torch.load(os.path.join(input_folder, "info.pth"))
        info_ground = torch.load(os.path.join(input_folder, "info_ground.pth"))
        
        print(f">> Processing {len(data_loader.dataset._dataset)} images with PCA")
        
        for idx in tqdm(range(len(data_loader.dataset._dataset))):
            im_id = data_loader.dataset._dataset[idx]["image_id"]
            if im_id not in info or not info[im_id]:
                continue
            
            depth = np.load(f"{input_folder}/depth/{im_id}.npy")
            mask = np.load(f"{input_folder}/mask/{im_id}.npy")
            mask = adaptive_erode_mask(mask.astype(float), 12, 2, 6, 2)
            K = data_loader.dataset._dataset[idx]["K"]
            
            has_ground, ground_equ = process_ground(info_ground, im_id, depth, input_folder, K)
            
            # Process with PCA (no MASt3R/TRELLIS)
            boxes3d, center_cam_list, dimension_list, R_cam_list, bbox2d_tight_list = process_instances(
                mask, depth, K, info[im_id], cat_prior, has_ground, ground_equ,
                image=None, yaw_method="pca", im_id=im_id,
                trellis_cache_dir=None, mast3r_model=None)
            
            info[im_id].update({"boxes3d": boxes3d, "center_cam": center_cam_list, 
                              "dimensions": dimension_list, "R_cam": R_cam_list, 
                              "bbox2d_tight": bbox2d_tight_list})
        
        # Save result
        torch.save(info, os.path.join(input_folder, "info_3d_pca.pth"))
        print(f">> Stage 1 complete! Saved to {input_folder}/info_3d_pca.pth")
    
    return True


def process_stage2_mast3r(cfg, complex_only=True, reuse_pca_results=True):
    """
    Stage 2: MASt3R for COMPLEX objects only
    Skips simple planar objects (walls, floors, etc.)
    """
    from detectron2.data import build_detection_test_loader
    from cubercnn import generate_label
    from cubercnn.generate_label.process_indoor import llm_generated_prior, process_instances, adaptive_erode_mask, process_ground
    import cv2
    
    print("="*60)
    print("STAGE 2: MASt3R for Complex Objects Only")
    print("="*60)
    
    input_folder = 'pseudo_label/SUNRGBD/train'
    image_folder = 'datasets/SUNRGBD'
    dataset_names = cfg.DATASETS.TRAIN
    
    # Load PCA results if available
    pca_info = None
    if reuse_pca_results and os.path.exists(os.path.join(input_folder, "info_3d_pca.pth")):
        pca_info = torch.load(os.path.join(input_folder, "info_3d_pca.pth"))
        print(">> Loaded PCA results, will reuse for simple objects")
    
    for dataset_name in dataset_names:
        dataset, mode = dataset_name.split('_')
        output_folder = os.path.join(cfg.OUTPUT_DIR, dataset, mode)
        
        data_loader = build_detection_test_loader(cfg, dataset_name)
        cat_prior = llm_generated_prior[dataset]
        
        info = torch.load(os.path.join(input_folder, "info.pth"))
        info_ground = torch.load(os.path.join(input_folder, "info_ground.pth"))
        
        # Setup MASt3R
        from cubercnn.generate_label.process_indoor import load_mast3r_model, set_trellis_cache_dir
        trellis_cache_dir = os.path.join(output_folder, "trellis_models")
        set_trellis_cache_dir(trellis_cache_dir)
        mast3r_model = load_mast3r_model()
        
        complex_count = 0
        simple_count = 0
        total_objects = 0
        
        print(f">> Processing images, MASt3R for complex objects only...")
        
        for idx in tqdm(range(len(data_loader.dataset._dataset))):
            im_id = data_loader.dataset._dataset[idx]["image_id"]
            if im_id not in info or not info[im_id]:
                continue
            
            depth = np.load(f"{input_folder}/depth/{im_id}.npy")
            mask = np.load(f"{input_folder}/mask/{im_id}.npy")
            mask = adaptive_erode_mask(mask.astype(float), 12, 2, 6, 2)
            K = data_loader.dataset._dataset[idx]["K"]
            
            # Load image for MASt3R
            file_name = data_loader.dataset._dataset[idx].get("file_name", "")
            image = None
            if file_name and os.path.exists(file_name):
                image = cv2.imread(file_name)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            has_ground, ground_equ = process_ground(info_ground, im_id, depth, input_folder, K)
            
            # Process each instance
            boxes3d_list = []
            center_cam_list = []
            dimension_list = []
            R_cam_list = []
            bbox2d_tight_list = []
            
            for inst_idx, inst_info in enumerate(info[im_id]):
                total_objects += 1
                cat_id = inst_info.get('category_id', 0)
                bbox = inst_info.get('bbox', [0, 0, 100, 100])
                
                # Extract point cloud for this instance
                mask_inst = (mask == inst_idx + 1).astype(float)
                depth_masked = depth * mask_inst
                valid = mask_inst > 0
                
                if valid.sum() < 10:
                    boxes3d_list.append([])
                    continue
                
                y_coords = depth_masked[valid]
                x_coords = np.arange(depth.shape[1])[None, :].repeat(depth.shape[0], 0)[valid]
                z_coords = np.arange(depth.shape[0])[:, None].repeat(depth.shape[1], 1)[valid]
                
                pts_cam = np.stack([x_coords * y_coords / K[0, 0], y_coords, -z_coords * y_coords / K[1, 1]], axis=-1)
                in_pc = pts_cam.reshape(-1, 3)
                
                prior = cat_prior.get(cat_id, [0.5, 0.5, 0.5])
                
                # Check if complex
                if complex_only and not is_complex_object(in_pc):
                    simple_count += 1
                    # Use PCA result if available
                    if pca_info and im_id in pca_info and inst_idx < len(pca_info[im_id]):
                        pca_inst = pca_info[im_id][inst_idx]
                        boxes3d_list.append(pca_inst.get('boxes3d', []))
                        center_cam_list.extend(pca_inst.get('center_cam', []))
                        dimension_list.extend(pca_inst.get('dimensions', []))
                        R_cam_list.extend(pca_inst.get('R_cam', []))
                        bbox2d_tight_list.extend(pca_inst.get('bbox2d_tight', []))
                    else:
                        # Quick PCA fallback
                        center, dims, R = estimate_bbox_pca(in_pc, prior, ground_equ)
                        center_cam_list.append(center.tolist())
                        dimension_list.append(dims.tolist())
                        R_cam_list.append(R.tolist())
                        boxes3d_list.append({'center': center, 'dimensions': dims, 'R': R})
                else:
                    complex_count += 1
                    # Use MASt3R for complex objects
                    boxes3d, centers, dims, Rs, bboxes = process_instances(
                        mask_inst.astype(np.uint8), depth, K, [inst_info], cat_prior, 
                        has_ground, ground_equ, image=image, yaw_method="mast3r", 
                        im_id=im_id, trellis_cache_dir=trellis_cache_dir, mast3r_model=mast3r_model)
                    boxes3d_list.extend(boxes3d)
                    center_cam_list.extend(centers)
                    dimension_list.extend(dims)
                    R_cam_list.extend(Rs)
                    bbox2d_tight_list.extend(bboxes)
            
            info[im_id].update({"boxes3d": boxes3d_list, "center_cam": center_cam_list,
                              "dimensions": dimension_list, "R_cam": R_cam_list,
                              "bbox2d_tight": bbox2d_tight_list})
        
        # Save final result
        torch.save(info, os.path.join(input_folder, "info_3d.pth"))
        print(f">> Stage 2 complete!")
        print(f">> Complex objects (MASt3R): {complex_count}")
        print(f">> Simple objects (PCA): {simple_count}")
        print(f">> Total objects: {total_objects}")
        print(f">> MASt3R saved to {input_folder}/info_3d.pth")
        
        # Cleanup
        del mast3r_model
        import gc
        gc.collect()
        torch.cuda.empty_cache()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', type=str, required=True)
    parser.add_argument('--stage', type=int, choices=[1, 2, 3], default=3,
                       help='Stage 1: PCA only, Stage 2: MASt3R complex only, Stage 3: Both')
    parser.add_argument('--output-dir', type=str, default='output/generate_pseudo_label')
    parser.add_argument('--reuse-pca', action='store_true', default=True,
                       help='Reuse PCA results for simple objects in stage 2')
    args = parser.parse_args()
    
    from detectron2.config import get_cfg
    
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.OUTPUT_DIR = args.output_dir
    
    if args.stage == 1:
        process_stage1_pca(cfg)
    elif args.stage == 2:
        process_stage2_mast3r(cfg, complex_only=True, reuse_pca_results=args.reuse_pca)
    else:  # Stage 3 = both
        process_stage1_pca(cfg)
        print("\n" + "="*60)
        print("Stage 1 complete! Starting Stage 2...")
        print("="*60 + "\n")
        process_stage2_mast3r(cfg, complex_only=True, reuse_pca_results=True)


if __name__ == '__main__':
    main()
