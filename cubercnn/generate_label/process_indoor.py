# Copyright (c) Meta Platforms, Inc. and affiliates
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from cubercnn import util
import math
from cubercnn.generate_label.util import *
from cubercnn.generate_label.raytrace import calc_dis_ray_tracing, calc_inside_ratio
from cubercnn.generate_label.mast3r_yaw import (
    estimate_yaw_mast3r_complete,
    estimate_yaw_hybrid,
    load_mast3r_model,
    unload_trellis_model
)
from cubercnn.generate_label.labelany3d_pipeline import (
    estimate_yaw_labelany3d,
    set_trellis_cache_dir
)
from tqdm import tqdm
from scipy.spatial import ConvexHull


def is_complex_object(in_pc, category_name=None, complexity_threshold=0.05):
    """
    Determine if an object needs MASt3R+TRELLIS (complex) vs PCA (simple).

    Multi-factor decision with expert knowledge:
    1. Category-based hard rules (SUNRGBD 38-class)
    2. Point cloud quality assessment
    3. Geometric regularity analysis
    4. Shape-based confidence estimation

    Returns True for complex objects needing visual matching (MASt3R+TRELLIS+PnP).
    """
    # SUNRGBD 38-class categorization with expert knowledge
    # Always use MASt3R: visual features dominate orientation
    ALWAYS_MAST3R = {
        # Planar/architectural elements - XZ projection highly ambiguous
        'window', 'door', 'blinds', 'curtain',
        # Visual feature dependent - texture/pattern/frame defines orientation
        'picture', 'mirror',
        # Appliances with functional parts that define front/back
        'refrigerator', 'stove', 'oven',
        # Electronics - screen/panel defines orientation
        'television',
    }

    # Always use PCA: clear geometric structure dominates
    ALWAYS_PCA = {
        # Planar elements that are mostly regular
        'floor mat', 'machine',
        # Seating - typically rectangular with clear main axis
        'chair', 'sofa',
        # Tables - rectangular/circular
        'table', 'desk', 'night stand',
        # Beds - rectangular with clear head direction
        'bed',
        # Storage - box-like shapes
        'box', 'bin', 'bookcase', 'shelves',
        # Bathroom fixtures - regular shapes
        'sink', 'bathtub', 'toilet', 'counter',
        # Small objects - similar difficulty for both methods
        'bottle', 'cup', 'lamp', 'pillow', 'towel', 'books',
        'shoes', 'clothes', 'stationery', 'bicycle',
        # Electronics with clear geometric shape
        'laptop',
    }

    # Category-dependent: decision based on geometry
    GEOMETRY_DEPENDENT = {
        'cabinet',  # Can be box-like OR flat panel depending on view
    }

    # ========================================================================
    # Helper: Compute geometric properties
    # ========================================================================
    def compute_geometric_properties(pc):
        """Compute comprehensive geometric properties for decision making."""
        try:
            # Subsample for efficiency
            if pc.shape[0] > 500:
                rand_ind = np.random.randint(0, pc.shape[0], 500)
                pc = pc[rand_ind]

            # Center and compute covariance
            pc_centered = pc - pc.mean(axis=0)
            cov = np.cov(pc_centered.T)

            if np.linalg.det(cov) < 1e-10:
                return None

            eigenvalues, _ = np.linalg.eig(cov)
            eigenvalues = np.abs(np.sort(eigenvalues)[::-1])  # λ1 >= λ2 >= λ3
            total = eigenvalues.sum()
            if total < 1e-10:
                return None

            # Normalized ratios
            r = eigenvalues / total  # r1 + r2 + r3 = 1

            # Key metrics
            flatness = r[2] / (r[0] + 1e-10)  # λ3/λ1
            elongation = (r[0] - r[1]) / (r[0] + 1e-10)  # (λ1-λ2)/λ1
            second_ratio = r[1] / (r[0] + 1e-10)  # λ2/λ1

            return {
                'eigenvalues': eigenvalues,
                'ratios': r,
                'flatness': flatness,
                'elongation': elongation,
                'second_ratio': second_ratio,
                'point_count': pc.shape[0],
            }
        except Exception:
            return None

    def compute_regularity_score(pc):
        """
        Compute regularity score based on convex hull vs bounding box.
        - High score (~1.0): rectangular, regular shape → PCA reliable
        - Low score (~0.5): L-shape, irregular → MASt3R helpful
        """
        try:
            # Use XZ plane for yaw estimation
            xz = pc[:, [0, 2]] if pc.shape[1] >= 3 else pc[:, :2]

            if xz.shape[0] < 10:
                return None

            # Convex hull area
            hull = ConvexHull(xz)
            hull_area = hull.volume  # ConvexHull uses 'volume' for 2D area

            # Axis-aligned bounding box
            x_min, x_max = xz[:, 0].min(), xz[:, 0].max()
            z_min, z_max = xz[:, 1].min(), xz[:, 1].max()
            bbox_area = (x_max - x_min) * (z_max - z_min)

            if bbox_area < 1e-10:
                return None

            regularity = hull_area / bbox_area
            return regularity
        except Exception:
            return None

    def estimate_occlusion_level(pc):
        """
        Estimate occlusion level based on point count and distribution.
        Returns: 'low', 'medium', 'high'
        """
        n_points = pc.shape[0]

        # Also check spread in Y (height) dimension
        if pc.shape[1] >= 3:
            y_range = pc[:, 1].max() - pc[:, 1].min()
            expected_height = 0.5  # Expected minimum height for most objects
            height_ratio = y_range / (expected_height + 1e-10)
        else:
            height_ratio = 1.0

        if n_points < 50:
            return 'high'
        elif n_points < 150 or height_ratio < 0.2:
            return 'medium'
        else:
            return 'low'

    # ========================================================================
    # Decision Logic
    # ========================================================================

    # Step 1: Category-based hard rules
    if category_name is not None:
        cat = category_name.lower().strip()

        if cat in ALWAYS_MAST3R:
            return True  # Visual features critical for orientation

        if cat in ALWAYS_PCA:
            geom = compute_geometric_properties(in_pc)
            occ = estimate_occlusion_level(in_pc)

            # Heavily occluded → MASt3R helps
            if occ == 'high':
                return True

            # Very flat → XZ projection unreliable
            if geom is not None and geom['flatness'] < 0.02:
                return True

            # Otherwise PCA is reliable
            return False

        if cat in GEOMETRY_DEPENDENT:
            # Cabinet: depends on geometry
            geom = compute_geometric_properties(in_pc)
            if geom is not None:
                regularity = compute_regularity_score(in_pc)
                if regularity is not None and regularity > 0.75:
                    return False  # Regular box shape → PCA
            return True  # Default to MASt3R for ambiguous cases

    # Step 2: Unknown category or no category → geometric analysis
    geom = compute_geometric_properties(in_pc)
    occ = estimate_occlusion_level(in_pc)

    # High occlusion → prefer visual matching
    if occ == 'high':
        return True

    if geom is None:
        return True  # Can't analyze → use MASt3R

    # Step 3: Geometric regularity assessment
    regularity = compute_regularity_score(in_pc)

    # Very thin objects (paper-like, plank-like)
    if geom['flatness'] < 0.015:
        return True

    # Very elongated non-flat objects → reliable PCA
    if geom['elongation'] > 0.75 and geom['flatness'] > 0.05:
        return False

    # Regular shape (high convex hull ratio) → PCA reliable
    if regularity is not None and regularity > 0.8:
        return False

    # Irregular shape → MASt3R helpful
    if regularity is not None and regularity < 0.65:
        return True

    # Medium complexity: check elongation ratio
    # Very low second_ratio (λ2 << λ1) = clear axis → PCA
    if geom['second_ratio'] < 0.15 and geom['flatness'] > 0.03:
        return False

    # Otherwise default to MASt3R for safety
    return True


def lshape_mabr_yaw(rotated_pc_xz):
    """MABR (Minimum Area Bounding Rectangle) L-Shape Fitting."""
    if rotated_pc_xz.shape[0] < 10:
        return 0.0, float("inf")
    rng = np.random.default_rng(42)
    sample = rotated_pc_xz if rotated_pc_xz.shape[0] <= 2000 else rotated_pc_xz[rng.choice(rotated_pc_xz.shape[0], size=2000, replace=False)]
    center = sample.mean(axis=0)
    dists = np.linalg.norm(sample - center, axis=1)
    if len(dists) > 20:
        threshold = np.mean(dists) + 3 * np.std(dists)
        sample = sample[dists <= threshold]
    try:
        hull = ConvexHull(sample)
        hull_pts = sample[hull.vertices]
    except Exception:
        hull_pts = sample
    min_area = float("inf")
    best_yaw = 0.0
    n_edges = len(hull_pts)
    for i in range(n_edges):
        p1, p2 = hull_pts[i], hull_pts[(i + 1) % n_edges]
        for angle in [np.arctan2(p2[1]-p1[1], p2[0]-p1[0]), np.arctan2(p2[1]-p1[1], p2[0]-p1[0]) + np.pi/2]:
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, -s], [s, c]])
            rotated = hull_pts @ R.T
            x_min, x_max = rotated[:, 0].min(), rotated[:, 0].max()
            y_min, y_max = rotated[:, 1].min(), rotated[:, 1].max()
            area = (x_max - x_min) * (y_max - y_min)
            if area < min_area:
                min_area, best_yaw = area, angle
    for angle in np.linspace(best_yaw - np.pi/4, best_yaw + np.pi/4, 180):
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        rotated = hull_pts @ R.T
        area = (rotated[:,0].max()-rotated[:,0].min()) * (rotated[:,1].max()-rotated[:,1].min())
        if area < min_area:
            min_area, best_yaw = area, angle
    return best_yaw, min_area


def create_uv_depth(depth, mask=None):
    if mask is not None:
        depth = depth * mask
    x, y = np.meshgrid(np.linspace(0, depth.shape[1] - 1, depth.shape[1]), np.linspace(0, depth.shape[0] - 1, depth.shape[0]))
    uv_depth = np.stack((x, y, depth), axis=-1).reshape(-1, 3)
    return uv_depth[uv_depth[:, 2] != 0]


def process_ground(info_ground, im_id, depth, input_folder, K):
    if im_id not in info_ground or not info_ground[im_id]:
        return False, None
    ground_mask = np.load(f"{input_folder}/ground_mask/{im_id}.npy")
    ground_mask = erode_mask(ground_mask.astype(float), 4, 4)
    ground_mask = ground_mask[np.argmax(info_ground[im_id]["conf"])]
    ground_depth = depth * ground_mask.squeeze()
    uv_depth = create_uv_depth(ground_depth)
    pseudo_lidar_ground = project_image_to_cam(uv_depth, np.array(K))
    if pseudo_lidar_ground.shape[0] > 10:
        ground_equ = extract_ground(pseudo_lidar_ground)
        return True, ground_equ
    return False, None


def compute_2d_bbox_from_mask(mask):
    """Compute tight axis-aligned 2D bbox from binary mask."""
    if mask.sum() < 10:
        return [-1, -1, -1, -1]
    indices = np.argwhere(mask > 0)
    if len(indices) == 0:
        return [-1, -1, -1, -1]
    min_row, min_col = indices.min(axis=0)
    max_row, max_col = indices.max(axis=0)
    x1, y1 = float(min_col), float(min_row)
    x2, y2 = float(max_col), float(max_row)
    return [x1, y1, x2, y2]


def process_instances(mask_instance, depth, K, info_i, cat_prior, has_ground, ground_equ, image=None, yaw_method="pca", im_id=None, trellis_cache_dir=None, mast3r_model=None):
    boxes3d, center_cam_list, dimension_list, R_cam_list, bbox2d_tight_list = [], [], [], [], []
    complex_count, simple_count = 0, 0

    for mask_ind, cur_mask in enumerate(mask_instance):
        cur_mask_squeeze = cur_mask.squeeze(0)
        bbox2d_tight = compute_2d_bbox_from_mask(cur_mask_squeeze)
        bbox2d_tight_list.append(bbox2d_tight)
        if cur_mask.sum() < 10:
            boxes3d.append(np.full((8, 3), -1))
            center_cam_list.append(-1 * np.ones(3))
            dimension_list.append([-1, -1, -1])
            R_cam_list.append(-1 * np.ones((3, 3)))
            continue
        cur_depth = depth * cur_mask.squeeze(0)
        uv_depth = create_uv_depth(cur_depth)
        pseudo_lidar = project_image_to_cam(uv_depth, np.array(K))
        category_name = info_i["phrases"][mask_ind]
        prior = np.array(cat_prior[category_name])

        # Generate unique obj_id for TRELLIS caching
        obj_id = f"{im_id}_{mask_ind}" if im_id is not None else None

        # For pca_trellis_hybrid mode, pre-compute complexity
        pca_trellis_mode = (yaw_method == "pca_trellis_hybrid")
        is_complex = False
        if pca_trellis_mode:
            is_complex = is_complex_object(pseudo_lidar, category_name=category_name, complexity_threshold=0.05)
            if is_complex:
                complex_count += 1
            else:
                simple_count += 1

        bbox_params = estimate_bbox(pseudo_lidar, prior, ground_equ if has_ground else None,
                                   image=image, bbox2d=bbox2d_tight, K=K,
                                   yaw_method=yaw_method, obj_id=obj_id,
                                   cache_dir=trellis_cache_dir, mast3r_model=mast3r_model)
        boxes3d.extend(bbox_params[0])
        center_cam_list.extend(bbox_params[1])
        dimension_list.extend(bbox_params[2])
        R_cam_list.extend(bbox_params[3])

    # Print complexity stats for first few images
    if pca_trellis_mode and im_id is not None and simple_count + complex_count > 0:
        print(f">> [{im_id}] Complex: {complex_count}, Simple: {simple_count}")

    return boxes3d, center_cam_list, dimension_list, R_cam_list, bbox2d_tight_list


def process_indoor(dataset, cat_prior, input_folder, output_folder, yaw_method="pca", image_folder=None, cache_dir=None, start_idx=0, end_idx=None):
    info = torch.load(os.path.join(input_folder, "info.pth"))
    info_ground = torch.load(os.path.join(input_folder, "info_ground.pth"))

    # Support for parallel processing: filter by index range
    total_images = len(dataset._dataset)
    if end_idx is None:
        end_idx = total_images
    start_idx = max(0, start_idx)
    end_idx = min(end_idx, total_images)

    # Setup TRELLIS cache directory if using TRELLIS-based yaw estimation
    trellis_cache_dir = None
    use_trellis = yaw_method in ["mast3r_trellis", "mast3r_trellis_hybrid", "labelany3d", "pca_trellis_hybrid"]

    if use_trellis:
        if cache_dir:
            trellis_cache_dir = os.path.join(cache_dir, "trellis_models")
        else:
            trellis_cache_dir = os.path.join(output_folder, "trellis_models")
        set_trellis_cache_dir(trellis_cache_dir)
        print(f">> TRELLIS cache directory: {trellis_cache_dir}")

    # Pre-load MASt3R model for memory-efficient sequential processing
    mast3r_model = None
    if yaw_method in ["mast3r", "mast3r_hybrid", "mast3r_trellis", "mast3r_trellis_hybrid", "labelany3d", "pca_trellis_hybrid"]:
        mast3r_model = load_mast3r_model()
        if mast3r_model is None:
            print(">> WARNING: MASt3R not available, will fallback to L-Shape")

    print(f">> Starting process_indoor for {len(dataset._dataset)} images ({start_idx} to {end_idx}) with method: {yaw_method}")

    # Track progress for periodic saving
    processed_count = 0
    save_interval = 100  # Save every 100 images

    for idx in tqdm(range(start_idx, end_idx)):
        if idx == 0:
            print(f">> DEBUG: Starting first iteration...")
        im_id = dataset._dataset[idx]["image_id"]
        # DEBUG: print actual dataset entry structure for first few items
        if idx < 2:
            print(f">> DEBUG: dataset._dataset[{idx}] keys: {list(dataset._dataset[idx].keys())}")
            print(f">> DEBUG: dataset._dataset[{idx}] content: {{'id': {dataset._dataset[idx].get('id', 'N/A')}, 'file_path': '{dataset._dataset[idx].get('file_path', 'N/A')}', 'file_name': '{dataset._dataset[idx].get('file_name', 'N/A')}'}}")
        if im_id not in info:
            print(f">> DEBUG: im_id {im_id} not in info keys")
            continue
        if not info[im_id]:
            print(f">> DEBUG: im_id {im_id} has empty info")
            continue
        depth = np.load(f"{input_folder}/depth/{im_id}.npy")
        mask = np.load(f"{input_folder}/mask/{im_id}.npy")
        mask = adaptive_erode_mask(mask.astype(float), 12, 2, 6, 2)
        K = dataset._dataset[idx]["K"]

        # Load image if needed for MASt3R/TRELLIS
        image = None
        if yaw_method in ["mast3r", "mast3r_hybrid", "mast3r_trellis", "mast3r_trellis_hybrid", "labelany3d", "pca_trellis_hybrid"] and image_folder is not None:
            # Get file_name from dataset (contains path like "datasets/SUNRGBD/kv2/...")
            file_name = dataset._dataset[idx].get("file_name", "")
            if file_name:
                # file_name starts with "datasets/SUNRGBD/" - use it directly as the full path
                # since image_folder is also "datasets/SUNRGBD"
                full_path = file_name  # file_name already contains the full relative path
                if os.path.exists(full_path):
                    image = cv2.imread(full_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    if idx < 3:  # Only print for first few images
                        print(f">> DEBUG: file_name not found: {full_path}")
                        print(f">> DEBUG: image_folder = {image_folder}")
                        print(f">> DEBUG: original file_name = {file_name}")
            # Fallback: try jpg/png with image_id
            if image is None:
                img_path = os.path.join(image_folder, f"{im_id}.jpg")
                if os.path.exists(img_path):
                    image = cv2.imread(img_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    img_path = os.path.join(image_folder, f"{im_id}.png")
                    if os.path.exists(img_path):
                        image = cv2.imread(img_path)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    else:
                        if idx < 3:  # Only print for first few images
                            print(f">> DEBUG: image not found at {img_path}")
                            print(f">> DEBUG: image_folder = {image_folder}")
                            print(f">> DEBUG: file_name = {file_name}")

        has_ground, ground_equ = process_ground(info_ground, im_id, depth, input_folder, K)

        # Generate unique obj_id for TRELLIS caching
        obj_id = f"{im_id}" if use_trellis else None

        print(f">> DEBUG: Processing {len([i for i in info[im_id].keys() if i.startswith('box')])} instances for {im_id}")
        boxes3d, center_cam_list, dimension_list, R_cam_list, bbox2d_tight_list = process_instances(
            mask, depth, K, info[im_id], cat_prior, has_ground, ground_equ,
            image=image, yaw_method=yaw_method, im_id=im_id,
            trellis_cache_dir=trellis_cache_dir, mast3r_model=mast3r_model)
        info[im_id].update({"boxes3d": boxes3d, "center_cam": center_cam_list, "dimensions": dimension_list, "R_cam": R_cam_list, "bbox2d_tight": bbox2d_tight_list})

        # Periodic saving for crash recovery
        processed_count += 1
        if processed_count % save_interval == 0:
            torch.save(info, os.path.join(input_folder, "info_3d.pth"))
            print(f">> Checkpoint saved: {processed_count} images processed")

    # Cleanup: unload models to free GPU memory
    if mast3r_model is not None:
        del mast3r_model
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print(">> MASt3R model unloaded, GPU memory freed")

    torch.save(info, os.path.join(input_folder, "info_3d.pth"))


def estimate_bbox(in_pc, prior, ground_equ=None, image=None, bbox2d=None, K=None, yaw_method="pca", obj_id=None, cache_dir=None, mast3r_model=None):
    if in_pc.shape[0] > 500:
        rand_ind = np.random.randint(0, in_pc.shape[0], 500)
        in_pc = in_pc[rand_ind]
    w, h, l = prior
    if ground_equ is not None:
        dot_product = np.dot([0, -1, 0], ground_equ[:3])
        if dot_product <= 0:
            ground_equ = -ground_equ
        new_ground_equ = np.array([0, -1, 0, point_to_plane_distance(ground_equ, 0, 0, 0)])
        rotation_matrix = rotation_matrix_from_vectors([0, -1, 0], ground_equ[:3])
    else:
        rotation_matrix = np.eye(3)
        new_ground_equ = None
    rotated_pc = np.dot(in_pc, rotation_matrix)
    xz = rotated_pc[:, [0, 2]]
    hybrid_mode = (yaw_method == "hybrid")
    mast3r_mode = (yaw_method in ["mast3r", "mast3r_hybrid"])
    mast3r_trellis_mode = (yaw_method in ["mast3r_trellis", "mast3r_trellis_hybrid"])
    labelany3d_mode = (yaw_method == "labelany3d")
    pca_trellis_hybrid_mode = (yaw_method == "pca_trellis_hybrid")

    if pca_trellis_hybrid_mode:
        # Smart hybrid: PCA for simple objects, MASt3R+TRELLIS for complex objects
        is_complex = is_complex_object(in_pc, complexity_threshold=0.25)

        if is_complex and image is not None and bbox2d is not None and K is not None:
            # Complex object: use MASt3R+TRELLIS
            skip_step7_8 = os.environ.get("SKIP_STEP7_8", "1") == "1"
            try:
                la3d_yaw, la3d_conf = estimate_yaw_labelany3d(
                    in_pc, image, bbox2d, K, obj_id,
                    cache_dir=cache_dir, device="cuda",
                    skip_step7_8=skip_step7_8
                )
            except Exception as e:
                print(f">> ERROR in estimate_yaw_labelany3d: {e}")
                la3d_yaw, la3d_conf = None, 0.0

            if la3d_yaw is not None and la3d_conf > 0.05:
                yaw = la3d_yaw
            else:
                # Fallback to L-Shape
                pca_local = PCA(2)
                pca_local.fit(xz)
                mabr_yaw, mabr_area = lshape_mabr_yaw(xz)
                yaw = mabr_yaw
        else:
            # Simple object: use fast PCA
            pca_local = PCA(2)
            pca_local.fit(xz)
            pca_yaw = np.arctan2(pca_local.components_[0, 1], pca_local.components_[0, 0])
            mabr_yaw, mabr_area = lshape_mabr_yaw(xz)
            yaw = mabr_yaw

    elif labelany3d_mode:
        # EXACT LabelAny3D pipeline: TRELLIS + MASt3R + Two-stage PnP
        skip_step7_8 = os.environ.get("SKIP_STEP7_8", "1") == "1"
        if image is not None and bbox2d is not None and K is not None:
            try:
                la3d_yaw, la3d_conf = estimate_yaw_labelany3d(
                    in_pc, image, bbox2d, K, obj_id,
                    cache_dir=cache_dir, device="cuda",
                    skip_step7_8=skip_step7_8
                )
            except Exception as e:
                import traceback
                print(f">> ERROR in estimate_yaw_labelany3d: {e}")
                traceback.print_exc()
                la3d_yaw, la3d_conf = None, 0.0
            
            if la3d_yaw is not None and la3d_conf > 0.05:
                yaw = la3d_yaw
                use_labelany3d = True
            else:
                use_labelany3d = False
        else:
            print(f">> WARNING: labelany3d skipped - image={image is not None}, bbox2d={bbox2d is not None}, K={K is not None}")
            use_labelany3d = False

        if not use_labelany3d:
            # Fallback to L-Shape
            pca_local = PCA(2)
            pca_local.fit(xz)
            pca_yaw = np.arctan2(pca_local.components_[0, 1], pca_local.components_[0, 0])
            mabr_yaw, mabr_area = lshape_mabr_yaw(xz)
            yaw = mabr_yaw
    elif mast3r_trellis_mode:
        # MASt3R + TRELLIS-based yaw estimation (best quality)
        if image is not None and bbox2d is not None and K is not None:
            mast3r_yaw, mast3r_conf = estimate_yaw_mast3r_complete(
                in_pc, image, bbox2d, K, prior,
                use_trellis=True, obj_id=obj_id, cache_dir=cache_dir,
                mast3r_model=mast3r_model
            )
            if mast3r_yaw is not None and mast3r_conf > 0.1:
                yaw = mast3r_yaw
                use_mast3r = True
            else:
                use_mast3r = False
        else:
            use_mast3r = False

        if not use_mast3r or yaw_method == "mast3r_trellis_hybrid":
            # Also compute L-Shape for hybrid or fallback
            pca_local = PCA(2)
            pca_local.fit(xz)
            pca_yaw = np.arctan2(pca_local.components_[0, 1], pca_local.components_[0, 0])
            mabr_yaw, mabr_area = lshape_mabr_yaw(xz)

            if yaw_method == "mast3r_trellis_hybrid" and use_mast3r and mast3r_conf > 0.1:
                # Fuse MASt3R with L-Shape
                yaw = mast3r_conf * mast3r_yaw + (1 - mast3r_conf) * mabr_yaw
            else:
                # Fallback to L-Shape
                yaw = mabr_yaw
    elif mast3r_mode:
        # MASt3R-based yaw estimation (with pointcloud rendering fallback)
        if image is not None and bbox2d is not None and K is not None:
            mast3r_yaw, mast3r_conf = estimate_yaw_mast3r_complete(
                in_pc, image, bbox2d, K, prior,
                use_trellis=False,  # Use pointcloud rendering
                mast3r_model=mast3r_model
            )
            if mast3r_yaw is not None and mast3r_conf > 0.1:
                yaw = mast3r_yaw
                use_mast3r = True
            else:
                use_mast3r = False
        else:
            use_mast3r = False

        if not use_mast3r or yaw_method == "mast3r_hybrid":
            # Also compute L-Shape for hybrid or fallback
            pca_local = PCA(2)
            pca_local.fit(xz)
            pca_yaw = np.arctan2(pca_local.components_[0, 1], pca_local.components_[0, 0])
            mabr_yaw, mabr_area = lshape_mabr_yaw(xz)

            if yaw_method == "mast3r_hybrid" and use_mast3r and mast3r_conf > 0.1:
                # Fuse MASt3R with L-Shape
                yaw = mast3r_conf * mast3r_yaw + (1 - mast3r_conf) * mabr_yaw
            else:
                # Fallback to L-Shape
                yaw = mabr_yaw
    elif hybrid_mode:
        pca_local = PCA(2)
        pca_local.fit(xz)
        pca_yaw = np.arctan2(pca_local.components_[0, 1], pca_local.components_[0, 0])
        mabr_yaw, mabr_area = lshape_mabr_yaw(xz)
        yaw = pca_yaw
    elif yaw_method == "lshape_mabr":
        yaw, mabr_area = lshape_mabr_yaw(xz)
    else:
        pca = PCA(2)
        pca.fit(xz)
        yaw = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
    rotated_pc_2 = rotate_y(yaw) @ rotated_pc.T
    x_min, x_max = rotated_pc_2[0, :].min(), rotated_pc_2[0, :].max()
    y_min, y_max = rotated_pc_2[1, :].min(), rotated_pc_2[1, :].max()
    z_min, z_max = rotated_pc_2[2, :].min(), rotated_pc_2[2, :].max()
    dx, dy, dz = x_max - x_min, y_max - y_min, z_max - z_min
    cx, cy, cz = (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2
    if new_ground_equ is not None and dy < h * 0.5:
        dy = h
        cdis = point_to_plane_distance(new_ground_equ, cx, cy, cz)
        if cdis - dy / 2 < 0.5:
            cy += cdis - dy / 2
    vertives_list, center_cam_list, dimension_list, R_cam_list = [], [], [], []
    use_measured_dims = (l * 0.5 <= dx and w * 0.5 <= dz) or (l * 0.5 <= dz and w * 0.5 <= dx)
    if use_measured_dims:
        vertives = convert_box_vertices(cx, cy, cz, dx, dy, dz, 0).astype(np.float16)
        vertives = np.dot(rotate_y(-yaw), vertives.T).T
        vertives = np.dot(vertives, rotation_matrix.T)
        vertives_list.append(vertives)
        center_cam_list.append(vertives.mean(0))
        dimension_list.append([dz, dy, dx])
        R_cam_list.append(rotation_matrix @ rotate_y(-yaw))
    else:
        possible_bboxs = generate_possible_bboxs(cx, cz, dx, dz, w, l)
        min_loss = float("inf")
        best_vertives, best_dx_p, best_dz_p, best_yaw_final = None, dx, dz, yaw
        for possible_bbox in possible_bboxs:
            x_min_p, x_max_p, z_min_p, z_max_p = possible_bbox
            dx_p, dz_p = x_max_p - x_min_p, z_max_p - z_min_p
            cx_p, cz_p = (x_min_p + x_max_p) / 2, (z_min_p + z_max_p) / 2
            inside_ratio = calc_inside_ratio(rotated_pc_2, x_min_p, x_max_p, z_min_p, z_max_p)
            pc_tensor = torch.from_numpy(rotated_pc).float()
            yaw_candidates = [(yaw, "default")]
            if hybrid_mode:
                yaw_candidates.append((mabr_yaw, "mabr"))
            best_for_this_bbox, best_verts_this = float("inf"), None
            for cand_yaw, cand_name in yaw_candidates:
                pc_rotated = rotate_y(cand_yaw) @ rotated_pc.T
                vertives_p = convert_box_vertices(cx_p, cy, cz_p, dx_p, dz_p, dy, 0).astype(np.float16)
                vertives_p = np.dot(rotate_y(-cand_yaw), vertives_p.T).T
                new_cx_p, new_cz_p = vertives_p[:, 0].mean(), vertives_p[:, 2].mean()
                loss_ray = calc_dis_ray_tracing(torch.Tensor([dz_p, dx_p]), torch.Tensor([cand_yaw]), pc_tensor[:, [0, 2]], torch.Tensor([new_cx_p, new_cz_p]))
                loss = loss_ray + 5 * (1 - inside_ratio)
                if loss < best_for_this_bbox:
                    best_for_this_bbox, best_verts_this = loss, (vertives_p, cand_yaw)
            if best_for_this_bbox < min_loss:
                min_loss, best_vertives, best_dx_p, best_dz_p, best_yaw_final = best_for_this_bbox, best_verts_this[0], dx_p, dz_p, best_verts_this[1]
        if best_vertives is None:
            vertives = convert_box_vertices(cx, cy, cz, dx, dz, dy, 0).astype(np.float16)
            vertives = np.dot(rotate_y(-yaw), vertives.T).T
            vertives = np.dot(vertives, rotation_matrix.T)
            vertives_list.append(vertives)
            center_cam_list.append(vertives.mean(0))
            dimension_list.append([dz, dy, dx])
            R_cam_list.append(rotation_matrix @ rotate_y(-yaw))
        else:
            best_vertives = np.dot(best_vertives, rotation_matrix.T)
            vertives_list.append(best_vertives)
            center_cam_list.append(best_vertives.mean(0))
            dimension_list.append([best_dz_p, dy, best_dx_p])
            R_cam_list.append(rotation_matrix @ rotate_y(-best_yaw_final))
    return vertives_list, center_cam_list, dimension_list, R_cam_list
