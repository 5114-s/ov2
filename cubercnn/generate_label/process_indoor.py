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
from tqdm import tqdm
from scipy.spatial import ConvexHull


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


def process_instances(mask_instance, depth, K, info_i, cat_prior, has_ground, ground_equ, yaw_method="pca"):
    boxes3d, center_cam_list, dimension_list, R_cam_list, bbox2d_tight_list = [], [], [], [], []
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
        bbox_params = estimate_bbox(pseudo_lidar, prior, ground_equ if has_ground else None, yaw_method=yaw_method)
        boxes3d.extend(bbox_params[0])
        center_cam_list.extend(bbox_params[1])
        dimension_list.extend(bbox_params[2])
        R_cam_list.extend(bbox_params[3])
    return boxes3d, center_cam_list, dimension_list, R_cam_list, bbox2d_tight_list


def process_indoor(dataset, cat_prior, input_folder, output_folder, yaw_method="pca"):
    info = torch.load(os.path.join(input_folder, "info.pth"))
    info_ground = torch.load(os.path.join(input_folder, "info_ground.pth"))
    for idx in tqdm(range(len(dataset._dataset))):
        im_id = dataset._dataset[idx]["image_id"]
        if im_id not in info or not info[im_id]:
            continue
        depth = np.load(f"{input_folder}/depth/{im_id}.npy")
        mask = np.load(f"{input_folder}/mask/{im_id}.npy")
        mask = adaptive_erode_mask(mask.astype(float), 12, 2, 6, 2)
        K = dataset._dataset[idx]["K"]
        has_ground, ground_equ = process_ground(info_ground, im_id, depth, input_folder, K)
        boxes3d, center_cam_list, dimension_list, R_cam_list, bbox2d_tight_list = process_instances(
            mask, depth, K, info[im_id], cat_prior, has_ground, ground_equ, yaw_method=yaw_method)
        info[im_id].update({"boxes3d": boxes3d, "center_cam": center_cam_list, "dimensions": dimension_list, "R_cam": R_cam_list, "bbox2d_tight": bbox2d_tight_list})
    torch.save(info, os.path.join(input_folder, "info_3d.pth"))


def estimate_bbox(in_pc, prior, ground_equ=None, yaw_method="pca"):
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
    if hybrid_mode:
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
