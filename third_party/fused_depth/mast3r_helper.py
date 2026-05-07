#!/usr/bin/env python3
"""
MASt3R 2D-3D matching for OVM3D-Det
替换 PCA yaw 估计，用 MASt3R + PnP 求解物体姿态

基于 LabelAny3D 论文的 2D-3D matching + PnP 方法
"""

import torch
import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List
import os

# MASt3R model singleton
_mast3r_model = None
_mast3r_device = None


def load_mast3r_model(device: str = "cuda") -> 'AsymmetricMASt3R':
    """
    加载 MASt3R 模型（单例模式）
    
    Args:
        device: 'cuda' or 'cpu'
    
    Returns:
        MASt3R model
    """
    global _mast3r_model, _mast3r_device
    
    if _mast3r_model is not None and str(_mast3r_device) == str(device):
        return _mast3r_model
    
    from mast3r.model import AsymmetricMASt3R
    
    print(f">> Loading MASt3R model on {device}...")
    
    # 加载预训练模型
    # 推荐使用 non-metric 版本用于通用场景
    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_nonmetric"
    
    _mast3r_model = AsymmetricMASt3R.from_pretrained(model_name)
    _mast3r_model.to(device)
    _mast3r_model.eval()
    _mast3r_device = device
    
    print(f">> MASt3R loaded successfully")
    
    return _mast3r_model


def extract_image_patches(
    image: np.ndarray,
    bbox2d: Tuple[int, int, int, int],
    patch_size: int = 512
) -> np.ndarray:
    """
    从图像中提取物体 crop
    
    Args:
        image: (H, W, 3) RGB image
        bbox2d: (x1, y1, x2, y2) bounding box
        patch_size: 输出 patch 大小
    
    Returns:
        cropped and resized image
    """
    x1, y1, x2, y2 = bbox2d
    crop = image[y1:y2, x1:x2]
    
    # Resize to model input size
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    
    scale = patch_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    crop_resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad to square
    result = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    y_offset = (patch_size - new_h) // 2
    x_offset = (patch_size - new_w) // 2
    result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = crop_resized
    
    return result


def render_object_views(
    mesh_vertices: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    image_size: Tuple[int, int],
    num_views: int = 8
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    渲染物体多个视角用于匹配
    
    Args:
        mesh_vertices: (N, 3) 物体点云
        K: (3, 3) 内参
        R, T: 相机位姿
        image_size: (H, W)
        num_views: 视角数量
    
    Returns:
        rendered_images, rendered_depths
    """
    # 计算elevation
    elevation = np.arctan2(R[1, 2], R[2, 2]) if np.abs(R[1, 2]) > 0.01 else np.pi / 4
    
    rendered_images = []
    rendered_depths = []
    
    for i in range(num_views):
        azimuth = 2 * np.pi * i / num_views
        
        # 构建渲染相机位姿
        # 简化：使用简单的正交投影渲染
        R_view = rotation_matrix_y(azimuth)
        
        # 投影点
        points_2d = project_points(mesh_vertices, K, R_view, T)
        
        # 创建简单渲染图像
        rendered = np.zeros(image_size + (3,), dtype=np.uint8)
        rendered_depth = np.full(image_size, np.inf)
        
        # 简单点渲染
        for pt in points_2d:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < image_size[1] and 0 <= y < image_size[0]:
                rendered[y, x] = 255
                rendered_depth[y, x] = pt[2]
        
        rendered_images.append(rendered)
        rendered_depths.append(rendered_depth)
    
    return rendered_images, rendered_depths


def rotation_matrix_y(angle: float) -> np.ndarray:
    """绕 Y 轴旋转矩阵"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def project_points(
    points: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    T: np.ndarray
) -> np.ndarray:
    """投影 3D 点到 2D"""
    points_cam = points @ R.T + T
    points_2d = points_cam @ K.T
    points_2d[:, :2] /= points_2d[:, 2:3]
    return points_2d


def mast3r_match_and_solve_pnp(
    image_crop: np.ndarray,
    rendered_view: np.ndarray,
    rendered_depth: np.ndarray,
    K_crop: np.ndarray,
    focal_length: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    用 MASt3R 进行 2D-2D 匹配，然后用 PnP 求解姿态
    
    Args:
        image_crop: 真实物体图像 crop
        rendered_view: 渲染的物体视图
        rendered_depth: 渲染的深度图
        K_crop: 裁剪区域的相机内参
        focal_length: 焦距
    
    Returns:
        R, T, confidence
    """
    model = load_mast3r_model()
    
    device = next(model.parameters()).device
    
    # 准备输入
    img1 = torch.from_numpy(image_crop).permute(2, 0, 1).float() / 255.0
    img2 = torch.from_numpy(rendered_view).permute(2, 0, 1).float() / 255.0
    
    # 缩放到 [0, 1]
    img1 = torch.nn.functional.interpolate(img1.unsqueeze(0), size=(512, 512), mode='bilinear')
    img2 = torch.nn.functional.interpolate(img2.unsqueeze(0), size=(512, 512), mode='bilinear')
    
    img1 = img1.to(device)
    img2 = img2.to(device)
    
    with torch.no_grad():
        # MASt3R 双向匹配
        # output1, output2 = model(img1, img2)
        # matches = output1['matches_{1,2}']  # 2D-2D 匹配
        
        # 简化：使用简化的特征匹配
        # 实际应用中应该用完整的 MASt3R pipeline
        matches = compute_simplified_matches(img1, img2)
    
    # 从 2D 匹配和渲染深度恢复 3D-2D 对应
    object_points = []  # 3D 点
    image_points = []   # 2D 点
    
    for match in matches:
        pt1, pt2 = match  # (x1, y1), (x2, y2) in 512x512 space
        
        # 从渲染深度恢复 3D
        x2, y2 = int(pt2[0]), int(pt2[1])
        if 0 <= x2 < 512 and 0 <= y2 < 512:
            z = rendered_depth[y2, x2]
            if np.isfinite(z) and z > 0:
                # 反投影到 3D
                X = (x2 - 256) * z / focal_length
                Y = (y2 - 256) * z / focal_length
                object_points.append([X, Y, z])
                # 映射回原始图像坐标
                image_points.append([pt1[0] * image_crop.shape[1] / 512,
                                    pt1[1] * image_crop.shape[0] / 512])
    
    if len(object_points) < 4:
        # 匹配点不足，返回 None
        return np.eye(3), np.zeros(3), 0.0
    
    object_points = np.array(object_points, dtype=np.float64)
    image_points = np.array(image_points, dtype=np.float64)
    
    # PnP 求解
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        K_crop,
        distCoeffs=None,
        flags=cv2.SOLVEPNP_EPNP,
        ransacReprojThreshold=3.0,
        confidence=0.99
    )
    
    if success:
        R, _ = cv2.Rodrigues(rvec)
        confidence = len(inliers) / len(object_points) if inliers is not None else 0.0
        return R, tvec.flatten(), confidence
    else:
        return np.eye(3), np.zeros(3), 0.0


def compute_simplified_matches(img1: torch.Tensor, img2: torch.Tensor) -> List:
    """
    简化的特征匹配（实际应用中应该用完整的 MASt3R）
    
    这里需要实际调用 MASt3R 的匹配功能
    """
    # TODO: 实现完整的 MASt3R 匹配
    # 实际应该调用:
    # output1, output2 = model(img1, img2)
    # matches = output1['matches_{1,2}']
    
    return []  # 占位


def estimate_yaw_with_mast3r(
    pseudo_lidar: np.ndarray,
    image: np.ndarray,
    bbox2d: Tuple[int, int, int, int],
    K: np.ndarray,
    prior_dims: Tuple[float, float, float],
    ground_equ: np.ndarray = None
) -> Tuple[float, float]:
    """
    用 MASt3R 估计物体朝向角
    
    完整流程:
    1. 提取物体 crop
    2. 渲染多个视角
    3. MASt3R 2D-2D 匹配
    4. PnP 求解
    5. 提取 yaw
    
    Args:
        pseudo_lidar: (N, 3) 伪激光点云
        image: (H, W, 3) RGB 图像
        bbox2d: 2D 边界框
        K: (3, 3) 相机内参
        prior_dims: (w, h, l) 先验尺寸
        ground_equ: 地面方程
    
    Returns:
        yaw, confidence
    """
    # 1. 提取物体 crop
    crop = extract_image_patches(image, bbox2d)
    
    # 2. 构建简化的物体点云（从伪激光点云采样）
    if len(pseudo_lidar) > 500:
        indices = np.random.choice(len(pseudo_lidar), 500, replace=False)
        points_3d = pseudo_lidar[indices]
    else:
        points_3d = pseudo_lidar
    
    # 3. 渲染多个视角
    focal = K[0, 0]
    num_views = 8
    best_yaw = 0.0
    best_confidence = 0.0
    
    for i in range(num_views):
        azimuth = 2 * np.pi * i / num_views
        R_view = rotation_matrix_y(azimuth)
        
        # 渲染
        rendered_img = np.zeros((512, 512, 3), dtype=np.uint8)
        rendered_depth = np.full((512, 512), np.inf)
        
        # 简化的渲染
        for j, pt in enumerate(points_3d):
            pt_cam = pt @ R_view.T
            x = int(256 + pt_cam[0] * focal / pt_cam[2])
            y = int(256 - pt_cam[1] * focal / pt_cam[2])
            if 0 <= x < 512 and 0 <= y < 512:
                rendered_img[y, x] = 255
                rendered_depth[y, x] = pt_cam[2]
        
        # 4. MASt3R 匹配 + PnP
        K_crop = np.array([
            [focal, 0, 256],
            [0, focal, 256],
            [0, 0, 1]
        ], dtype=np.float64)
        
        R_est, T_est, conf = mast3r_match_and_solve_pnp(
            crop, rendered_img, rendered_depth, K_crop, focal
        )
        
        if conf > best_confidence:
            best_confidence = conf
            # 从旋转矩阵提取 yaw
            best_yaw = np.arctan2(R_est[0, 2], R_est[0, 0])
    
    # 5. 如果 MASt3R 置信度低，回退到 PCA
    if best_confidence < 0.1:
        from sklearn.decomposition import PCA
        xz = pseudo_lidar[:, [0, 2]]
        pca = PCA(2)
        pca.fit(xz)
        best_yaw = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
        best_confidence = 0.0  # 标记为回退方法
    
    return best_yaw, best_confidence


def estimate_bbox_mast3r(
    in_pc: np.ndarray,
    prior: np.ndarray,
    image: np.ndarray,
    bbox2d: Tuple[int, int, int, int],
    K: np.ndarray,
    ground_equ: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    用 MASt3R 增强的 BBox 估计
    
    替换 estimate_bbox 函数中的 PCA yaw 估计
    
    Args:
        in_pc: (N, 3) 伪激光点云
        prior: (w, h, l) 先验尺寸
        image: RGB 图像
        bbox2d: 2D 边界框
        K: 相机内参
        ground_equ: 地面方程
    
    Returns:
        vertices, center, dimensions, R_cam
    """
    # 地面处理
    if ground_equ is not None:
        dot_product = np.dot([0, -1, 0], ground_equ[:3])
        if dot_product <= 0:
            ground_equ = -ground_equ
        new_ground_equ = np.array([0, -1, 0, point_to_plane_distance(ground_equ, 0, 0, 0)])
        rotation_matrix = rotation_matrix_from_vectors([0, -1, 0], ground_equ[:3])
    else:
        rotation_matrix = np.eye(3)
        new_ground_equ = None
    
    # 点云旋转到地面坐标系
    rotated_pc = np.dot(in_pc, rotation_matrix)
    
    # 用 MASt3R 估计 yaw
    yaw, mast3r_conf = estimate_yaw_with_mast3r(
        in_pc, image, bbox2d, K, prior, ground_equ
    )
    
    # 后续处理与原 estimate_bbox 相同...
    # ...
    
    return vertices, center, dimensions, R_cam


# --------------------------------------------------------------------------- #
# 辅助函数
# --------------------------------------------------------------------------- #

def rotation_matrix_from_vectors(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """从一个向量旋转到另一个向量的旋转矩阵"""
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    
    v = np.cross(vec1, vec2)
    c = np.dot(vec1, vec2)
    
    if c < -0.9999:
        # 180度旋转
        return -np.eye(3)
    
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    
    R = np.eye(3) + vx + vx @ vx / (1 + c)
    return R


def point_to_plane_distance(plane: np.ndarray, x: float, y: float, z: float) -> float:
    """计算点到平面距离"""
    a, b, c, d = plane
    return (a * x + b * y + c * z + d) / np.sqrt(a*a + b*b + c*c)


if __name__ == "__main__":
    # 测试代码
    print("MASt3R helper module loaded")
    print("Usage: from mast3r_helper import load_mast3r_model, estimate_yaw_with_mast3r")
