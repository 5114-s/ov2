# Copyright (c) Meta Platforms, Inc. and affiliates
"""
MASt3R-based Yaw Estimation Module for OVM3D-Det

Complete implementation following LabelAny3D's approach:
- Uses TRELLIS for 3D model generation
- Uses MASt3R for 2D-2D matching
- Uses PnP for 6-DoF pose estimation
- Extracts yaw from rotation matrix

This is the improved version using TRELLIS to generate textured 3D models,
which provides much better matching quality than pointcloud rendering.
"""

import torch
import numpy as np
import cv2
from typing import Tuple, Optional, List, Dict
import os
import warnings
from scipy.spatial import ConvexHull
from pathlib import Path

# MASt3R model singleton
_mast3r_model = None
_mast3r_device = None
_mast3r_loaded = False

# TRELLIS model singleton
_trellis_pipeline = None
_trellis_loaded = False

# Cache for generated GLB files
_trellis_cache_dir = None


def load_mast3r_model(device: str = "cuda") -> 'AsymmetricMASt3R':
    """
    Load MASt3R model (singleton pattern).

    Args:
        device: 'cuda' or 'cpu'

    Returns:
        MASt3R model
    """
    global _mast3r_model, _mast3r_device, _mast3r_loaded

    if _mast3r_loaded and _mast3r_model is not None and str(_mast3r_device) == str(device):
        return _mast3r_model

    try:
        from mast3r.model import AsymmetricMASt3R
    except ImportError:
        warnings.warn("MASt3R not installed. Install from: https://github.com/naver/mast3r")
        return None

    print(f">> Loading MASt3R model on {device}...")

    # Using metric version for better accuracy
    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"

    try:
        _mast3r_model = AsymmetricMASt3R.from_pretrained(model_name)
        _mast3r_model.to(device)
        _mast3r_model.eval()
        _mast3r_device = device
        _mast3r_loaded = True
        print(f">> MASt3R loaded successfully")
    except Exception as e:
        print(f">> Failed to load MASt3R: {e}")
        _mast3r_loaded = False
        return None

    return _mast3r_model


def set_trellis_cache_dir(cache_dir: str):
    """Set the cache directory for TRELLIS generated models."""
    global _trellis_cache_dir
    _trellis_cache_dir = cache_dir
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)


def load_trellis_model(device: str = "cuda"):
    """
    Load TRELLIS model (singleton pattern).

    Following LabelAny3D's approach:
    - Uses TrellisImageTo3DPipeline
    - Lazy loading

    Args:
        device: 'cuda' or 'cpu'

    Returns:
        TRELLIS pipeline
    """
    global _trellis_pipeline, _trellis_loaded

    if _trellis_loaded and _trellis_pipeline is not None:
        return _trellis_pipeline

    try:
        import sys
        trellis_path = "/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/TRELLIS"
        if trellis_path not in sys.path:
            sys.path.insert(0, trellis_path)

        os.environ['ATTN_BACKEND'] = 'xformers'
        os.environ['SPCONV_ALGO'] = 'native'

        from trellis.pipelines import TrellisImageTo3DPipeline

        print(f">> Loading TRELLIS model on {device}...")
        _trellis_pipeline = TrellisImageTo3DPipeline.from_pretrained(
            "JeffreyXiang/TRELLIS-image-large",
        )
        _trellis_pipeline = _trellis_pipeline.to(dtype=torch.float16)
        _trellis_pipeline.cuda()
        if hasattr(_trellis_pipeline, 'eval'):
            _trellis_pipeline.eval()
        _trellis_loaded = True
        print(f">> TRELLIS loaded successfully (FP16)")
        return _trellis_pipeline
    except ImportError as e:
        warnings.warn(f"TRELLIS not installed: {e}")
        return None
    except Exception as e:
        print(f">> Failed to load TRELLIS: {e}")
        return None


def unload_trellis_model():
    """
    Unload TRELLIS model to free GPU memory.
    Call this after generating models and before loading MASt3R.
    """
    global _trellis_pipeline, _trellis_loaded

    if _trellis_pipeline is not None:
        del _trellis_pipeline
        _trellis_pipeline = None
        _trellis_loaded = False
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print(">> TRELLIS model unloaded, GPU memory freed")


def preprocess_image_for_trellis(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for TRELLIS - remove background using rembg.
    
    IMPORTANT: Keep soft alpha channel. TRELLIS expects soft alpha for bbox calculation.

    Args:
        image: (H, W, 3) RGB image

    Returns:
        RGBA image with transparent background (soft alpha)
    """
    try:
        from rembg import remove
        rgba = remove(image)
        return rgba
    except ImportError:
        # Fallback: simple edge-based mask
        import cv2
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = np.ones((10, 10), np.uint8)
        mask = cv2.dilate(edges, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask.astype(np.float32), (21, 21), 0) * 255
        rgba = np.concatenate([image, mask.astype(np.uint8)[..., None]], axis=-1)
        return rgba


def generate_3d_model_trellis(
    image_crop: np.ndarray,
    obj_id: str,
    cache_dir: str = None,
    seed: int = 42
) -> Optional[Dict]:
    """
    Generate 3D model using TRELLIS.

    Following LabelAny3D's approach:
    1. Preprocess image (remove background)
    2. Run TRELLIS pipeline
    3. Export to GLB with texture baking

    Args:
        image_crop: (H, W, 3) RGB image crop
        obj_id: unique identifier for caching
        cache_dir: directory to cache generated models
        seed: random seed

    Returns:
        Dictionary with 'mesh', 'glb_path', 'gaussian' or None if failed
    """
    from PIL import Image

    global _trellis_cache_dir

    cache_dir = cache_dir or _trellis_cache_dir

    # Check cache
    if cache_dir and obj_id:
        glb_path = os.path.join(cache_dir, f"{obj_id}.glb")
        if os.path.exists(glb_path):
            return {'glb_path': glb_path, 'cached': True}

    pipeline = load_trellis_model()
    if pipeline is None:
        return None

    try:
        # Preprocess image (remove background)
        rgba_image = preprocess_image_for_trellis(image_crop)
        pil_image = Image.fromarray(rgba_image)

        # Run TRELLIS pipeline
        outputs = pipeline.run(
            pil_image,
            seed=seed,
            sparse_structure_sampler_params={"steps": 12, "cfg_strength": 7.5},
            slat_sampler_params={"steps": 12, "cfg_strength": 3},
        )

        result = {}

        # Get mesh and gaussian representations
        if 'mesh' in outputs and len(outputs['mesh']) > 0:
            result['mesh'] = outputs['mesh'][0]

        if 'gaussian' in outputs and len(outputs['gaussian']) > 0:
            result['gaussian'] = outputs['gaussian'][0]

        # Export to GLB with texture baking
        if cache_dir and 'gaussian' in result and 'mesh' in result:
            os.makedirs(os.path.join(cache_dir, 'object_space'), exist_ok=True)
            glb_path = os.path.join(cache_dir, 'object_space', f"{obj_id}.glb")

            from trellis.utils import postprocessing_utils
            glb = postprocessing_utils.to_glb(
                result['gaussian'],
                result['mesh'],
                texture_size=1024,
                simplify=0.95,
            )
            glb.export(glb_path)
            result['glb_path'] = glb_path

        return result

    except Exception as e:
        print(f">> TRELLIS inference failed for {obj_id}: {e}")
        return None


def render_glb_views_pytorch3d(
    glb_path: str,
    num_views: int = 8,
    resolution: int = 512,
    radius: float = 1.5,
    elevation: float = 0.0
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Render GLB model from multiple viewpoints using PyTorch3D (following LabelAny3D).

    Key differences from nvdiffrast version:
    - Properly loads textures from GLB
    - Uses HardPhongShader with ambient lighting for proper visualization
    - Returns proper rotation/translation matrices

    Args:
        glb_path: path to GLB file
        num_views: number of viewpoints (evenly distributed in azimuth)
        resolution: output resolution
        radius: camera distance from object
        elevation: camera elevation angle (degrees)

    Returns:
        renderings: list of rendered RGB images
        depths: list of depth maps
        Rs: list of rotation matrices (world to camera)
        Ts: list of translation vectors
    """
    try:
        import torch
        import numpy as np
        from pytorch3d.io import IO
        from pytorch3d.io.experimental_gltf_io import MeshGlbFormat
        from pytorch3d.renderer import (
            PerspectiveCameras,
            RasterizationSettings,
            MeshRendererWithFragments,
            MeshRasterizer,
            HardPhongShader,
            PointLights,
            look_at_view_transform
        )
    except ImportError as e:
        warnings.warn(f"PyTorch3D not installed: {e}")
        return [], [], [], []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    renderings = []
    depths = []
    Rs = []
    Ts = []

    try:
        # Load mesh with textures using PyTorch3D's IO (like LabelAny3D)
        io = IO()
        io.register_meshes_format(MeshGlbFormat())
        mesh = io.load_mesh(glb_path, include_textures=True)
        mesh = mesh.to(device)

        # Camera intrinsics (same as LabelAny3D)
        focal_length = 560.44
        principal_point = (256, 256)

        # Render each view
        for i in range(num_views):
            azimuth = 360.0 * i / num_views

            # Get camera rotation and translation (LabelAny3D uses look_at_view_transform)
            R_view, T_view = look_at_view_transform(
                dist=radius,
                elev=elevation,
                azim=azimuth,
                device=device
            )

            # Setup camera with LabelAny3D parameters
            cameras = PerspectiveCameras(
                focal_length=((focal_length, focal_length),),
                principal_point=((principal_point[0], principal_point[1]),),
                in_ndc=False,
                image_size=[[resolution, resolution]],
                device=device,
                R=R_view.permute(0, 2, 1),  # Convert to camera convention
                T=T_view
            )

            # Rasterization settings
            raster_settings = RasterizationSettings(
                image_size=resolution,
                blur_radius=0.0,
                faces_per_pixel=1,
            )

            # Lighting (ambient for better visibility)
            lights = PointLights(
                device=device,
                location=((0.0, 2.98e-08, -1.0),),
                ambient_color=((1.0, 1.0, 1.0),),
                diffuse_color=((0.0, 0.0, 0.0),),
                specular_color=((0.0, 0.0, 0.0),),
            )

            # Create renderer (same as LabelAny3D)
            renderer = MeshRendererWithFragments(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=raster_settings
                ),
                shader=HardPhongShader(
                    device=device,
                    cameras=cameras,
                    lights=lights
                )
            )

            # Render
            with torch.no_grad():
                image, fragments = renderer(meshes_world=mesh)

            # Convert to numpy
            rgb = image.cpu().numpy().squeeze()  # [H, W, 4] in [0, 1]
            depth = fragments.zbuf.cpu().numpy().squeeze()  # [H, W]

            # Convert RGB to uint8 [0, 255]
            rgb = (rgb[..., :3] * 255).clip(0, 255).astype(np.uint8)

            # Convert depth (from NDC to linear if needed, handle invalid values)
            # depth values are in NDC space, need to convert
            depth = depth.astype(np.float32)
            depth[depth < 0] = np.inf  # Mark invalid depth as infinity

            renderings.append(rgb)
            depths.append(depth)

            # Store rotation and translation in numpy format
            # R_view is [1, 3, 3], T_view is [1, 3]
            Rs.append(R_view.cpu().numpy().squeeze())
            Ts.append(T_view.cpu().numpy().squeeze())

        return renderings, depths, Rs, Ts

    except Exception as e:
        print(f">> PyTorch3D GLB rendering failed for {glb_path}: {e}")
        import traceback
        traceback.print_exc()
        return [], [], [], []


def render_glb_views(
    glb_path: str,
    num_views: int = 8,
    resolution: int = 512,
    radius: float = 1.5
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Render GLB model from multiple viewpoints.
    
    Now uses PyTorch3D (following LabelAny3D) for proper texture handling.

    Args:
        glb_path: path to GLB file
        num_views: number of viewpoints (evenly distributed in azimuth)
        resolution: output resolution
        radius: camera distance from object

    Returns:
        renderings: list of rendered RGB images
        depths: list of depth maps
        Rs: list of rotation matrices
        Ts: list of translation vectors
    """
    return render_glb_views_pytorch3d(glb_path, num_views, resolution, radius)


def render_mesh_views_pytorch3d(
    mesh_data: Dict,
    num_views: int = 8,
    resolution: int = 512,
    radius: float = 1.5
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Render mesh using PyTorch3D.

    Alternative to GLB rendering if nvdiffrast is not available.

    Args:
        mesh_data: dict with 'mesh' from TRELLIS outputs
        num_views: number of viewpoints
        resolution: output resolution
        radius: camera distance

    Returns:
        renderings, depths, Rs, Ts
    """
    try:
        import pytorch3d
        from pytorch3d.renderer import (
            FoVPerspectiveCameras,
            RasterizationSettings,
            MeshRenderer,
            MeshRasterizer,
            SoftPhongShader,
            TexturesVertex
        )
        from pytorch3d.structures import Meshes
        import torch.nn.functional as F
    except ImportError:
        warnings.warn("PyTorch3D not installed")
        return [], [], [], []

    renderings = []
    depths = []
    Rs = []
    Ts = []

    try:
        mesh_obj = mesh_data.get('mesh')
        if mesh_obj is None:
            return [], [], [], []

        # Extract vertices and faces from mesh
        # The exact format depends on TRELLIS output
        if hasattr(mesh_obj, 'vertices') and hasattr(mesh_obj, 'faces'):
            verts = torch.tensor(mesh_obj.vertices, dtype=torch.float32).cuda()
            faces = torch.tensor(mesh_obj.faces, dtype=torch.int64).cuda()
        else:
            # Try to get from internal structure
            return [], [], [], []

        # Center and scale
        center = verts.mean(dim=0)
        verts = verts - center
        scale = 1.0 / (verts.abs().max() + 1e-6)
        verts = verts * scale

        # Default vertex colors
        textures = TexturesVertex(verts_features=torch.ones_like(verts).cuda())

        # Create mesh
        mesh = Meshes(verts=[verts], faces=[faces], textures=textures)

        # Camera setup
        focal_length = resolution / (2 * np.tan(np.radians(60) / 2))

        for i in range(num_views):
            azimuth = 2 * np.pi * i / num_views
            cam_x = radius * np.sin(azimuth)
            cam_z = radius * np.cos(azimuth)
            cam_y = radius * 0.2

            # Look-at rotation
            R = pytorch3d.renderer.quaternions.quat_to_rotmat(
                pytorch3d.renderer.look_at_viewpoint(
                    (cam_x, cam_y, cam_z),
                    (0, 0, 0),
                    (0, 1, 0)
                )
            )
            T = torch.tensor([[-cam_x, -cam_y, -cam_z]], dtype=torch.float32).cuda()

            Rs.append(R.cpu().numpy())
            Ts.append(np.array([cam_x, cam_y, cam_z]))

            # Camera object
            cameras = FoVPerspectiveCameras(
                R=R.permute(1, 0).unsqueeze(0).cuda(),
                T=T.cuda(),
                fov=60,
            )

            # Renderer
            raster_settings = RasterizationSettings(
                image_size=resolution,
                blur_radius=0,
                faces_per_pixel=1,
            )

            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
                shader=SoftPhongShader(device='cuda', cameras=cameras)
            )

            # Render
            image = renderer(mesh)
            color = image[0, :, :, :3].cpu().numpy()
            color = (color * 255).clip(0, 255).astype(np.uint8)

            # Get depth approximation
            depth = image[0, :, :, 3].cpu().numpy()

            renderings.append(color)
            depths.append(depth)

        return renderings, depths, Rs, Ts

    except Exception as e:
        print(f">> PyTorch3D rendering failed: {e}")
        return [], [], [], []


def render_pointcloud_views(
    points_3d: np.ndarray,
    image_rgb: np.ndarray,
    bbox2d: Tuple[int, int, int, int],
    num_views: int = 8,
    patch_size: int = 512,
    focal_length: float = 560.44
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Render object from multiple viewpoints using depth-based rendering.
    
    This replaces LabelAny3D's GLB rendering with pointcloud rendering.
    
    Args:
        points_3d: (N, 3) object point cloud in camera coordinates
        image_rgb: (H, W, 3) original RGB image
        bbox2d: (x1, y1, x2, y2) 2D bounding box
        num_views: number of viewpoints
        patch_size: output image size
        focal_length: virtual focal length
    
    Returns:
        renderings: list of rendered RGB images
        depths: list of depth maps
        Rs: list of rotation matrices (camera to world)
        Ts: list of translation vectors
    """
    renderings = []
    depths = []
    Rs = []
    Ts = []
    
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = bbox2d
    
    # Extract and crop object region
    obj_crop = image_rgb[y1:y2, x1:x2]
    obj_h, obj_w = obj_crop.shape[:2]
    
    # Sample points if too many
    if len(points_3d) > 1000:
        indices = np.random.choice(len(points_3d), 1000, replace=False)
        pts = points_3d[indices]
    else:
        pts = points_3d
    
    # Center the point cloud
    center = pts.mean(axis=0)
    pts_centered = pts - center
    
    # Estimate bounding sphere
    distances = np.linalg.norm(pts_centered, axis=1)
    radius = distances.max() * 1.2  # Add margin
    
    # Camera distance for rendering
    cam_distance = radius * 2.0
    
    for i in range(num_views):
        azimuth = 2 * np.pi * i / num_views
        
        # Camera position on sphere
        cam_x = cam_distance * np.sin(azimuth)
        cam_z = cam_distance * np.cos(azimuth)
        cam_y = radius * 0.3  # Slight elevation
        
        # Look-at rotation: camera looks at origin
        # R_cam2world: transforms points from camera to world
        # We want: camera at (cam_x, cam_y, cam_z) looking at origin
        
        # Z-axis of camera (forward) points from camera to origin
        z_axis = -np.array([cam_x, cam_y, cam_z])
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        # Y-axis (up) 
        up = np.array([0, 1, 0])
        if abs(np.dot(z_axis, up)) > 0.99:
            up = np.array([0, 0, 1])
        
        # X-axis (right)
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Y-axis
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # Rotation matrix: camera to world
        R = np.column_stack([x_axis, y_axis, z_axis])
        t = np.array([cam_x, cam_y, cam_z])
        
        Rs.append(R.copy())
        Ts.append(t.copy())
        
        # Project points to image
        pts_cam = (R.T @ (pts_centered.T - t.reshape(3, 1))).T  # Points in camera coords
        
        # Filter points in front of camera
        valid = pts_cam[:, 2] > 0.01
        pts_cam = pts_cam[valid]
        
        if len(pts_cam) == 0:
            renderings.append(np.zeros((patch_size, patch_size, 3), dtype=np.uint8))
            depths.append(np.full((patch_size, patch_size), np.inf))
            continue
        
        # Project to 2D
        fx = focal_length
        fy = focal_length
        cx, cy = patch_size / 2, patch_size / 2
        
        x_2d = (pts_cam[:, 0] * fx / pts_cam[:, 2] + cx).astype(int)
        y_2d = (-pts_cam[:, 1] * fy / pts_cam[:, 2] + cy).astype(int)
        z_2d = pts_cam[:, 2]
        
        # Create rendering and depth map
        rendering = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
        depth_map = np.full((patch_size, patch_size), np.inf)
        
        # Map original point colors
        valid_pts = pts[valid]
        colors = obj_crop.reshape(-1, 3) if len(obj_crop.shape) == 3 else np.zeros((len(pts), 3))
        
        # Simple nearest-neighbor coloring based on UV
        u_coords = ((pts_cam[:, 0] / radius + 1) * obj_w / 2).astype(int).clip(0, obj_w - 1)
        v_coords = ((-pts_cam[:, 1] / radius + 1) * obj_h / 2).astype(int).clip(0, obj_h - 1)
        point_colors = obj_crop[v_coords, u_coords] if len(pts_cam) > 0 else colors
        
        for j in range(len(pts_cam)):
            xi, yi = x_2d[j], y_2d[j]
            if 0 <= xi < patch_size and 0 <= yi < patch_size:
                if z_2d[j] < depth_map[yi, xi]:
                    depth_map[yi, xi] = z_2d[j]
                    # Color by depth + original color blend
                    depth_norm = (z_2d[j] - pts_cam[:, 2].min()) / (pts_cam[:, 2].max() - pts_cam[:, 2].min() + 1e-6)
                    base_color = point_colors[j] if j < len(point_colors) else [128, 128, 128]
                    color = (np.array(base_color) * 0.5 + np.array([255 * depth_norm, 128, 255 * (1 - depth_norm)]) * 0.5).astype(np.uint8)
                    rendering[yi, xi] = color
        
        renderings.append(rendering)
        depths.append(depth_map)
    
    return renderings, depths, Rs, Ts


def render_trellis_views(
    image_rgb: np.ndarray,
    bbox2d: Tuple[int, int, int, int],
    obj_id: str,
    cache_dir: str = None,
    num_views: int = 8,
    resolution: int = 512,
    radius: float = 1.5
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Render object from TRELLIS-generated 3D model.

    This is the improved version following LabelAny3D's approach:
    1. Use TRELLIS to generate textured 3D mesh from image crop
    2. Render virtual views using nvdiffrast or PyTorch3D

    Args:
        image_rgb: (H, W, 3) RGB image
        bbox2d: (x1, y1, x2, y2) 2D bounding box
        obj_id: unique identifier for caching
        cache_dir: directory to cache generated models
        num_views: number of viewpoints
        resolution: output resolution
        radius: camera distance

    Returns:
        renderings: list of rendered RGB images
        depths: list of depth maps
        Rs: list of rotation matrices
        Ts: list of translation vectors
    """
    # Extract object crop
    x1, y1, x2, y2 = bbox2d
    obj_crop = image_rgb[y1:y2, x1:x2]

    if obj_crop.size == 0:
        return [], [], [], []

    # Generate 3D model using TRELLIS
    mesh_data = generate_3d_model_trellis(
        obj_crop, obj_id, cache_dir
    )

    if mesh_data is None:
        return [], [], [], []

    # Try GLB rendering first, then fallback to mesh rendering
    glb_path = mesh_data.get('glb_path')
    if glb_path and os.path.exists(glb_path):
        return render_glb_views(glb_path, num_views, resolution, radius)

    # Fallback to mesh rendering via PyTorch3D
    if 'mesh' in mesh_data:
        return render_mesh_views_pytorch3d(mesh_data, num_views, resolution, radius)

    return [], [], [], []


def get_mast3r_matches_fast(
    image1: np.ndarray,
    image2: np.ndarray,
    model: 'AsymmetricMASt3R',
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get 2D-2D matches using MASt3R.
    
    Args:
        image1: first image (H, W, 3) RGB [0-255]
        image2: second image (H, W, 3) RGB [0-255]
        model: MASt3R model
        device: cuda/cpu
    
    Returns:
        matches_im0: (M, 2) matches in image1 coordinates
        matches_im1: (M, 2) matches in image2 coordinates
        conf: (M,) confidence scores
    """
    try:
        from mast3r.fast_nn import fast_reciprocal_NNs
        from dust3r.inference import inference
        from dust3r.utils.image import load_images
    except ImportError:
        warnings.warn("MASt3R/DUSt3R dependencies not installed")
        return np.array([]), np.array([]), np.array([])
    
    # Prepare images (MASt3R expects [0,1] float)
    img1 = torch.from_numpy(image1).permute(2, 0, 1).float() / 255.0
    img2 = torch.from_numpy(image2).permute(2, 0, 1).float() / 255.0
    
    # Resize to 512x512
    img1 = torch.nn.functional.interpolate(img1.unsqueeze(0), size=(512, 512), mode='bilinear')
    img2 = torch.nn.functional.interpolate(img2.unsqueeze(0), size=(512, 512), mode='bilinear')
    
    img1 = img1.to(device)
    img2 = img2.to(device)
    
    with torch.no_grad():
        # MASt3R forward pass
        output = inference([(img1, img2)], model, device, batch_size=1, verbose=False)
        
        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']
        
        # Get descriptors
        desc1 = pred1['desc'].squeeze(0).detach()
        desc2 = pred2['desc'].squeeze(0).detach()
        
        # Find reciprocal nearest neighbors
        matches_im0, matches_im1 = fast_reciprocal_NNs(
            desc1, desc2,
            subsample_or_initxy1=8,
            device=device,
            dist='dot',
            block_size=2**13
        )
        
        # Get confidence
        conf1 = pred1['conf'].squeeze(0).detach().cpu().numpy()
        
        # Filter by confidence
        H0, W0 = view1['true_shape'][0].item(), view1['true_shape'][1].item()
        conf_threshold = 0.3
        
        # Convert to numpy
        matches_im0 = matches_im0.cpu().numpy()
        matches_im1 = matches_im1.cpu().numpy()
        
        # Filter border matches
        valid = (
            (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < W0 - 3) &
            (matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < H0 - 3) &
            (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < 512 - 3) &
            (matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < 512 - 3)
        )
        
        matches_im0 = matches_im0[valid]
        matches_im1 = matches_im1[valid]
        
        # Get confidence for valid matches
        conf = np.array([conf1[0, int(m[1]), int(m[0])] for m in matches_im0])
        
    return matches_im0, matches_im1, conf


def solve_pnp(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray = None,
    iterations: int = 1000,
    reproj_threshold: float = 5.0
) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Solve PnP using RANSAC.
    
    Args:
        object_points: (N, 3) 3D points in world coordinates
        image_points: (N, 2) 2D image points
        camera_matrix: (3, 3) camera intrinsic
        dist_coeffs: distortion coefficients
        iterations: RANSAC iterations
        reproj_threshold: RANSAC reprojection threshold
    
    Returns:
        success: bool
        rvec: rotation vector
        tvec: translation vector
        inliers: inlier indices
        error: mean reprojection error
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5)
    
    object_points = np.array(object_points, dtype=np.float64)
    image_points = np.array(image_points, dtype=np.float64)
    camera_matrix = np.array(camera_matrix, dtype=np.float64)
    dist_coeffs = np.array(dist_coeffs, dtype=np.float64)
    
    if len(object_points) < 4:
        return False, None, None, None, None
    
    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=object_points,
            imagePoints=image_points,
            cameraMatrix=camera_matrix,
            distCoeffs=dist_coeffs,
            iterationsCount=iterations,
            reprojectionError=reproj_threshold,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success and inliers is not None and len(inliers) >= 4:
            # Calculate reprojection error
            projected, _ = cv2.projectPoints(
                object_points,
                rvec, tvec,
                camera_matrix,
                dist_coeffs
            )
            error = np.linalg.norm(image_points - projected.reshape(-1, 2), axis=1).mean()
            return True, rvec, tvec, inliers, error
    except Exception as e:
        print(f"PnP error: {e}")
    
    return False, None, None, None, None


def rotation_vector_to_yaw(rvec: np.ndarray) -> float:
    """Extract yaw angle (rotation around Y) from rotation vector."""
    R, _ = cv2.Rodrigues(rvec)
    # For yaw (rotation around Y axis):
    yaw = np.arctan2(R[0, 2], R[0, 0])
    return yaw


def estimate_yaw_mast3r_complete(
    pseudo_lidar: np.ndarray,
    image_rgb: np.ndarray,
    bbox2d: Tuple[int, int, int, int],
    K: np.ndarray,
    prior_dims: Tuple[float, float, float] = None,
    num_views: int = 8,
    obj_id: str = None,
    cache_dir: str = None,
    use_trellis: bool = True,
    mast3r_model=None
) -> Tuple[float, float]:
    """
    Complete MASt3R + PnP yaw estimation following LabelAny3D.

    Pipeline:
    1. Render object from multiple virtual views (TRELLIS-based or pointcloud fallback)
    2. MASt3R match: real image vs rendered views
    3. Convert matches to 3D-2D correspondences
    4. PnP solve for 6-DoF pose
    5. Extract yaw from rotation

    Args:
        pseudo_lidar: (N, 3) pseudo lidar points in camera coords
        image_rgb: (H, W, 3) RGB image
        bbox2d: (x1, y1, x2, y2) 2D bounding box
        K: (3, 3) camera intrinsic matrix
        prior_dims: prior dimensions (w, h, l)
        num_views: number of virtual views
        obj_id: unique identifier for TRELLIS caching
        cache_dir: directory to cache TRELLIS models
        use_trellis: if True, use TRELLIS for 3D generation; else fallback to pointcloud
        mast3r_model: pre-loaded MASt3R model (for memory management)

    Returns:
        yaw: estimated yaw angle
        confidence: confidence score (0-1)
    """
    # Load MASt3R if not provided (and unload TRELLIS first to free memory)
    if mast3r_model is None:
        # Unload TRELLIS to free GPU memory before loading MASt3R
        unload_trellis_model()
        mast3r_model = load_mast3r_model()

    if mast3r_model is None:
        return None, 0.0  # Signal fallback

    device = str(next(mast3r_model.parameters()).device)

    # Extract object crop from image
    x1, y1, x2, y2 = bbox2d
    obj_crop = image_rgb[y1:y2, x1:x2]

    if obj_crop.size == 0 or len(pseudo_lidar) < 10:
        return None, 0.0

    # Resize crop to 512x512 (MASt3R input size)
    obj_crop_512 = cv2.resize(obj_crop, (512, 512))

    # Render virtual views - use TRELLIS if enabled
    if use_trellis:
        renderings, depths, Rs, Ts = render_trellis_views(
            image_rgb, bbox2d, obj_id, cache_dir,
            num_views=num_views, resolution=512
        )
        # If TRELLIS fails, fallback to pointcloud rendering
        if len(renderings) == 0:
            renderings, depths, Rs, Ts = render_pointcloud_views(
                pseudo_lidar, image_rgb, bbox2d,
                num_views=num_views,
                patch_size=512,
                focal_length=560.44
            )
    else:
        renderings, depths, Rs, Ts = render_pointcloud_views(
            pseudo_lidar, image_rgb, bbox2d,
            num_views=num_views,
            patch_size=512,
            focal_length=560.44
        )
    
    best_yaw = 0.0
    best_conf = 0.0
    best_error = float('inf')
    
    # Process each virtual view
    for i in range(num_views):
        rendering = renderings[i]
        depth = depths[i]
        R_render = Rs[i]
        T_render = Ts[i]
        
        # Get MASt3R matches
        matches_im0, matches_im1, conf = get_mast3r_matches_fast(
            obj_crop_512, rendering, mast3r_model, device
        )
        
        if len(matches_im0) < 10:
            continue
        
        # Filter by confidence
        conf_threshold = np.percentile(conf, 50) if len(conf) > 0 else 0.3
        high_conf = conf > conf_threshold
        
        if high_conf.sum() < 10:
            continue
        
        matches_im0 = matches_im0[high_conf]
        matches_im1 = matches_im1[high_conf]
        conf = conf[high_conf]
        
        # Convert matches from 512 coords to full crop coords
        crop_h, crop_w = obj_crop.shape[:2]
        scale_x = crop_w / 512.0
        scale_y = crop_h / 512.0
        
        # Map render matches back to camera space using depth
        pts_3d_world = []
        pts_2d_image = []
        
        for j in range(len(matches_im0)):
            m0 = matches_im0[j]  # In 512x512 crop
            m1 = matches_im1[j]  # In 512x512 render
            
            # Get depth at render match position
            u_render = int(m1[0])
            v_render = int(m1[1])
            
            if 0 <= u_render < 512 and 0 <= v_render < 512:
                z_cam = depth[v_render, u_render]
                
                if np.isfinite(z_cam) and z_cam > 0:
                    # Convert render pixel to 3D using render camera parameters (NOT K)
                    # The render uses focal_length=560.44 with center at (256, 256)
                    # This gives us the 3D point in the virtual camera frame
                    fx_render = 560.44
                    fy_render = 560.44
                    cx_render, cy_render = 256, 256

                    # Point in render camera coordinates
                    x_cam = (u_render - cx_render) * z_cam / fx_render
                    y_cam = -(v_render - cy_render) * z_cam / fy_render
                    z_cam_val = z_cam
                    
                    # Transform to world coordinates
                    pt_world = R_render @ np.array([x_cam, y_cam, z_cam_val]) + T_render
                    
                    # Corresponding 2D point in original image (crop)
                    u_crop = m0[0] * scale_x
                    v_crop = m0[1] * scale_y
                    
                    # Convert to full image coordinates
                    u_full = u_crop + x1
                    v_full = v_crop + y1
                    
                    pts_3d_world.append(pt_world)
                    pts_2d_image.append([u_full, v_full])
        
        if len(pts_3d_world) < 6:
            continue
        
        pts_3d_world = np.array(pts_3d_world)
        pts_2d_image = np.array(pts_2d_image)
        
        # Solve PnP
        dist_coeffs = np.zeros(5)
        success, rvec, tvec, inliers, error = solve_pnp(
            pts_3d_world,
            pts_2d_image,
            K,
            dist_coeffs,
            iterations=1000,
            reproj_threshold=10.0
        )
        
        if success and inliers is not None:
            # Extract yaw
            yaw = rotation_vector_to_yaw(rvec)
            
            # Confidence based on inlier ratio
            conf_score = len(inliers) / len(pts_3d_world)
            
            # Prefer lower reprojection error
            if conf_score > best_conf or (conf_score == best_conf and error < best_error):
                best_yaw = yaw
                best_conf = conf_score
                best_error = error
    
    if best_conf < 0.05:
        return None, 0.0
    
    return best_yaw, best_conf


def estimate_yaw_pca(pseudo_lidar: np.ndarray) -> float:
    """Fallback PCA-based yaw estimation."""
    xz = pseudo_lidar[:, [0, 2]]
    
    # Remove outliers
    center = xz.mean(axis=0)
    dists = np.linalg.norm(xz - center, axis=1)
    threshold = np.mean(dists) + 2 * np.std(dists)
    xz = xz[dists <= threshold]
    
    if len(xz) < 10:
        return 0.0
    
    # PCA
    cov = np.cov(xz.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    principal = eigenvectors[np.argmax(eigenvalues)]
    yaw = np.arctan2(principal[1], principal[0])
    
    return yaw


def estimate_yaw_lshape(pseudo_lidar: np.ndarray) -> float:
    """L-Shape MABR yaw estimation."""
    xz = pseudo_lidar[:, [0, 2]]
    
    if len(xz) < 10:
        return 0.0
    
    try:
        hull = ConvexHull(xz)
        hull_pts = xz[hull.vertices]
    except:
        return estimate_yaw_pca(pseudo_lidar)
    
    min_area = float('inf')
    best_yaw = 0.0
    
    n_edges = len(hull_pts)
    for i in range(n_edges):
        p1, p2 = hull_pts[i], hull_pts[(i + 1) % n_edges]
        for angle in [np.arctan2(p2[1]-p1[1], p2[0]-p1[0]), 
                      np.arctan2(p2[1]-p1[1], p2[0]-p1[0]) + np.pi/2]:
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, -s], [s, c]])
            rotated = hull_pts @ R.T
            area = (rotated[:, 0].max() - rotated[:, 0].min()) * \
                   (rotated[:, 1].max() - rotated[:, 1].min())
            if area < min_area:
                min_area, best_yaw = area, angle
    
    return best_yaw


def estimate_yaw_hybrid(
    pseudo_lidar: np.ndarray,
    image_rgb: np.ndarray,
    bbox2d: Tuple[int, int, int, int],
    K: np.ndarray,
    prior_dims: Tuple[float, float, float] = None
) -> Tuple[float, float]:
    """
    Hybrid yaw estimation: try MASt3R first, fallback to L-Shape.
    
    Args:
        pseudo_lidar: (N, 3) pseudo lidar points
        image_rgb: (H, W, 3) RGB image
        bbox2d: 2D bounding box
        K: camera intrinsic
        prior_dims: prior dimensions
    
    Returns:
        yaw, confidence
    """
    # Try MASt3R + PnP
    mast3r_yaw, mast3r_conf = estimate_yaw_mast3r_complete(
        pseudo_lidar, image_rgb, bbox2d, K, prior_dims
    )
    
    if mast3r_yaw is not None and mast3r_conf > 0.1:
        return mast3r_yaw, mast3r_conf
    
    # Fallback to L-Shape
    lshape_yaw = estimate_yaw_lshape(pseudo_lidar)
    return lshape_yaw, 0.3


def rotation_matrix_y(angle: float) -> np.ndarray:
    """Create rotation matrix around Y axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def rotation_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    """Convert rotation matrix to Euler angles (roll, pitch, yaw)."""
    yaw = np.arctan2(R[0, 2], R[0, 0])
    pitch = np.arctan2(-R[1, 2], np.sqrt(R[1, 0]**2 + R[1, 1]**2))
    roll = np.arctan2(R[1, 0], R[1, 1])
    return roll, pitch, yaw
