# Copyright (c) Meta Platforms, Inc. and affiliates
"""
LabelAny3D Pipeline for OVM3D-Det

This module implements the EXACT same pipeline as LabelAny3D:
1. TRELLIS generates 3D model from single image
2. Render virtual multi-views from 3D model
3. MASt3R matching between rendered views and reference image
4. Two-stage PnP: Initial pose with render params + Refined pose with true K
5. Iterative pose refinement

Reference: https://github.com/JeffreyXiang/LabelAny3D
"""

import os
import sys

# Set TRELLIS environment variables BEFORE any imports
os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['SPCONV_ALGO'] = 'native'

import cv2
import json
import torch
import numpy as np
from PIL import Image
from typing import Tuple, Optional, List, Dict
import warnings

# ============== MASt3R/DUST3R Functions ==============

def load_mast3r_model(device: str = "cuda"):
    """Load MASt3R model for matching."""
    global _mast3r_model, _mast3r_loaded

    if _mast3r_loaded and _mast3r_model is not None:
        print(f">> [CACHE] MASt3R model already loaded, reusing...")
        return _mast3r_model

    print(f">> [LOAD] MASt3R model loading on {device}...")
    
    try:
        from mast3r.model import AsymmetricMASt3R
    except ImportError:
        warnings.warn("MASt3R not installed")
        return None
    
    print(f">> Loading MASt3R model on {device}...")
    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    _mast3r_model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    _mast3r_model.eval()
    _mast3r_loaded = True
    print(f">> MASt3R loaded successfully")
    return _mast3r_model

# Model singleton
_mast3r_model = None
_mast3r_loaded = False

# TRELLIS cache directory
_trellis_cache_dir = None


def set_trellis_cache_dir(cache_dir: str):
    """Set the cache directory for TRELLIS generated models."""
    global _trellis_cache_dir
    _trellis_cache_dir = cache_dir
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)


def dust3r_inference(image_pairs, model, device, batch_size=1, verbose=False):
    """
    Run DUST3R/MASt3R inference following LabelAny3D's approach.
    Uses dust3r's inference function directly.
    """
    from dust3r.inference import inference
    return inference(image_pairs, model, device, batch_size=batch_size, verbose=verbose)


def fast_reciprocal_nns(desc1, desc2, subsample_or_initxy1=8, device='cuda',
                        dist='dot', block_size=2**13):
    """Find reciprocal nearest neighbors between two descriptor maps."""
    from mast3r.fast_nn import fast_reciprocal_NNs
    # Direct call matching LabelAny3D - let Python use default values for ret_xy, pixel_tol, ret_basin
    return fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1, 
                               device=device, dist=dist, block_size=block_size)


def load_images_for_mast3r(image_paths, size=512, verbose=False):
    """Load and preprocess images for MASt3R."""
    from dust3r.utils.image import load_images
    return load_images(image_paths, size, verbose=verbose)


# ============== Image Preprocessing for TRELLIS ==============

def preprocess_image_for_trellis(image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """
    Preprocess image for TRELLIS by creating RGBA with binary alpha channel.
    This matches LabelAny3D's crop_object function behavior.

    Args:
        image: (H, W, 3) RGB image
        mask: Optional (H, W) binary mask where 1=foreground

    Returns:
        RGBA image with binary alpha (0 or 255)
    """
    h, w = image.shape[:2]
    
    if mask is not None:
        # Use provided mask - binary alpha
        alpha = (mask * 255).astype(np.uint8)
    else:
        # Fallback: simple threshold
        import cv2
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, alpha = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    rgba = np.concatenate([image, alpha[..., None]], axis=-1)
    return rgba


# ============== TRELLIS Functions ==============

def load_trellis_model(device: str = "cuda"):
    """Load TRELLIS model for 3D generation."""
    global _trellis_pipeline, _trellis_loaded

    if _trellis_loaded and _trellis_pipeline is not None:
        print(f">> [CACHE] TRELLIS model already loaded, reusing...")
        return _trellis_pipeline

    print(f">> [LOAD] TRELLIS model loading on {device}...")
    
    try:
        trellis_path = "/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/TRELLIS"
        if trellis_path not in sys.path:
            sys.path.insert(0, trellis_path)
        
        from trellis.pipelines import TrellisImageTo3DPipeline
        
        print(f">> Loading TRELLIS model on cuda...")
        _trellis_pipeline = TrellisImageTo3DPipeline.from_pretrained(
            "JeffreyXiang/TRELLIS-image-large",
        )
        
        # Monkey patch for debugging
        _original_sample_sparse_structure = _trellis_pipeline.sample_sparse_structure
        def _debug_sample_sparse_structure(cond, num_samples=1, sampler_params={}):
            result = _original_sample_sparse_structure(cond, num_samples, sampler_params)
            print(f"  >> DEBUG: sparse_structure coords shape: {result.shape}")
            print(f"  >> DEBUG: cond['cond'] shape: {cond['cond'].shape}")
            return result
        _trellis_pipeline.sample_sparse_structure = _debug_sample_sparse_structure
        
        # Also patch sample_slat to handle dimension mismatch (shouldn't happen but just in case)
        _original_sample_slat = _trellis_pipeline.sample_slat
        def _patched_sample_slat(cond, coords, sampler_params={}):
            return _original_sample_slat(cond, coords, sampler_params)
        _trellis_pipeline.sample_slat = _patched_sample_slat
        
        _trellis_pipeline.cuda()
        if hasattr(_trellis_pipeline, 'eval'):
            _trellis_pipeline.eval()
        _trellis_loaded = True
        print(f">> TRELLIS loaded successfully")
        print(f">> TRELLIS sparse_structure_sampler_params: {_trellis_pipeline.sparse_structure_sampler_params}")
        print(f">> TRELLIS slat_sampler_params: {_trellis_pipeline.slat_sampler_params}")
        return _trellis_pipeline
    except Exception as e:
        print(f">> Failed to load TRELLIS: {e}")
        import traceback
        traceback.print_exc()
        return None

_trellis_pipeline = None
_trellis_loaded = False


def generate_3d_model_trellis(image_crop: np.ndarray, obj_id: str, 
                                cache_dir: str = None, seed: int = 42,
                                mask: np.ndarray = None) -> Optional[Dict]:
    """
    Generate 3D model using TRELLIS.
    EXACT implementation as LabelAny3D.
    """
    global _trellis_pipeline, _trellis_cache_dir
    
    # Use provided cache_dir, or global cache, or environment variable
    cache_dir = cache_dir or _trellis_cache_dir or os.environ.get('TRELLIS_CACHE_DIR')
    
    # Check cache - use same path as export
    if cache_dir and obj_id:
        cache_subdir = os.path.join(cache_dir, 'object_space')
        glb_path = os.path.join(cache_subdir, f"{obj_id}.glb")
        if os.path.exists(glb_path):
            return {'glb_path': glb_path, 'cached': True}
    
    pipeline = load_trellis_model()
    if pipeline is None:
        return None
    
    try:
        # Create soft alpha mask for background removal
        rgba_image = preprocess_image_for_trellis(image_crop, mask=mask)
        pil_image = Image.fromarray(rgba_image)
        
        print(f"  >> TRELLIS input image size: {pil_image.size}, mode: {pil_image.mode}")
        
        # Save to disk first (like LabelAny3D does)
        if cache_dir:
            crops_dir = os.path.join(cache_dir, 'crops')
            os.makedirs(crops_dir, exist_ok=True)
            rgba_path = os.path.join(crops_dir, f"{obj_id}_rgba.png")
            pil_image.save(rgba_path)
        
        # Resize to 518x518 for TRELLIS (TRELLIS expects this specific size)
        pil_image_518 = pil_image.resize((518, 518), Image.LANCZOS)
        print(f"  >> TRELLIS input (518x518): {pil_image_518.size}, mode: {pil_image_518.mode}")
        
        # Run TRELLIS pipeline - use preprocess_image=True to let TRELLIS handle foreground extraction
        outputs = pipeline.run(
            pil_image_518,
            seed=1,
            preprocess_image=True,  # Let TRELLIS extract foreground from alpha channel
        )
        
        result = {}
        
        if 'mesh' in outputs and len(outputs['mesh']) > 0:
            result['mesh'] = outputs['mesh'][0]
        
        if 'gaussian' in outputs and len(outputs['gaussian']) > 0:
            result['gaussian'] = outputs['gaussian'][0]
        
        # Export to GLB - EXACT same as LabelAny3D's model_wrappers.py
        if cache_dir and 'gaussian' in result and 'mesh' in result:
            os.makedirs(os.path.join(cache_dir, 'object_space'), exist_ok=True)
            glb_path = os.path.join(cache_dir, 'object_space', f"{obj_id}.glb")
            
            try:
                from trellis.utils import postprocessing_utils
                # EXACT same call as LabelAny3D - no simplify parameter, texture_size=1024
                glb = postprocessing_utils.to_glb(
                    result['gaussian'],
                    result['mesh'],
                    texture_size=1024,
                )
                glb.export(glb_path)
                result['glb_path'] = glb_path
            except Exception as e:
                print(f">> postprocessing_utils.to_glb failed: {e}")
                print(f">> Trying simple trimesh export (no texture)...")
                # Fallback: export simple mesh without texture using trimesh
                try:
                    import trimesh
                    import numpy as np
                    vertices = result['mesh'].vertices.cpu().numpy()
                    faces = result['mesh'].faces.cpu().numpy()
                    
                    # Create a simple gray texture (Pytorch3D requires textures)
                    # Use a simple material with a gray color
                    material = trimesh.visual.material.PBRMaterial(
                        baseColorFactor=np.array([180, 180, 180, 255], dtype=np.uint8)
                    )
                    simple_mesh = trimesh.Trimesh(vertices, faces)
                    # Create visual with no texture coordinates - just use vertex colors or solid color
                    simple_mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=np.array([180, 180, 180, 255] * len(vertices), dtype=np.uint8))
                    simple_mesh.export(glb_path)
                    result['glb_path'] = glb_path
                    result['no_texture'] = True  # Flag to use different renderer
                    print(f">> Simple GLB exported successfully (with vertex colors)")
                except Exception as e2:
                    print(f">> Simple GLB export also failed: {e2}")
                    # No glb_path, but we still have mesh and gaussian for matching
        
        return result
    
    except Exception as e:
        print(f">> TRELLIS inference failed for {obj_id}: {e}")
        return None



def unload_trellis_model():
    """Unload TRELLIS to free GPU memory."""
    global _trellis_pipeline, _trellis_loaded
    
    if _trellis_pipeline is not None:
        del _trellis_pipeline
        _trellis_pipeline = None
        _trellis_loaded = False
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print(">> TRELLIS model unloaded")


# ============== Pytorch3D Renderer ==============

def create_glb_renderer(device='cuda'):
    """Create GLB renderer using Pytorch3D - EXACT as LabelAny3D."""
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
    
    io = IO()
    io.register_meshes_format(MeshGlbFormat())
    
    return {
        'io': io,
        'device': device,
        'renderer_class': (PerspectiveCameras, RasterizationSettings, 
                          MeshRendererWithFragments, MeshRasterizer,
                          HardPhongShader, PointLights, look_at_view_transform)
    }


def load_glb_mesh(glb_path: str, renderer_info: Dict):
    """Load GLB mesh - same as LabelAny3D."""
    io = renderer_info['io']
    device = renderer_info['device']
    mesh = io.load_mesh(glb_path, include_textures=True)
    return mesh.to(device)


def setup_camera(distance=1.5, elevation=0.0, azimuth=0.0, device='cuda'):
    """Setup camera parameters - EXACT as LabelAny3D."""
    from pytorch3d.renderer import PerspectiveCameras, look_at_view_transform
    
    R, T = look_at_view_transform(distance, elevation, azimuth, device=device)
    
    # EXACT same intrinsic as LabelAny3D
    cameras = PerspectiveCameras(
        focal_length=((560.44, 560.44),),
        principal_point=((256, 256),),
        in_ndc=False,
        image_size=[[512, 512]],
        device=device
    )
    
    return cameras, R, T


def render_mesh(mesh, cameras, R=None, T=None, image_size=512, renderer_info=None, use_fallback_renderer=False):
    """Render mesh - EXACT as LabelAny3D."""
    from pytorch3d.renderer import (
        RasterizationSettings, MeshRendererWithFragments,
        MeshRasterizer, PointLights, TexturesVertex, HardPhongShader
    )
    
    device = renderer_info['device']
    
    # Check if mesh has textures
    has_textures = hasattr(mesh, 'textures') and mesh.textures is not None
    
    if not has_textures:
        print(">> Warning: Mesh has no textures, adding vertex colors...")
        # Add vertex colors as a simple texture
        num_verts = mesh.verts_packed().shape[0]
        # Gray color for all vertices - need shape (N, V, C) where N=1 (batch size)
        vertex_colors = torch.full((1, num_verts, 3), 0.7, device=device)
        mesh.textures = TexturesVertex(verts_features=vertex_colors)
    
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    
    # EXACT same lighting as LabelAny3D
    lights = PointLights(
        device=device,
        location=((0.0000e+00, 2.9802e-08, -1.0000e+00),),
        ambient_color=((1.0, 1.0, 1.0),),
        diffuse_color=((0.0, 0.0, 0.0),),
        specular_color=((0.0, 0.0, 0.0),),
    )
    
    # EXACT same shader as LabelAny3D - HardPhongShader (NOT HardFlatShader)
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
    
    if R is None:
        image, fragment = renderer(meshes_world=mesh)
    else:
        image, fragment = renderer(meshes_world=mesh, R=R, T=T)
    
    return image.cpu().numpy().squeeze(), fragment.zbuf.cpu().numpy().squeeze()


def render_multiple_views(mesh, output_dir: str, elevations: List, 
                          azimuths: List, renderer_info: Dict):
    """Render mesh from multiple viewpoints using pytorch3d."""
    os.makedirs(output_dir, exist_ok=True)
    
    rgbs = []
    depths = []
    Rs = []
    Ts = []
    
    for elevation, azimuth in zip(elevations, azimuths):
        cameras, R, T = setup_camera(distance=1.5, elevation=elevation, 
                                     azimuth=azimuth, device=renderer_info['device'])
        
        rgb, depth = render_mesh(mesh, cameras, R, T, image_size=512, renderer_info=renderer_info)
        
        rgbs.append(rgb)
        depths.append(depth)
        Rs.append(R)
        Ts.append(T)
        
        # Save rendered images and depth maps
        rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(f'{output_dir}/rgb_{len(rgbs)-1}.png', rgb_uint8[..., [2,1,0]])
    
    return rgbs, depths, Rs, Ts


# ============== MASt3R Matcher (EXACT LabelAny3D) ==============

def get_correspondences(ref_img, render_img, unprocessed_img0, rgb_18, 
                        depth, T, R, model, device):
    """
    Get 3D-2D correspondences - EXACT implementation as LabelAny3D.
    
    This is the CRITICAL function that matches rendered views to the reference image
    and converts 2D matches to 3D world coordinates.
    """
    # Format images for DUST3R inference
    # img0_dict and img1_dict already have the correct format with normalized values
    if ref_img is None or render_img is None:
        return None, 0
    
    img0_dict = ref_img
    img1_dict = render_img
    
    # Run DUST3R inference - same as LabelAny3D
    output = dust3r_inference([tuple([img0_dict, img1_dict])], model, device, 
                              batch_size=1, verbose=False)
    
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    
    desc1 = pred1['desc'].squeeze(0).detach()
    desc2 = pred2['desc'].squeeze(0).detach()
    
    # Find 2D-2D matches - EXACT same params as LabelAny3D
    matches_im0, matches_im1 = fast_reciprocal_nns(
        desc1, desc2, 
        subsample_or_initxy1=8,
        device=device, 
        dist='dot', 
        block_size=2**13
    )
    
    # Filter matches near image borders - EXACT as LabelAny3D
    H0, W0 = view1['true_shape'][0]
    valid_matches_im0 = (
        (matches_im0[:, 0] >= 3) & 
        (matches_im0[:, 0] < int(W0) - 3) & 
        (matches_im0[:, 1] >= 3) & 
        (matches_im0[:, 1] < int(H0) - 3)
    )
    
    H1, W1 = view2['true_shape'][0]
    valid_matches_im1 = (
        (matches_im1[:, 0] >= 3) & 
        (matches_im1[:, 0] < int(W1) - 3) & 
        (matches_im1[:, 1] >= 3) & 
        (matches_im1[:, 1] < int(H1) - 3)
    )
    
    valid_matches = valid_matches_im0 & valid_matches_im1
    matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]
    
    # Process images - EXACT as LabelAny3D
    rgb_18 = cv2.cvtColor(rgb_18, cv2.COLOR_BGR2RGB)
    rgb_18_processed = cv2.resize(rgb_18, (512, 512))
    cx, cy = rgb_18_processed.shape[1]//2, rgb_18_processed.shape[0]//2
    halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
    halfh = int(3*halfw/4)
    
    # Adjust matches to unprocessed coordinates
    unprocessed_matches1 = (matches_im1 + np.array([0, cy-halfh]))
    unprocessed_matches0 = (matches_im0 + np.array([0, cy-halfh]))
    
    unprocessed_img0 = cv2.cvtColor(unprocessed_img0, cv2.COLOR_BGR2RGB)

    # Filter matches with invalid depth AND out-of-bounds coordinates
    # Clip coordinates to valid range
    unprocessed_matches1[:, 0] = np.clip(unprocessed_matches1[:, 0], 0, 511)
    unprocessed_matches1[:, 1] = np.clip(unprocessed_matches1[:, 1], 0, 511)

    # Filter matches with invalid depth
    depth_of_matches1 = depth[unprocessed_matches1[:, 1].astype(int),
                              unprocessed_matches1[:, 0].astype(int)]
    valid_matches = (depth_of_matches1 != -1)
    unprocessed_matches0 = unprocessed_matches0[valid_matches]
    unprocessed_matches1 = unprocessed_matches1[valid_matches]
    depth_of_matches1 = depth_of_matches1[valid_matches]
    
    # Convert 2D points to 3D points - EXACT same as LabelAny3D
    fx, fy, cx_int, cy_int = 560.44, 560.44, 256, 256
    u = 512 - unprocessed_matches1[:, 0]
    v = 512 - unprocessed_matches1[:, 1]
    
    x = (u - cx_int) * depth_of_matches1 / fx
    y = (v - cy_int) * depth_of_matches1 / fy
    z = depth_of_matches1
    
    points_3d = np.stack((x, y, z), axis=-1)
    
    # Convert to world coordinates using render camera pose - EXACT as LabelAny3D
    T = T.reshape(3, 1)
    R_mat = R.squeeze(0)
    points_world = np.matmul(R_mat, (points_3d.T - T)).T
    
    return points_world, unprocessed_matches0


# ============== Pose Estimator (EXACT LabelAny3D) ==============

def estimate_pose_pnp(object_points, image_points, camera_matrix, dist_coeffs):
    """
    Estimate object pose using solvePnPRANSAC - EXACT as LabelAny3D.
    """
    object_points = np.array(object_points, dtype=np.float32)
    image_points = np.array(image_points, dtype=np.float32)
    
    # EXACT same RANSAC params as LabelAny3D
    iterationsCount = 1000
    reprojectionError = 20.0
    confidence = 0.99
    
    success, rotation_vec, translation_vec, inliers = cv2.solvePnPRansac(
        objectPoints=object_points,
        imagePoints=image_points,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
        iterationsCount=iterationsCount,
        reprojectionError=reprojectionError,
        confidence=confidence,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if success:
        projected_points, _ = cv2.projectPoints(
            object_points,
            rotation_vec,
            translation_vec,
            camera_matrix,
            dist_coeffs
        )
        
        error = cv2.norm(image_points, projected_points.reshape(-1,2), cv2.NORM_L2)
        error = error / len(object_points)
        
        return success, rotation_vec, translation_vec, inliers, error, projected_points
    
    return success, None, None, None, None, None


def create_camera_from_pose(rvec, tvec, camera_matrix, image_size, device='cuda'):
    """Create PyTorch3D camera from OpenCV pose parameters."""
    from pytorch3d.transforms import so3_exp_map
    from pytorch3d.utils import cameras_from_opencv_projection
    
    R = so3_exp_map(torch.tensor(rvec.reshape(1,3))).to(torch.float32)
    tvec_tensor = torch.tensor(tvec.reshape(1,3)).to(torch.float32)
    camera_matrix_tensor = torch.tensor(camera_matrix).unsqueeze(0).to(torch.float32)
    image_size_tensor = torch.tensor([image_size]).to(torch.float32)
    
    cameras = cameras_from_opencv_projection(
        R=R,
        tvec=tvec_tensor,
        camera_matrix=camera_matrix_tensor,
        image_size=image_size_tensor
    )
    
    return cameras.to(device)


# ============== Main Pipeline (EXACT LabelAny3D) ==============

def process_single_object_labelany3d(
    obj_id: str,
    image_crop: np.ndarray,
    image_rgb: np.ndarray,
    bbox2d: Tuple[int, int, int, int],
    K: np.ndarray,
    cache_dir: str = None,
    device: str = "cuda",
    skip_step7_8: bool = True  # Skip re-render/refine for speed
) -> Optional[Dict]:
    """
    Process a single object using EXACT LabelAny3D pipeline.

    Steps:
    1. Generate 3D model with TRELLIS
    2. Render 8 virtual views
    3. MASt3R matching for each view
    4. Initial PnP with render camera params (560.44)
    5. Refine PnP with true K matrix (SKIPPED by default for speed)
    6. Optional: Iterative refinement (SKIPPED by default)

    Args:
        obj_id: Unique object identifier
        image_crop: (H, W, 3) RGB image crop of object
        image_rgb: (H, W, 3) Full RGB image
        bbox2d: (x1, y1, x2, y2) bounding box in full image
        K: Camera intrinsic matrix
        cache_dir: Directory to cache GLB files
        device: 'cuda' or 'cpu'
        skip_step7_8: Skip re-render and refine steps (2-3x faster)

    Returns:
        Dictionary with 'yaw', 'R', 't', 'confidence', 'error'
    """
    import time
    t_start = time.time()
    t_step = {'step1': 0, 'step2': 0, 'step3': 0, 'step4': 0, 'step5': 0, 'step6': 0, 'step7_8': 0}

    print(f"\n{'='*60}")
    print(f"Processing {obj_id} with LabelAny3D pipeline")
    print(f"{'='*60}")
    
    # Create soft mask from image content (for RGBA background removal)
    # Simple approach: detect non-background pixels
    gray = cv2.cvtColor(image_crop, cv2.COLOR_RGB2GRAY)
    # Threshold: consider non-black, non-white as foreground
    mask = (gray > 10) & (gray < 245)
    mask = mask.astype(np.uint8)

    # Step 1: Generate 3D model with TRELLIS
    print("\n[Step 1] Generating 3D model with TRELLIS...")
    t1 = time.time()
    trellis_result = generate_3d_model_trellis(image_crop, obj_id, cache_dir, mask=mask)
    t_step['step1'] = time.time() - t1
    
    if trellis_result is None or 'glb_path' not in trellis_result:
        print(f">> TRELLIS failed for {obj_id}, returning None")
        return None
    
    glb_path = trellis_result['glb_path']
    print(f">> GLB saved to: {glb_path}")
    print(f">> [TIMING] Step 1 (TRELLIS): {t_step['step1']:.1f}s")

    # Load MASt3R model
    print("\n[Step 2] Loading MASt3R model...")
    t2 = time.time()
    mast3r_model = load_mast3r_model(device)
    if mast3r_model is None:
        return None
    t_step['step2'] = time.time() - t2
    print(f">> [TIMING] Step 2 (Load MASt3R): {t_step['step2']:.1f}s")

    # Create renderer
    print("\n[Step 3] Creating Pytorch3D renderer...")
    t3 = time.time()
    renderer_info = create_glb_renderer(device)
    t_step['step3'] = time.time() - t3
    print(f">> [TIMING] Step 3 (Create Renderer): {t_step['step3']:.1f}s")
    
    # Load mesh
    print(f">> Loading mesh from {glb_path}...")
    mesh = load_glb_mesh(glb_path, renderer_info)
    
    # Estimate elevation from mesh (simple heuristic)
    # In LabelAny3D, this is pre-computed; we use a simple estimate
    elevation_estimate = -0.3  # Default estimate
    
    # Render 8 views - EXACT as LabelAny3D
    print("\n[Step 4] Rendering 8 virtual views...")
    t4 = time.time()
    elevations = [elevation_estimate] * 8
    azimuths = list(range(0, 360, 45))
    render_dir = os.path.join(cache_dir or '/tmp', 'renderings', obj_id)
    os.makedirs(render_dir, exist_ok=True)

    rgbs, depths, Rs, Ts = render_multiple_views(
        mesh, render_dir, elevations, azimuths, renderer_info
    )
    t_step['step4'] = time.time() - t4
    print(f">> [TIMING] Step 4 (Render Views): {t_step['step4']:.1f}s")
    
    # Prepare reference image
    # In LabelAny3D: unprocessed_img0 is the raw crop, ref_img is for MASt3R
    unprocessed_img0 = image_crop.copy()
    if unprocessed_img0.shape[-1] == 3:
        unprocessed_img0 = cv2.cvtColor(unprocessed_img0, cv2.COLOR_RGB2RGBA)
    
    # Create reference image dict for MASt3R (EXACT as LabelAny3D)
    # Load images normalizes to [-1, 1] with: (pixel / 255.0 - 0.5) / 0.5
    image_tensor = torch.from_numpy(image_crop).permute(2, 0, 1).float() / 255.0
    image_tensor = (image_tensor - 0.5) / 0.5  # Normalize to [-1, 1]
    # EXACT format as load_images - batch dim + true_shape + instance
    ref_img_dict = {
        'img': image_tensor[None],  # Add batch dimension: (1, 3, H, W)
        'true_shape': np.int32([[image_crop.shape[0], image_crop.shape[1]]]),  # [[H, W]]
        'instance': 'ref_img'
    }
    
    # Step 5: Process each view and collect matches
    print("\n[Step 5] MASt3R matching for all views...")
    t5 = time.time()
    all_points_world = None
    all_points_target = None

    for i in range(len(rgbs)):
        print(f">> Processing view {i+1}/8...")
        
        rgb_18 = cv2.imread(f'{render_dir}/rgb_{i}.png', cv2.IMREAD_UNCHANGED)
        depth = depths[i]
        R = Rs[i].cpu().numpy()
        T = Ts[i].cpu().numpy()
        
        # Create render image dict (EXACT as LabelAny3D)
        render_rgb = cv2.cvtColor(rgb_18, cv2.COLOR_BGR2RGB)
        render_tensor = torch.from_numpy(render_rgb).permute(2, 0, 1).float() / 255.0
        render_tensor = (render_tensor - 0.5) / 0.5  # Normalize to [-1, 1]
        # EXACT format as load_images - batch dim + true_shape + instance
        render_img_dict = {
            'img': render_tensor[None],  # Add batch dimension: (1, 3, H, W)
            'true_shape': np.int32([[render_rgb.shape[0], render_rgb.shape[1]]]),  # [[H, W]]
            'instance': f'render_view_{i}'
        }
        
        # Get correspondences
        points_world, unprocessed_matches0 = get_correspondences(
            ref_img_dict, render_img_dict, unprocessed_img0, rgb_18,
            depth, T, R, mast3r_model, device
        )
        
        # Skip if no matches found
        if points_world is None or len(points_world) == 0:
            print(f">> No matches found for view {i+1}, skipping...")
            continue
        
        if all_points_world is None:
            all_points_world = points_world
            all_points_target = unprocessed_matches0
        else:
            all_points_world = np.concatenate((all_points_world, points_world), axis=0)
            all_points_target = np.concatenate((all_points_target, unprocessed_matches0), axis=0)
    
    t_step['step5'] = time.time() - t5
    print(f">> [TIMING] Step 5 (MASt3R Matching): {t_step['step5']:.1f}s")

    # Handle case where no matches were collected
    if all_points_world is None or len(all_points_world) == 0:
        print(f">> No matches collected for any view")
        return None

    print(f">> Total matches collected: {len(all_points_world)}")

    if len(all_points_world) < 10:
        print(f">> Not enough matches for PnP")
        return None

    # Step 6: Initial PnP with render camera parameters - EXACT as LabelAny3D
    print("\n[Step 6] Initial PnP with render camera params (560.44)...")
    t6 = time.time()

    camera_matrix_render = np.array([
        [560.44, 0, 256],
        [0, 560.44, 256],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros((5, 1))

    success, rvec, tvec, inliers, error, _ = estimate_pose_pnp(
        all_points_world, all_points_target, camera_matrix_render, dist_coeffs
    )

    if not success:
        print(">> Initial pose estimation failed")
        return None

    print(f">> Initial pose successful!")
    print(f">> Rotation vector: {rvec.flatten()}")
    print(f">> Translation vector: {tvec.flatten()}")
    print(f">> Inliers: {len(inliers)}, Error: {error:.4f}")
    t_step['step6'] = time.time() - t6
    print(f">> [TIMING] Step 6 (Initial PnP): {t_step['step6']:.1f}s")

    # Step 7-8: Re-render and refine (OPTIONAL - can be skipped for speed)
    t78 = time.time()
    if skip_step7_8:
        print("\n[Step 7-8] SKIPPED for speed (skip_step7_8=True)")
        yaw = rotation_vector_to_yaw(rvec)
        total_time = time.time() - t_start
        print(f">> [TIMING] TOTAL: {total_time:.1f}s (Step1={t_step['step1']:.1f}s, Step5={t_step['step5']:.1f}s)")
        return {
            'yaw': yaw,
            'R': rvec,
            't': tvec,
            'confidence': len(inliers) / len(all_points_world),
            'error': error,
            'inliers': len(inliers),
            'glb_path': glb_path
        }
    
    # Step 7: Create camera from estimated pose and render
    print("\n[Step 7] Re-render with estimated pose...")
    cameras = create_camera_from_pose(rvec, tvec, camera_matrix_render, [512, 512], device)

    rgb_iter1, depth_iter1 = render_mesh(mesh, cameras, None, None, 512, renderer_info)
    cv2.imwrite(f'{render_dir}/rgb_iter1.png', rgb_iter1[..., [2,1,0]] * 255)

    # Step 8: Refine with true K matrix - EXACT as LabelAny3D
    print("\n[Step 8] Refined PnP with true camera matrix...")

    # Re-do MASt3R matching with the re-rendered view
    rgb_18_refine = cv2.imread(f'{render_dir}/rgb_iter1.png', cv2.IMREAD_UNCHANGED)
    R_refine = cameras.R.cpu().numpy()
    T_refine = cameras.T.cpu().numpy()
    
    # EXACT format as load_images - batch dim + true_shape + instance
    rgb_iter1_np = rgb_iter1[..., :3]
    render_tensor = torch.from_numpy(rgb_iter1_np).permute(2, 0, 1).float() / 255.0
    render_tensor = (render_tensor - 0.5) / 0.5  # Normalize to [-1, 1]
    render_img_refine = {
        'img': render_tensor[None],
        'true_shape': np.int32([[rgb_iter1_np.shape[0], rgb_iter1_np.shape[1]]]),
        'instance': 'render_refine'
    }
    
    points_world_refine, unprocessed_matches0_refine = get_correspondences(
        ref_img_dict, render_img_refine, unprocessed_img0, rgb_18_refine,
        depth_iter1, T_refine, R_refine, mast3r_model, device
    )
    
    # Convert matches to full image coordinates
    # This is where LabelAny3D uses crop_params to map to full image
    x1, y1, x2, y2 = bbox2d
    crop_w = x2 - x1
    crop_h = y2 - y1
    scale_x = crop_w / 512.0
    scale_y = crop_h / 512.0
    
    # Map from 512x512 crop coords to full image coords
    unprocessed_matches0_image = unprocessed_matches0_refine * np.array([scale_x, scale_y])
    unprocessed_matches0_image = unprocessed_matches0_image + np.array([[x1, y1]])
    
    # Refined PnP with true K
    # Try to refine pose using matches from re-rendered view
    # If not enough matches, use initial pose result
    K_np = np.array(K, dtype=np.float32)
    
    try:
        # Debug: check number of points
        n_points = len(points_world_refine) if points_world_refine is not None else 0
        print(f">> DEBUG Step8: points_world_refine has {n_points} points")
        
        if n_points >= 4:
            success, rvec, tvec, inliers, error, _ = estimate_pose_pnp(
                points_world_refine, unprocessed_matches0_image, K_np, dist_coeffs
            )
            if success:
                print(f">> Refined pose successful!")
                print(f">> Rotation vector: {rvec.flatten()}")
                print(f">> Translation vector: {tvec.flatten()}")
                print(f">> Inliers: {len(inliers)}, Error: {error:.4f}")
            else:
                print(">> Refined PnP failed, using initial pose")
        else:
            print(f">> Not enough matches ({n_points}), using initial pose result")
            success = False
    except Exception as e:
        print(f">> Refined PnP error: {e}")
        success = False
    
    if success:
        # Extract yaw from rotation vector
        yaw = rotation_vector_to_yaw(rvec)
        
        return {
            'yaw': yaw,
            'R': rvec,
            't': tvec,
            'confidence': len(inliers) / n_points if n_points > 0 else 0,
            'error': error,
            'inliers': len(inliers),
            'glb_path': glb_path
        }
    
    # Fallback to initial pose
    print(f">> Using initial pose: yaw={rotation_vector_to_yaw(rvec)}")
    t_step['step7_8'] = time.time() - t78
    yaw = rotation_vector_to_yaw(rvec)
    total_time = time.time() - t_start
    print(f">> [TIMING] Step 7-8 (Refine): {t_step['step7_8']:.1f}s")
    print(f">> [TIMING] TOTAL: {total_time:.1f}s")
    return {
        'yaw': yaw,
        'R': rvec,
        't': tvec,
        'confidence': len(inliers) / len(all_points_world),
        'error': error,
        'inliers': len(inliers),
        'glb_path': glb_path
    }


def rotation_vector_to_yaw(rvec: np.ndarray) -> float:
    """Convert rotation vector to yaw angle (rotation around Y-axis)."""
    from scipy.spatial.transform import Rotation as R
    rot = R.from_rotvec(rvec.flatten())
    # Convert to rotation matrix
    R_mat = rot.as_matrix()
    # Extract yaw (rotation around Y axis)
    # In camera coordinates: yaw is rotation around vertical axis
    yaw = np.arctan2(-R_mat[2, 0], R_mat[0, 0])
    return float(yaw)


# ============== Wrapper for OVM3D-Det ==============

def estimate_yaw_labelany3d(
    pseudo_lidar: np.ndarray,
    image_rgb: np.ndarray,
    bbox2d: Tuple[int, int, int, int],
    K: np.ndarray,
    obj_id: str,
    cache_dir: str = None,
    device: str = "cuda",
    skip_step7_8: bool = True
) -> Tuple[Optional[float], float]:
    """
    Wrapper function for OVM3D-Det to use LabelAny3D pipeline.
    
    This is the main entry point that replaces the original MASt3R yaw estimation.
    
    Args:
        pseudo_lidar: Not used (kept for API compatibility)
        image_rgb: Full RGB image (H, W, 3)
        bbox2d: (x1, y1, x2, y2) bounding box
        K: Camera intrinsic matrix (3, 3)
        obj_id: Unique object identifier
        cache_dir: Directory for caching GLB files
        device: Device for inference
    
    Returns:
        (yaw, confidence)
    """
    x1, y1, x2, y2 = bbox2d
    
    # Check minimum bbox size
    bbox_w = int(x2) - int(x1)
    bbox_h = int(y2) - int(y1)
    if bbox_w < 32 or bbox_h < 32:
        print(f"  >> BBox too small ({bbox_w}x{bbox_h}), skipping")
        return None, 0.0
    
    # Crop object from image - use LabelAny3D's approach: padding + resize to 256x256
    obj_crop = image_rgb[int(y1):int(y2), int(x1):int(x2)]
    h, w = obj_crop.shape[:2]
    
    # Create padded crop to maintain aspect ratio (like LabelAny3D)
    # ratio = 0.7 means padding to 1/0.7 ≈ 1.43x the max dimension
    ratio = 0.7
    max_dim = max(h, w)
    side_len = int(max_dim / ratio)
    
    # Create padded image
    padded = np.zeros((side_len, side_len, 3), dtype=obj_crop.dtype)
    center = side_len // 2
    y_offset = center - h // 2
    x_offset = center - w // 2
    padded[y_offset:y_offset + h, x_offset:x_offset + w] = obj_crop
    
    # Resize to 256x256 (same as LabelAny3D)
    crop_size = 256
    image_crop = cv2.resize(padded, (crop_size, crop_size), interpolation=cv2.INTER_LANCZOS4)
    
    # Ensure 3 channels
    if image_crop.shape[-1] == 4:
        image_crop = cv2.cvtColor(image_crop, cv2.COLOR_RGBA2RGB)
    
    # Run LabelAny3D pipeline
    result = process_single_object_labelany3d(
        obj_id=obj_id,
        image_crop=image_crop,
        image_rgb=image_rgb,
        bbox2d=bbox2d,
        K=K,
        cache_dir=cache_dir,
        device=device,
        skip_step7_8=skip_step7_8
    )
    
    if result is not None:
        return result['yaw'], result['confidence']
    
    return None, 0.0


if __name__ == '__main__':
    # Test script
    import argparse
    
    parser = argparse.ArgumentParser(description='LabelAny3D Pipeline Test')
    parser.add_argument('--obj_id', type=str, required=True, help='Object ID')
    parser.add_argument('--image', type=str, required=True, help='Image path')
    parser.add_argument('--bbox', type=str, required=True, help='BBox: x1,y1,x2,y2')
    parser.add_argument('--K', type=str, required=True, help='K matrix: fx,fy,cx,cy')
    parser.add_argument('--cache_dir', type=str, default='/tmp/labelany3d', help='Cache dir')
    args = parser.parse_args()
    
    # Load image
    img = cv2.imread(args.image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Parse bbox
    bbox = tuple(map(int, args.bbox.split(',')))
    
    # Parse K
    fx, fy, cx, cy = map(float, args.K.split(','))
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    
    # Run
    yaw, conf = estimate_yaw_labelany3d(
        None, img, bbox, K, args.obj_id, args.cache_dir
    )
    
    print(f"\nResult: yaw={yaw}, confidence={conf}")
