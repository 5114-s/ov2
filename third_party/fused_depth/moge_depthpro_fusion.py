# Copyright (c) Teacher-Student Distillation Pipeline
# MoGe + Depth Pro fusion depth predictor.
# Outputs (H, W) metric depth arrays compatible with the UniDepth .npy format.
import torch
import numpy as np
from typing import Dict, Optional
from PIL import Image
import os
import sys

# Compute project root: walk up from this file until we find 'external/MoGe'.
# This works regardless of whether the module is imported or run as a script.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Walk up from this file: fused_depth/ -> third_party/ -> project root
for _candidate in [
    os.path.dirname(os.path.dirname(_THIS_DIR)),  # /data/ZhaoX/OVM3D-Det
    os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR))),  # parent of project root
    os.getcwd(),  # current working directory (for script invocation)
]:
    _resolved = _candidate if os.path.isabs(_candidate) else os.path.join(os.getcwd(), _candidate)
    if os.path.exists(os.path.join(_resolved, 'external', 'MoGe')):
        _PROJECT_ROOT = _resolved
        break
else:
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))  # fallback

def _get_project_root():
    return _PROJECT_ROOT

# --------------------------------------------------------------------------- #
# MoGe loader
# --------------------------------------------------------------------------- #
class MoGeLoader:
    _instance = None
    _model = None

    def __init__(self, device=None):
        self.model = None
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device) if isinstance(device, str) else device

    @classmethod
    def get_instance(cls, device=None):
        if cls._instance is None or (device is not None and str(cls._instance.device) != str(device)):
            cls._instance = cls(device=device)
        return cls._instance

    def load_model(self):
        if self.model is not None:
            return self.model
        project_root = _get_project_root()
        moge_base = os.path.join(project_root, "external", "MoGe")
        if os.path.exists(moge_base):
            # CRITICAL: Initialize the local utils3d shim BEFORE importing moge_model.
            # This registers sys.modules['utils3d'] so that 'import utils3d' inside
            # moge_model.py resolves to our shim (which has .torch submodule),
            # not the global pip package (which does not).
            moge_utils3d = os.path.join(moge_base, "moge", "utils3d.py")
            if os.path.exists(moge_utils3d):
                import importlib.util
                spec = importlib.util.spec_from_file_location("utils3d", moge_utils3d)
                shim = importlib.util.module_from_spec(spec)
                sys.modules['utils3d'] = shim
                spec.loader.exec_module(shim)
            sys.path.insert(0, moge_base)
            sys.path.insert(0, os.path.join(moge_base, "moge"))
        try:
            from moge.model.moge_model import MoGeModel
            self.model = MoGeModel.from_pretrained("Ruicheng/moge-vitl")
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f">> MoGe loaded (device={self.device})")
            return self.model
        except Exception as e:
            print(f"Warning: MoGe load failed: {e}")
            return None

    def infer(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        if self.model is None:
            self.load_model()
        if self.model is None:
            return None
        H, W = image.shape[:2]
        image_tensor = torch.tensor(
            image / 255.0, dtype=torch.float32, device=self.device
        ).permute(2, 0, 1)
        with torch.no_grad():
            output = self.model.infer(image_tensor)
        depth = output["depth"]
        mask = output.get("mask", None)
        intrinsics = output["intrinsics"]
        if depth.dim() == 4:
            depth = depth.squeeze(0)
        if depth.dim() == 3:
            depth = depth.squeeze(0)
        depth = depth.detach().float().cpu().numpy()
        result = {
            "depth": depth,
            "intrinsics": intrinsics.squeeze().cpu().numpy(),
        }
        if mask is not None:
            if mask.dim() == 4:
                mask = mask.squeeze(0)
            if mask.dim() == 3:
                mask = mask.squeeze(0)
            if torch.is_tensor(mask):
                mask = mask.detach().float().cpu().numpy()
            mask = np.atleast_1d(np.asarray(mask)).flatten()
            if mask.size == H * W:
                mask = mask.reshape(H, W)
            result["mask"] = mask
        W2, H2 = image.shape[1], image.shape[0]
        result["intrinsics"] = result["intrinsics"] * np.array([
            [W2, 1, W2], [1, H2, H2], [1, 1, 1]
        ])
        return result


# --------------------------------------------------------------------------- #
# Depth Pro loader
# --------------------------------------------------------------------------- #
class DepthProLoader:
    _instance = None
    _model = None
    _transform = None

    def __init__(self, device=None):
        self.model = None
        self.transform = None
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device) if isinstance(device, str) else device

    @classmethod
    def get_instance(cls, device=None):
        if cls._instance is None or (device is not None and str(cls._instance.device) != str(device)):
            cls._instance = cls(device=device)
        return cls._instance

    def load_model(self):
        if self.model is not None:
            return self.model, self.transform
        precision = torch.float16 if self.device.type == "cuda" else torch.float32
        project_root = _get_project_root()
        ml_dp_path = os.path.join(project_root, "external", "ml-depth-pro", "src")
        if os.path.exists(ml_dp_path):
            sys.path.insert(0, ml_dp_path)
            os.chdir(ml_dp_path)
        try:
            import depth_pro
            from depth_pro.depth_pro import DepthProConfig
            default_ckpt = os.path.join(project_root, "external", "checkpoints", "depth_pro.pt")
            config = DepthProConfig(
                patch_encoder_preset="dinov2l16_384",
                image_encoder_preset="dinov2l16_384",
                decoder_features=256,
                checkpoint_uri=None,  # Load manually after model creation
                use_fov_head=True,
                fov_encoder_preset="dinov2l16_384",
            )
            model, transform = depth_pro.create_model_and_transforms(
                config=config, device=self.device, precision=precision,
            )
            # Manually load full weights including fov head (same as test_depthpro_fixed2.py)
            state_dict = torch.load(default_ckpt, map_location=self.device)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            self.model = model
            self.transform = transform
            print(f">> DepthPro loaded (device={self.device}, precision={precision}, missing={len(missing)}, unexpected={len(unexpected)})")
            return model, transform
        except Exception as e:
            print(f"Warning: DepthPro load failed: {e}")
            return None, None

    def infer(self, image: np.ndarray, focal_length_px: float = None) -> Dict[str, np.ndarray]:
        if self.model is None or self.transform is None:
            self.load_model()
        if self.model is None:
            return None
        image_pil = Image.fromarray(image)
        W, H = image_pil.size
        image_tensor = self.transform(image_pil)
        with torch.no_grad():
            f_px_tensor = torch.tensor(focal_length_px, dtype=torch.float32, device=self.device) if focal_length_px is not None else None
            prediction = self.model.infer(image_tensor, f_px=f_px_tensor)
        result_f_px = prediction.get("focallength_px", focal_length_px)
        if torch.is_tensor(result_f_px) and result_f_px.numel() == 1:
            result_f_px = result_f_px.item()
        raw_depth = prediction["depth"]
        if torch.is_tensor(raw_depth):
            depth_np = raw_depth.detach().float().cpu().numpy()
        elif isinstance(raw_depth, np.ndarray):
            depth_np = np.asarray(raw_depth)
        else:
            depth_np = np.array(raw_depth)
        depth_np = np.atleast_1d(depth_np).flatten().reshape(H, W)
        return {"depth": depth_np, "focallength_px": result_f_px}


# --------------------------------------------------------------------------- #
# RANSAC alignment (from moge to depthpro scale)
# --------------------------------------------------------------------------- #
def align_depth_ransac(
    relative_depth: np.ndarray,
    metric_depth: np.ndarray,
    mask: np.ndarray = None,
    max_valid_depth: float = 400.0,
    verbose: bool = True,
) -> tuple:
    """
    Align MoGe scale-invariant depth to Depth Pro metric depth via RANSAC.
    Fits: metric = scale * relative (no intercept).
    Output: aligned_depth = relative * scale.
    """
    from sklearn.linear_model import RANSACRegressor, LinearRegression

    rel = np.atleast_1d(np.asarray(relative_depth)).flatten()
    met = np.atleast_1d(np.asarray(metric_depth)).flatten()

    H, W = relative_depth.shape[-2], relative_depth.shape[-1]
    expected = H * W
    if rel.shape[0] != expected:
        if verbose:
            print(f"Warning: flatten length {rel.shape[0]} != expected {expected}, skipping fusion")
        return metric_depth, {"scale": 1.0, "status": "failed"}

    # NOTE: data must stay in the SAME order as flatten() produced.
    # The depth arrays come from inference with shape (H, W) flattened in C-order.
    # Using reshape(H, W) without order= is fine as long as it matches how the
    # array was originally flattened. Flatten first, then reshape to avoid any
    # ambiguity with row/col major order.
    rel = rel.reshape(H, W)
    met = met.reshape(H, W)

    rel_flat = rel.flatten().astype(np.float64)
    met_flat = met.flatten().astype(np.float64)

    valid = (
        ~np.isinf(rel_flat)
        & np.isfinite(met_flat)
        & (met_flat > 0.01)
        & (met_flat < max_valid_depth)
    )
    if mask is not None:
        mask_valid = mask.flatten().astype(np.float64) > 0
        valid &= mask_valid

    rel_v = rel_flat[valid].reshape(-1, 1)
    met_v = met_flat[valid].reshape(-1, 1)

    if valid.sum() == 0:
        if verbose:
            print("Warning: no valid points, returning metric depth")
        return metric_depth, {"scale": 1.0, "status": "failed"}

    if len(rel_v) < 100:
        scale = met_v.mean() / (rel_v.mean() + 1e-6)
        depth = np.full_like(rel, 10000.0)
        depth.flat[valid] = rel_flat[valid] * scale
        if verbose:
            print(f"Warning: few valid points ({len(rel_v)}), mean scale={scale:.4f}")
        return depth, {"scale": float(scale), "status": "warning"}

    # RANSAC: fit metric = scale * relative (X=relative, y=metric)
    ransac = RANSACRegressor(
        estimator=LinearRegression(fit_intercept=False),
        min_samples=0.2,
        random_state=42,
    )
    try:
        ransac.fit(rel_v, met_v)
        scale = ransac.estimator_.coef_[0, 0]
        inlier_mask = ransac.inlier_mask_
    except Exception:
        scale = met_v.mean() / (rel_v.mean() + 1e-6)
        depth = np.full_like(rel, 10000.0)
        depth.flat[valid] = rel_flat[valid] * scale
        if verbose:
            print(f"RANSAC failed, using mean scale={scale:.4f}")
        return depth, {"scale": float(scale), "status": "warning"}

    if not np.isfinite(scale) or scale <= 0:
        scale = met_v.mean() / (rel_v.mean() + 1e-6)
        depth = np.full_like(rel, 10000.0)
        depth.flat[valid] = rel_flat[valid] * scale
        if verbose:
            print(f"Invalid scale, using mean scale={scale:.4f}")
        return depth, {"scale": float(scale), "status": "warning"}

    depth = np.full_like(rel, 10000.0)
    depth.flat[valid] = rel_flat[valid] * scale
    inlier_ratio = float(inlier_mask.sum() / len(rel_v))
    if verbose:
        print(f">> [RANSAC align] scale={scale:.4f}, inliers={inlier_ratio:.1%}, valid={valid.sum()}/{len(rel_flat)}")
    return depth, {"scale": float(scale), "status": "success"}


# --------------------------------------------------------------------------- #
# Single-image inference: MoGe + Depth Pro + RANSAC
# --------------------------------------------------------------------------- #
def infer_fused_depth(
    image: np.ndarray,
    K: np.ndarray = None,
    device: str = None,
    fusion_method: str = "ransac_align",
    verbose: bool = True,
) -> np.ndarray:
    """
    Run MoGe+DepthPro fusion on a single RGB image.

    Args:
        image: (H, W, 3) RGB numpy array, uint8
        K: (3, 3) camera intrinsic matrix
        device: 'cuda' or 'cpu'
        fusion_method: only 'ransac_align' is implemented
        verbose: print diagnostics

    Returns:
        depth: (H, W) metric depth in meters
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    moge_loader = MoGeLoader(device=device)
    dp_loader = DepthProLoader(device=device)

    # Run MoGe
    moge_out = moge_loader.infer(image)
    if moge_out is None:
        raise RuntimeError("MoGe inference failed")

    # Run Depth Pro
    focal_px = float(K[0, 0]) if K is not None else None
    dp_out = dp_loader.infer(image, focal_length_px=focal_px)
    if dp_out is None:
        raise RuntimeError("DepthPro inference failed")

    # Fuse via RANSAC
    depth_moge = moge_out["depth"]
    depth_dp = dp_out["depth"]

    # Resize to match if needed
    if depth_moge.shape != depth_dp.shape:
        import cv2
        depth_dp = cv2.resize(depth_dp, (depth_moge.shape[1], depth_moge.shape[0]), interpolation=cv2.INTER_LINEAR)

    mask = moge_out.get("mask", None)
    fused_depth, diag = align_depth_ransac(depth_moge, depth_dp, mask=mask, verbose=verbose)

    return fused_depth.astype(np.float32)
