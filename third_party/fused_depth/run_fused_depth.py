#!/usr/bin/env python
# Copyright (c) OVM3D-Det
# MoGe + Depth Pro fusion depth generation script (dual-GPU version).
# GPU 0: MoGe  |  GPU 1: DepthPro
# Main process: distributes tasks, runs RANSAC alignment, saves .npy files.
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1 python third_party/fused_depth/run_fused_depth.py --dataset SUNRGBD
#   python third_party/fused_depth/run_fused_depth.py --dataset SUNRGBD --gpu-moge 0 --gpu-dp 1
import torch
import numpy as np
import os
import sys
import argparse
import json
import time
import multiprocessing as mp
from PIL import Image
from tqdm import tqdm
from threading import Thread
from queue import Empty

# --------------------------------------------------------------------------- #
# Project root (set for all processes including spawned workers)
# --------------------------------------------------------------------------- #
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# fused_depth/ -> third_party/ -> project root (OVM3D-Det) = 2 levels up
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
os.environ["PYTHONPATH"] = _PROJECT_ROOT + ":" + os.environ.get("PYTHONPATH", "")


# --------------------------------------------------------------------------- #
# Worker entry points
# --------------------------------------------------------------------------- #
def moge_worker(device_id: int, task_queue: mp.Queue, result_queue: mp.Queue,
                project_root: str):
    """Run on GPU `device_id`. Pull tasks from queue, push depth to result queue."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    import torch
    torch.backends.cudnn.enabled = False
    torch.cuda.set_device(0)  # after CUDA_VISIBLE_DEVICES, only device 0 is visible

    # Note: ensure subprocess uses the same Python env as parent (ovm3d-1)
    _ROOT = project_root
    sys.path.insert(0, _ROOT)
    sys.path.insert(0, os.path.join(_ROOT, "external"))
    sys.path.insert(0, os.path.join(_ROOT, "external", "MoGe"))
    sys.path.insert(0, os.path.join(_ROOT, "external", "MoGe", "moge"))
    from third_party.fused_depth.moge_depthpro_fusion import MoGeLoader

    loader = MoGeLoader(device="cuda:0")
    model = loader.load_model()
    if model is None:
        raise RuntimeError("MoGe model failed to load")

    while True:
        task = task_queue.get()
        if task is None:  # sentinel
            break
        img_id, rgb = task
        try:
            out = loader.infer(rgb)
            depth = out["depth"].astype(np.float32)
            mask = out.get("mask", None)
            result_queue.put((img_id, depth, mask, None))
        except Exception as e:
            result_queue.put((img_id, None, None, str(e)))


def depthpro_worker(device_id: int, task_queue: mp.Queue, result_queue: mp.Queue,
                    focal_length_px: float = None, project_root: str = None):
    """Run on GPU `device_id`. Pull tasks from queue, push depth to result queue."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    import torch
    torch.backends.cudnn.enabled = False
    torch.cuda.set_device(0)  # after CUDA_VISIBLE_DEVICES, only device 0 is visible

    _ROOT = project_root or _PROJECT_ROOT
    sys.path.insert(0, _ROOT)
    sys.path.insert(0, os.path.join(_ROOT, "external"))
    sys.path.insert(0, os.path.join(_ROOT, "external", "MoGe"))
    sys.path.insert(0, os.path.join(_ROOT, "external", "MoGe", "moge"))
    from third_party.fused_depth.moge_depthpro_fusion import DepthProLoader

    loader = DepthProLoader(device="cuda:0")
    model = loader.load_model()
    if model is None:
        raise RuntimeError("DepthPro model failed to load")

    while True:
        task = task_queue.get()
        if task is None:
            break
        img_id, rgb = task
        try:
            out = loader.infer(rgb, focal_length_px=focal_length_px)
            depth = out["depth"].astype(np.float32)
            result_queue.put((img_id, depth, None, None))  # DepthPro doesn't return mask
        except Exception as e:
            result_queue.put((img_id, None, None, str(e)))


# --------------------------------------------------------------------------- #
# RANSAC alignment (same as LabelAny3D original: fit_intercept=True)
# --------------------------------------------------------------------------- #
def align_depth_ransac(relative_depth, metric_depth, mask=None, verbose=False):
    from sklearn.linear_model import RANSACRegressor, LinearRegression

    rel = np.asarray(relative_depth, dtype=np.float64).flatten()
    met = np.asarray(metric_depth, dtype=np.float64).flatten()

    # Same as LabelAny3D: only check inf for relative depth
    valid = (~np.isinf(rel)) & (met < 400.0)
    if mask is not None:
        m = np.asarray(mask, dtype=bool).flatten()
        valid = valid & m

    if valid.sum() == 0:
        print("Warning: No valid points for alignment. Returning metric depth.")
        return metric_depth.astype(np.float32), {"status": "failed"}

    try:
        regressor = RANSACRegressor(
            estimator=LinearRegression(fit_intercept=False),
            min_samples=0.2,
            random_state=42,
        )
        regressor.fit(rel[valid].reshape(-1, 1), met[valid].reshape(-1, 1))
    except Exception as e:
        print(f"Error fitting RANSACRegressor: {e}, using metric depth directly")
        return metric_depth.astype(np.float32), {"status": "failed"}

    # Initialize output depth array with large values (same as LabelAny3D)
    depth = np.full_like(rel, 10000.0)

    # Only predict for masked/valid regions (same as LabelAny3D)
    if mask is not None:
        masked_pred = regressor.predict(rel[valid].reshape(-1, 1)).flatten()
        depth[valid] = masked_pred
    else:
        valid_mask = ~np.isinf(rel)
        masked_pred = regressor.predict(rel[valid_mask].reshape(-1, 1)).flatten()
        depth[valid_mask] = masked_pred

    inlier_ratio = float(regressor.inlier_mask_.sum() / len(regressor.inlier_mask_)) if hasattr(regressor, 'inlier_mask_') and regressor.inlier_mask_ is not None else 0.0
    if verbose:
        scale = regressor.estimator_.coef_[0, 0] if hasattr(regressor.estimator_, 'coef_') else 1.0
        intercept = regressor.estimator_.intercept_ if hasattr(regressor.estimator_, 'intercept_') else 0.0
        print(f">> [RANSAC] scale={scale:.4f}, intercept={intercept:.4f}, inliers={inlier_ratio:.1%}")

    H, W = relative_depth.shape[-2], relative_depth.shape[-1]
    return depth.reshape(H, W).astype(np.float32), {"status": "success"}


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="MoGe+DepthPro fused depth (dual-GPU)")
    parser.add_argument("--dataset", type=str, default="SUNRGBD", help="Dataset name")
    parser.add_argument("--gpu-moge", type=int, default=0, help="GPU for MoGe")
    parser.add_argument("--gpu-dp", type=int, default=1, help="GPU for DepthPro")
    parser.add_argument("--outdir", type=str, default=None, help="Output dir prefix")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip images that already have .npy files")
    return parser.parse_args()


def process(args):
    dataset = args.dataset
    gpu_moge = args.gpu_moge
    gpu_dp = args.gpu_dp
    outdir = args.outdir or os.path.join(_PROJECT_ROOT, "pseudo_label")

    print(f">> MoGe GPU: {gpu_moge}  |  DepthPro GPU: {gpu_dp}")
    print(f">> Dataset: {dataset}")
    print(f">> Output:  {outdir}/{dataset}/{{train,val}}/depth/*.npy")

    # Estimate focal length from first image (use as default for DepthPro)
    focal_px = None
    for mode in ["train", "val"]:
        json_path = os.path.join(_PROJECT_ROOT, "datasets", "Omni3D", f"{dataset}_{mode}.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                data = json.load(f)
            if data["images"]:
                K = np.array(data["images"][0]["K"])
                focal_px = float(K[0, 0])
                print(f">> Focal length: {focal_px:.1f} (from {dataset}_{mode})")
                break

    # Shared queues
    moge_task_queue = mp.Queue()
    dp_task_queue = mp.Queue()
    result_queue = mp.Queue()

    # Start worker processes
    p_moge = mp.Process(target=moge_worker,
                        args=(gpu_moge, moge_task_queue, result_queue, _PROJECT_ROOT))
    p_dp = mp.Process(target=depthpro_worker,
                      args=(gpu_dp, dp_task_queue, result_queue, focal_px, _PROJECT_ROOT))
    p_moge.start()
    p_dp.start()
    print(f">> Workers started (PID: moge={p_moge.pid}, dp={p_dp.pid})")

    processed = 0
    errors = 0

    for mode in ["train", "val"]:
        json_path = os.path.join(_PROJECT_ROOT, "datasets", "Omni3D", f"{dataset}_{mode}.json")
        if not os.path.exists(json_path):
            print(f"  [skip] {json_path} not found")
            continue

        with open(json_path) as f:
            data = json.load(f)

        out_dir = os.path.join(outdir, dataset, mode, "depth")
        os.makedirs(out_dir, exist_ok=True)

        pbar = tqdm(total=len(data["images"]), desc=f"{dataset}_{mode}")

        # Track pending results: img_id -> (rgb, K)
        pending = {}   # img_id -> dict with rgb, K, out_path
        results_moge = {}  # img_id -> (depth, mask)
        results_dp = {}    # img_id -> depth

        i = 0
        while i < len(data["images"]) or pending:
            # Submit tasks until queue has max 32 pending or we run out of images
            while i < len(data["images"]) and len(pending) < 32:
                img_info = data["images"][i]
                file_name = img_info["id"]
                rgb_path = os.path.join(_PROJECT_ROOT, "datasets", img_info["file_path"])
                K = np.array(img_info["K"]).reshape(3, 3)
                out_path = os.path.join(out_dir, f"{file_name}.npy")

                if args.skip_existing and os.path.exists(out_path):
                    i += 1
                    continue

                if not os.path.exists(rgb_path):
                    i += 1
                    continue

                rgb = np.array(Image.open(rgb_path))
                pending[file_name] = {"rgb": rgb, "K": K, "out_path": out_path}

                # Submit to both workers
                moge_task_queue.put((file_name, rgb))
                dp_task_queue.put((file_name, rgb))
                i += 1

            # Collect results
            try:
                img_id, depth, mask, err = result_queue.get(timeout=60)
            except Empty:
                continue

            if err is not None:
                print(f"  [error] {img_id}: {err}")
                if img_id in pending:
                    del pending[img_id]
                errors += 1
                pbar.update(1)
                continue

            # Determine which model returned (4th element is None for DepthPro, not None for MoGe)
            if mask is not None:
                # This is MoGe result with mask
                results_moge[img_id] = (depth, mask)
            elif img_id not in results_dp:
                # This is DepthPro result
                results_dp[img_id] = depth
            else:
                continue  # duplicate, ignore

            # If both results are ready, fuse and save
            if img_id in results_moge and img_id in results_dp and img_id in pending:
                info = pending[img_id]
                moge_depth, moge_mask = results_moge[img_id]
                depth_fused, diag = align_depth_ransac(
                    moge_depth, results_dp[img_id], mask=moge_mask, verbose=False
                )
                np.save(info["out_path"], depth_fused)
                del pending[img_id]
                del results_moge[img_id]
                del results_dp[img_id]
                processed += 1
                pbar.update(1)

        pbar.close()

    # Shutdown workers
    moge_task_queue.put(None)
    dp_task_queue.put(None)
    p_moge.join()
    p_dp.join()
    print(f">> Done. Processed={processed}, Errors={errors}")


if __name__ == "__main__":
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass  # already set
    args = parse_args()
    process(args)
