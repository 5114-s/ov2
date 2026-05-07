#!/usr/bin/env python3
"""
MetricAnything depth estimation for OVM3D-Det.
Uses 1536x1536 resolution with multiple GPU workers.

Usage:
    cd /data/ZhaoX/OVM3D-Det/third_party/metric-anything/models/student_depthmap
    python run_metric_anything.py --root /data/ZhaoX/OVM3D-Det --json-dir datasets/Omni3D_pl-1 --gpu 0 --gpu 1
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import types
from PIL import Image
from tqdm import tqdm
import json
import argparse
import torch.nn.functional as F
from torchvision.transforms import v2
from multiprocessing import Process, Queue

torch.backends.cudnn.enabled = False

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, '.')
sys.path.insert(0, './network')


def parse_args():
    parser = argparse.ArgumentParser(description="MetricAnything Depth Estimation")
    parser.add_argument('--root', type=str, default='/data/ZhaoX/OVM3D-Det')
    parser.add_argument('--json-dir', type=str, default='datasets/Omni3D_pl-1')
    parser.add_argument('--dataset', type=str, default='SUNRGBD')
    parser.add_argument('--pretrained', type=str, default='yjh001/metricanything_student_depthmap')
    parser.add_argument('--gpu', type=str, default='0', help='GPU IDs to use (comma-separated, e.g., "0,1")')
    return parser.parse_args()


def create_backbone():
    """Create DINOv3 backbone with correct forward."""
    from network.vit_factory import VIT_CONFIG_DICT
    from dinov3.hub.backbones import dinov3_vith16plus

    preset = 'dinov3_vith16plus_224'
    config = VIT_CONFIG_DICT[preset]

    model = dinov3_vith16plus(pretrained=False, weights=None)
    model.patch_embed.img_size = (config.img_size, config.img_size)

    original = model.forward_features

    def forward_with_cls(self, x, is_training=False, **kwargs):
        feats = original(x)
        cls_token = feats['x_norm_clstoken'].unsqueeze(1)
        patch_tokens = feats['x_norm_patchtokens']
        return torch.cat([cls_token, patch_tokens], dim=1)

    model.forward = types.MethodType(forward_with_cls, model)
    model.start_index = 1
    model.patch_size = model.patch_embed.patch_size
    model.is_vit = True

    return model


def create_model():
    from network.encoder import MetricAnythingEncoder
    from network.decoder import MultiresConvDecoder

    hook_block_ids = [7, 13, 19, 25]
    encoder_feature_dims = [320, 256, 320, 640]
    decoder_features = 256

    backbone = create_backbone()
    encoder = MetricAnythingEncoder(
        dims_encoder=encoder_feature_dims,
        patch_encoder=backbone,
        hook_block_ids=hook_block_ids,
    )

    decoder_dims = [decoder_features] + [encoder.dims_encoder[0]] * 2 + list(encoder.dims_encoder)
    decoder = MultiresConvDecoder(dims_encoder=decoder_dims, dim_decoder=decoder_features)

    model = MetricAnythingModelWrapper(encoder, decoder)
    return model


class MetricAnythingModelWrapper(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.head = self._build_head(decoder.dim_decoder)

    def _build_head(self, dim_decoder):
        last_dims = (32, 1)
        layers = [
            nn.Conv2d(dim_decoder, dim_decoder // 2, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(dim_decoder // 2, dim_decoder // 2, kernel_size=2, stride=2, padding=0, bias=True),
            nn.Conv2d(dim_decoder // 2, last_dims[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(4):
            layers += [nn.Conv2d(last_dims[0], last_dims[0], kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(last_dims[0], last_dims[1], kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        encodings = self.encoder(x)
        features, _ = self.decoder(encodings)
        return self.head(features)

    @torch.no_grad()
    def infer(self, x, orig_size, f_px, interpolation_mode="bilinear"):
        if x.ndim == 3:
            x = x.unsqueeze(0)

        orig_h, orig_w = orig_size

        # Resize to 1536x1536 for the model
        target_size = 1536
        x = F.interpolate(x, size=(target_size, target_size), mode=interpolation_mode, align_corners=False)

        canonical_inverse_depth = self.forward(x)

        if f_px is None:
            f_px = 1000.0

        # Scale focal length based on resize
        scale = max(orig_h, orig_w) / target_size
        f_px_scaled = f_px / scale

        # Convert inverse depth to metric depth
        inverse_depth = canonical_inverse_depth * (target_size / f_px_scaled)

        # Resize back to original size
        inverse_depth = F.interpolate(inverse_depth, size=(orig_h, orig_w), mode=interpolation_mode, align_corners=False)

        depth = 1.0 / torch.clamp(inverse_depth, min=1e-3, max=1e3)
        return {"depth": depth.squeeze()}


def load_model(pretrained_path, device):
    from huggingface_hub import hf_hub_download

    print(f"[GPU {device}] Loading model...")
    checkpoint_path = hf_hub_download(repo_id=pretrained_path, repo_type="model", filename="student_depthmap.pt")

    model = create_model().to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    print(f"[GPU {device}] Model loaded!")
    return model


def make_transform():
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def worker_process(gpu_id, work_queue, result_queue, pretrained_path):
    """Worker process for each GPU."""
    device = torch.device(f"cuda:{gpu_id}")
    model = load_model(pretrained_path, device)
    transform = make_transform()

    while True:
        task = work_queue.get()
        if task is None:
            break

        idx, image_info, root = task
        filename = image_info['file_path']
        file_name = image_info['id']
        image_path = os.path.join(root, 'datasets', filename)

        try:
            image = Image.open(image_path).convert('RGB')
            orig_w, orig_h = image.size

            input_tensor = transform(image).unsqueeze(0).to(device)

            intrinsics = np.array(image_info['K']).reshape(3, 3)
            f_px = float(intrinsics[0, 0])

            prediction = model.infer(input_tensor, orig_size=(orig_h, orig_w), f_px=f_px)
            depth = prediction["depth"].detach().cpu().numpy()

            result_queue.put((idx, file_name, depth, None))
        except Exception as e:
            result_queue.put((idx, file_name, None, str(e)))


def main():
    args = parse_args()

    # Parse GPU IDs
    device_ids = [int(x) for x in args.gpu.split(',')]
    num_gpus = len(device_ids)
    print(f"Using {num_gpus} GPUs: {device_ids}")

    json_dir = os.path.join(args.root, args.json_dir)
    out_base = os.path.join(args.root, f'pseudo_label-{args.dataset}')

    transform = make_transform()

    for mode in ['train', 'val']:
        json_path = os.path.join(json_dir, f'{args.dataset}_{mode}.json')
        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found, skipping...")
            continue

        with open(json_path, 'r') as f:
            data = json.load(f)

        outdir = os.path.join(out_base, mode, 'depth')
        os.makedirs(outdir, exist_ok=True)

        # Filter out already processed images
        tasks = []
        for i, image_info in enumerate(data['images']):
            file_name = image_info['id']
            out_path = os.path.join(outdir, f"{file_name}.npy")
            if not os.path.exists(out_path):
                tasks.append((i, image_info, args.root))

        if not tasks:
            print(f"{mode}: all images already processed")
            continue

        print(f"\n{mode}: {len(tasks)} images to process on {num_gpus} GPUs")

        # Create queues
        work_queue = Queue()
        result_queue = Queue()

        # Start workers
        workers = []
        for gpu_id in device_ids:
            p = Process(target=worker_process, args=(gpu_id, work_queue, result_queue, args.pretrained))
            p.start()
            workers.append(p)

        # Add tasks to queue
        for task in tasks:
            work_queue.put(task)

        # Add sentinel values
        for _ in device_ids:
            work_queue.put(None)

        # Process results with progress bar
        pbar = tqdm(total=len(tasks), desc=f"{mode}")
        completed = 0
        errors = 0

        while completed + errors < len(tasks):
            idx, file_name, depth, error = result_queue.get()
            if error:
                print(f"\nError processing {file_name}: {error}")
                errors += 1
            else:
                out_path = os.path.join(outdir, f"{file_name}.npy")
                np.save(out_path, depth)
                completed += 1
            pbar.update(1)

        pbar.close()

        # Wait for workers
        for p in workers:
            p.join()

        print(f"{mode}: completed {completed}, errors {errors}")

    print("\nDone!")


if __name__ == "__main__":
    main()
