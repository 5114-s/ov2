#!/usr/bin/env python3
"""Check SUNRGBD image sizes and K matrices to understand focal length scaling"""

import json
import numpy as np
from PIL import Image
from pathlib import Path

# Load one of the JSON files
json_path = "/data/ZhaoX/OVM3D-Det/datasets/Omni3D_pl-1/SUNRGBD_train.json"
with open(json_path, 'r') as f:
    data = json.load(f)

print(f"Total images: {len(data['images'])}")
print()

# Check first few images
print("=== 前5张图片的信息 ===\n")
for i in range(5):
    img_info = data['images'][i]
    img_id = img_info['id']
    file_path = img_info['file_path']
    K = np.array(img_info['K'])

    # Try to load image to get actual size
    img_path = f"/data/ZhaoX/OVM3D-Det/datasets/{file_path}"
    try:
        img = Image.open(img_path)
        actual_w, actual_h = img.size
    except:
        actual_w, actual_h = "N/A", "N/A"

    # K matrix info
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    print(f"Image {img_id}:")
    print(f"  File: {file_path}")
    print(f"  Image size (from PIL): {actual_w} x {actual_h}")
    print(f"  K[0,0] (fx): {fx:.2f}")
    print(f"  K[1,1] (fy): {fy:.2f}")
    print(f"  K[0,2] (cx): {cx:.2f}")
    print(f"  K[1,2] (cy): {cy:.2f}")

    # Check if fx is close to image width
    if isinstance(actual_w, int):
        print(f"  fx / image_width = {fx / actual_w:.4f}")
        print(f"  图像宽度约是 fx 的 {actual_w / fx:.4f} 倍")

        # What if we use image width as focal?
        print(f"  使用 image.width ({actual_w}) 作为焦距")
        print(f"  如果用 730 焦距: 缩放比例 = {actual_w / 730:.4f}")
    print()

# Check K matrix consistency
print("=== K 矩阵统计 ===")
fxs = [np.array(img['K'])[0, 0] for img in data['images'][:100]]
print(f"fx 范围: {min(fxs):.2f} - {max(fxs):.2f}")
print(f"fx 平均: {np.mean(fxs):.2f}")
print(f"fx 标准差: {np.std(fxs):.2f}")
