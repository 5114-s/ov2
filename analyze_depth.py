#!/usr/bin/env python3
"""Quick analysis of depth issue"""

# SUNRGBD 真实 fx = 529.5
# run_metric_anything_768.py 第 150 行:
#   inv_depth = inv_depth * (target_size / f_px)
#   其中 target_size = 1536, f_px = 529.5 (从 JSON K[0,0])

target_size = 1536
f_px_real = 529.5
f_px_model = 1000  # 模型训练时假设的规范焦距

scale_wrong = target_size / f_px_real
scale_correct = target_size / f_px_model

print("=== 深度计算问题分析 ===")
print()
print("SUNRGBD 数据集:")
print(f"  真实焦距 fx = {f_px_real}")
print()
print("run_metric_anything_768.py 第 150 行:")
print(f"  inv_depth = inv_depth * ({target_size} / {f_px_real})")
print(f"             = inv_depth * {scale_wrong:.3f}")
print()
print("正确公式 (参考 depth_model.py):")
print(f"  inv_depth = inv_depth * ({target_size} / {f_px_model})")
print(f"             = inv_depth * {scale_correct:.3f}")
print()
print("问题: 使用真实焦距 529.5 而不是模型期望的 ~1000")
print(f"导致深度被额外缩小: {scale_wrong/scale_correct:.2f}x")
print()
print("验证:")
print(f"  实测新/旧深度比例: 0.714/1.812 = {0.714/1.812:.2f}")
print(f"  理论深度比例: {scale_correct/scale_wrong:.2f}")
print(f"  (差异来自模型本身输出尺度不同)")
