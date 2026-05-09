#!/bin/bash
set -e

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ovm3d-1

# 卸载旧版本
pip uninstall -y nvdiffrast 2>/dev/null || true

# 删除残留
rm -rf ~/miniconda3/envs/ovm3d-1/lib/python3.10/site-packages/nvdiffrast*

# 克隆 nvdiffrast
cd /tmp
rm -rf nvdiffrast
git clone https://github.com/NVIDIA/nvdiffrast.git
cd nvdiffrast

# 查看当前目录结构
echo "=== Clone structure ==="
find . -name "*.so" -o -name "*.cpp" | head -20

# 编译（输出到 build 目录）
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ..

echo "=== After build ==="
find . -name "*.so" 2>/dev/null

# 安装（使用 pip install . 会正确安装）
pip install . --no-build-isolation

echo "=== Installation complete ==="
pip show nvdiffrast
python -c "import nvdiffrast.torch as ndr; print('Import successful:', ndr)"
