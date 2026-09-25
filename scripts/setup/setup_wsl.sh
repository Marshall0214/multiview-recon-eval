#!/usr/bin/env bash
# Provision the `recon` WSL2 distro (Ubuntu 22.04, run as root).
# Everything installs inside the distro's ext4.vhdx, which lives under env/wsl/ on D:.
#
# Usage (from Windows):  wsl -d recon -u root -- bash /mnt/d/project_other/multiview-recon-eval/scripts/setup/setup_wsl.sh
#
# Network notes (mainland China): archive.ubuntu.com and GitHub are slow from WSL, so apt / PyPI
# go through the TUNA mirror. Large files that only live on GitHub (the gsplat wheel) are fetched
# on the Windows side into env/downloads/ and installed from there.
set -euo pipefail

REPO=/mnt/d/project_other/multiview-recon-eval
VENV=/opt/venvs/recon
DL=${REPO}/env/downloads
GSPLAT_WHL=gsplat-1.5.3+pt24cu124-cp310-cp310-linux_x86_64.whl

export DEBIAN_FRONTEND=noninteractive

echo "== [0/5] mirrors"
sed -i 's|http://archive.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g; s|http://security.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list
cat > /etc/pip.conf <<EOF
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
EOF

echo "== [1/5] apt packages"
apt-get update -q
apt-get install -y -q build-essential git wget curl ca-certificates unzip ffmpeg \
    python3.10 python3.10-venv python3.10-dev \
    libgl1 libglib2.0-0 libegl1 libxrender1 libxi6 libxkbcommon0 libsm6 colmap

echo "== [2/5] CUDA toolkit 12.4 (WSL repo: toolkit only, the driver comes from Windows)"
if [ ! -x /usr/local/cuda-12.4/bin/nvcc ]; then
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
    dpkg -i /tmp/cuda-keyring.deb
    apt-get update -q
    apt-get install -y -q cuda-toolkit-12-4
fi

echo "== [3/5] python 3.10 venv"
[ -x ${VENV}/bin/python ] || python3.10 -m venv ${VENV}
PY=${VENV}/bin/python
$PY -m pip install -q -U pip wheel setuptools

echo "== [4/5] PyTorch 2.4.1 + cu124, gsplat 1.5.3 prebuilt wheel"
$PY -m pip install -q torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
if [ -f "${DL}/${GSPLAT_WHL}" ]; then
    $PY -m pip install -q "${DL}/${GSPLAT_WHL}"
else
    $PY -m pip install -q "gsplat==1.5.3+pt24cu124" --index-url https://docs.gsplat.studio/whl
fi

echo "== [5/5] gsplat example deps (fused-ssim / fused-bilagrid compile against CUDA 12.4) + project deps"
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=${CUDA_HOME}/bin:${PATH}
export TORCH_CUDA_ARCH_LIST="8.9"   # RTX 4090 D
# The git+https deps are pinned to commits; if their tarballs were fetched into env/downloads/gitdeps
# (codeload.github.com/<owner>/<repo>/tar.gz/<sha>), install from those instead of cloning GitHub.
REQ=/tmp/gsplat-examples-req.txt
cp ${REPO}/third_party/gsplat/examples/requirements.txt ${REQ}
for f in ${DL}/gitdeps/*.tar.gz; do
    [ -e "$f" ] || continue
    name=$(basename "$f" .tar.gz); repo=${name%-*}; sha=${name##*-}
    sed -i "s|^git+https://github.com/[^/]*/${repo}@${sha}|${f}|" ${REQ}
done
$PY -m pip install -q --no-build-isolation -r ${REQ}
$PY -m pip install -q -r ${REPO}/requirements.txt

# convenience: auto-activate env for interactive shells
grep -q "${VENV}/bin/activate" /root/.bashrc || cat >> /root/.bashrc <<EOF
source ${VENV}/bin/activate
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=\${CUDA_HOME}/bin:\${PATH}
EOF

# headless Blender 4.2 LTS (tarball fetched into env/downloads/)
BLENDER_TAR=${DL}/blender-4.2.23-linux-x64.tar.xz
if [ -f "${BLENDER_TAR}" ] && [ ! -x /opt/blender-4.2.23-linux-x64/blender ]; then
    tar -xf "${BLENDER_TAR}" -C /opt
fi
[ -x /opt/blender-4.2.23-linux-x64/blender ] && ln -sf /opt/blender-4.2.23-linux-x64/blender /usr/local/bin/blender

echo "== done"
$PY -c "import torch, gsplat; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.get_device_name(0)); print('gsplat', gsplat.__version__)"
