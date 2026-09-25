#!/usr/bin/env bash
# Run a command inside the project environment (WSL distro `recon`).
# From Windows:  wsl -d recon -u root -- bash /mnt/d/project_other/multiview-recon-eval/scripts/env.sh python ...
source /opt/venvs/recon/bin/activate
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=${CUDA_HOME}/bin:${PATH}
cd "$(dirname "$0")/.."
exec "$@"
