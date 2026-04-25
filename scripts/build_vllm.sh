#!/bin/bash
#SBATCH --job-name=build-vllm
#SBATCH -p public
#SBATCH -q public
#SBATCH -A grp_ywang354
#SBATCH -t 6:00:00
#SBATCH --mem=128G
#SBATCH -c 24
#SBATCH --output=logs/jobs/build_vllm_%j.out
#SBATCH --error=logs/jobs/build_vllm_%j.err

cd /home/$USER/projects/CSE485-Capstone
mkdir -p logs/jobs

module load cuda-12.6.1-gcc-12.1.0
module load gcc-12.1.0-gcc-11.2.0
module load mamba/latest

rm -f /scratch/$USER/envs/venv/etc/conda/activate.d/libxml2_activate.sh 2>/dev/null || true
source activate /scratch/$USER/envs/venv

echo "Env check"
which python
python --version
gcc --version 2>&1 | head -1
nvcc --version 2>&1 | tail -1
ldd --version 2>&1 | head -1

echo "Uninstalling old vllm..."
pip uninstall vllm -y || true

echo "Building vLLM v0.11.0 from source..."
MAX_JOBS=4 VLLM_TARGET_DEVICE=cuda TORCH_CUDA_ARCH_LIST="8.0" \
    pip install "vllm @ git+https://github.com/vllm-project/vllm.git@v0.11.0" \
    --no-build-isolation 2>&1

echo "Verifying installation"
python -c "
import vllm
print('vLLM version:', vllm.__version__)
import vllm._moe_C
print('_moe_C extension: OK')
print('topk_softmax available:', hasattr(vllm._moe_C, 'topk_softmax') or 'topk_softmax' in dir(getatt    r(__import__('torch').ops, '_moe_C', None) or type('', (), {})))
"

echo "Build completed"