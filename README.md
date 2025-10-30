# CSE485 Capstone

## Table of Contents

- [Resources](#resources)
- [Setup](#setup)
  - [(A) Get a GPU Shell on SOL](#a-get-a-gpu-shell-on-sol)
  - [(B) Load required cluster modules](#b-load-required-cluster-modules)
  - [Create and activate virtual environment](#create-and-activate-virtual-environment)
  - [(D) Clone repo](#d-clone-repo)
  - [(E) Install core dependencies (GPU-ready)](#e-install-core-dependencies-gpu-ready)
  - [(F) Verify PyTorch sees the GPU](#f-verify-pytorch-sees-the-gpu)
  - [(G) Install vLLM](#g-install-vllm)
- [Recommendations (Optional)](#recommendations-optional)
- [Usage](#usage)
  - [Terminal A](#terminal-a)
  - [Terminal B](#terminal-b)
- [Troubleshooting](#troubleshooting)
  - [Common Issues](#common-issues)
- [Getting Help](#getting-help)

## Resources:

- [Measuring Bargaining Abilities of LLMs: A Benchmark and A Buyer-Enhancement Method](https://aclanthology.org/2024.findings-acl.213.pdf)
- [RetinalGPT: A Retinal Clinical Preference Conversational Assistant Powered by Large Vision-Language Models](https://arxiv.org/abs/2503.03987)
- [Jira Board](https://capstone-fall-2025-yalin-wang.atlassian.net/jira/software/projects/SCRUM/summary)

> **Need help?** Check out the [ASU Research Computing guide](https://asurc.atlassian.net/wiki/spaces/RC/pages/2319417345/A+Brief+Example#Step-3---Use-/-Test) for detailed setup instructions.

# Setup

### (A) Get a GPU Shell on SOL

```bash
# For new sessions
interactive -p htc -t 2:00:00 --gres=gpu:a100:1

# If already in an interactive session
salloc -p htc -t 2:00:00 --gres=gpu:a100:1
```

Wait until something like:

```bash
[ltnguy58@sg0XX ~]$
```

"sg\*" means GPU node.

> **Resource Limits**: HTC partition provides up to 240 minutes with a single A100 GPU

### (B) Load required cluster modules

```bash
# Load package manager
module load mamba/latest

# Load CUDA drivers
module load cuda-12.6.1-gcc-12.1.0

module list
```

### Create and activate virtual environment

```bash
# optional if you want to wipe old env
mamba env remove -n venv

# Create environment with Python 3.10
mamba create -n venv python=3.10 -y

# Activate environment
source activate venv
```

### (D) Clone repo

```bash
cd ~
rm -rf cse485-capstone
git clone https://github.com/dhruvb26/cse485-capstone.git
cd cse485-capstone

```

### (E) Install core dependencies (GPU-ready):

Correct combo for SOL and A100s. We use mamba for general libs, then pip for the CUDA 12.1 PyTorch wheel (this guarantees GPU support):

```bash
mamba install -y -c conda-forge \
  accelerate transformers datasets bitsandbytes sentencepiece \
  huggingface_hub tqdm numpy pandas scipy safetensors protobuf psutil

# Install GPU-enabled PyTorch from PyTorch's own wheel index (CUDA 12.1)
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> **NOTE: Look at [anaconda.org](https://anaconda.org/search) for the package versions and channels.**

### (F) Verify PyTorch sees the GPU:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

You should see something like:

```bash
2.x.x 12.1 True
NVIDIA A100-SXM4-80GB
```

### (G) Install vLLM:

```
pip install "vllm>=0.5.0" --extra-index-url https://download.pytorch.org/whl/cu121
python -c "import vllm; print('vLLM:', vllm.__version__)"
```

## Recommendations (Optional)

To keep the code manageable, we recommend using the [Ruff](https://marketplace.cursorapi.com/items/?itemName=charliermarsh.ruff) to check for code style issues. It can be installed in the extension marketplace of your IDE.

These are the settings for the `settings.json` file:

```json
"[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll.ruff": "explicit",
      "source.organizeImports.ruff": "explicit"
    }
  },
  "ruff.path": ["uv", "run", "ruff"],
  "ruff.fixAll": true,
  "ruff.organizeImports": true,
```

# Usage

**_Important: Open 2 terminals, one for vLLM and the other one for actual application_**

## Terminal A

### (A) Set env vars & cache dirs

```bash
export VLLM_MODEL="Qwen/Qwen2.5-7B-Instruct"   # or Qwen/Qwen2.5-7B-Chat
export VLLM_PORT=8000
export VLLM_HOST="0.0.0.0"
export VLLM_CACHE_DIR=$SCRATCH/hf_cache
mkdir -p "$VLLM_CACHE_DIR"
```

### (B) Load modules + env

```bash
module load mamba/latest
module load cuda-12.1.1-gcc-12.1.0
source activate venv
```

### (C) Launch vLLM

```
export VLLM_MODEL="mistralai/Mistral-7B-Instruct-v0.2" # or Qwen/Qwen2.5-7B-Instruct
export VLLM_PORT=8000
export VLLM_HOST="0.0.0.0"
export VLLM_CACHE_DIR=$SCRATCH/hf_cache_mistral # hf_cache_gwen
mkdir -p $VLLM_CACHE_DIR

# launch the server
python -m vllm.entrypoints.openai.api_server \
  --model $VLLM_MODEL \
  --port  $VLLM_PORT \
  --dtype float16 \
  --tensor-parallel-size 1 \
  --download-dir $VLLM_CACHE_DIR
```

This is going to run the vLLM server with Gwen running locally. Initalizing the model takes a while, you are free to proceed if you see this message:

```bash
...
(APIServer pid=2689620) INFO:     Started server process [2689620]
(APIServer pid=2689620) INFO:     Waiting for application startup.
(APIServer pid=2689620) INFO:     Application startup complete.
```

(Optional) You can also run vLLM in the background

```bash
nohup python -m vllm.entrypoints.openai.api_server \
  --model $VLLM_MODEL --port $VLLM_PORT \
  --dtype float16 --tensor-parallel-size 1 \
  --download-dir $VLLM_CACHE_DIR > vllm.log 2>&1 &
tail -f vllm.log    # to watch logs
# later: pkill -f vllm.entrypoints.openai.api_server
```

## Terminal B

### (A) Run the app

```bash
interactive -p htc -t 2:00:00 --gres=gpu:a100:1
module load mamba/latest
module load cuda-12.1.1-gcc-12.1.0
source activate venv
cd ~/cse485-capstone
python main.py
```

**Start the chatbot:**

```bash
python main.py
```

This launches the interactive chat interface with the Qwen2.5-7B-Instruct model, enabling you to engage in conversations and test the negotiation capabilities.

Make sure `main.py` uses:

```python
testing_model = "Qwen/Qwen2.5-7B-Instruct"
```

## Troubleshooting

### Common Issues

**Connection Problems:**

- Ensure Cisco VPN is active before SSH attempts
- Verify your ASU credentials are correct

**Resource Allocation:**

- If GPU allocation fails, try requesting during off-peak hours
- Check current cluster usage with `squeue`

**Environment Issues:**

- Verify CUDA module is loaded: `module list`
- Confirm virtual environment activation: `which python`

**Important about `testing_model.py`**

The code routes by prefix, if `gpt-*`, the program will use ChatGPT instead of anything else.

To use local Gwen, set this in `main.py`:

```python
run_session(
    testing_model="Qwen/Qwen2.5-7B-Instruct",  # or Qwen/Qwen2.5-7B-Chat
    product_limit=12,
    dataset_dir="data/amazon_history_price",
)
```

## Getting Help

- ASU Research Computing Support: [RC Documentation](https://asurc.atlassian.net/wiki/spaces/RC)
- Project Issues: Create an issue in the repository
- Technical Questions: Consult course materials or instructor

**Course**: CSE 485 - Capstone Project 1</br>
**Institution**: Arizona State University
