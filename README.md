# CSE485 Capstone

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

## Resources

- [Measuring Bargaining Abilities of LLMs: A Benchmark and A Buyer-Enhancement Method](https://aclanthology.org/2024.findings-acl.213.pdf)
- [RetinalGPT: A Retinal Clinical Preference Conversational Assistant Powered by Large Vision-Language Models](https://arxiv.org/abs/2503.03987)
- [Build a Large Language Model](resources/build_a_llm_sebastian_raschka.pdf)
- [Jira Board](https://capstone-fall-2025-yalin-wang.atlassian.net/jira/software/projects/SCRUM/summary)

> **Need help?** Check out the [ASU Research Computing guide](https://asurc.atlassian.net/wiki/spaces/RC/pages/2319417345/A+Brief+Example#Step-3---Use-/-Test) for detailed setup instructions.

# Setup

### (A) Clone repo

```bash
git clone https://github.com/dhruvb26/cse485-capstone.git
cd cse485-capstone

```

### (B) Get a GPU Shell on SOL

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

### (C) Load required cluster modules

```bash
# Load package manager
module load mamba/latest

# Load CUDA drivers
module load cuda-12.6.1-gcc-12.1.0

module list
```

### (D) Install core dependencies + create and activate virtual environment

```bash
mamba env remove -n venv # OPTIONAL: if you want to wipe old env

mamba env create -f environment.yml
conda activate venv
```

### (E) (Optional) Verify PyTorch, CUDA, and vLLM versions:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"
python -c "import vllm; print('vLLM:', vllm.__version__)"
```

You should see something like:

```bash
2.x.x 12.1 True
NVIDIA A100-SXM4-80GB
vLLM: 0.11.0
```

# Usage

**_Important: Open 2 terminals, one for vLLM and the other one for actual application_**

### (A) Set env vars & cache dirs

```bash
export USER="ltnguy58" # IMPORTANT: put your ASU alias here
export VLLM_MODEL="mistralai/Mistral-7B-Instruct-v0.2" # or Qwen/Qwen2.5-7B-Instruct
export VLLM_PORT=8000
export VLLM_HOST="0.0.0.0"
export VLLM_CACHE_DIR=$SCRATCH/$USER/hf_cache_mistral # or hf_cache_gwen
mkdir -p $VLLM_CACHE_DIR
```

### (B) Load modules + env

```bash
module load mamba/latest
module load cuda-12.1.1-gcc-12.1.0
mamba env create -f environment.yml
conda activate venv
```

## Terminal A
### Launch vLLM

``` bash
# launch the server
python -m vllm.entrypoints.openai.api_server \
  --model $VLLM_MODEL \
  --port  $VLLM_PORT \
  --dtype float16 \
  --tensor-parallel-size 1 \
  --download-dir $VLLM_CACHE_DIR
```

This is going to run the vLLM server locally. Initalizing the model takes a while, you are free to proceed if you see this message:

```bash
...
(APIServer pid=2689620) INFO:     Started server process [2689620]
(APIServer pid=2689620) INFO:     Waiting for application startup.
(APIServer pid=2689620) INFO:     Application startup complete.
```

## Terminal B

### Start the chatbot

```bash
cd ~/cse485-capstone
python main.py
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

## Troubleshooting

### Common Issues

**Connection Problems:**

- Ensure Cisco VPN is active before SSH attempts
- Verify your ASU credentials are correct

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
