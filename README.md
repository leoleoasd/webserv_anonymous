# WebServ

A full-stack, RL-ready web environment for scalable web-agent training and evaluation.

## Repository Structure

```
├── web_agent/          # Web agent training loop, browser env, prompts, data
├── shared/             # Common utilities (async rollout, metrics, distributed helpers)
├── dependencies/
│   ├── slime/          # RL training framework (Megatron-based GRPO)
│   └── rl_web_agent/   # Browser environment, evaluator, Incus container client
├── scripts/            # Cluster utilities (sglang, checkpoints, Ray)
├── data/               # SFT training data
├── Dockerfile          # Training container image
└── pyproject.toml      # Python dependencies
```

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- NVIDIA GPUs (H100/H200 recommended)
- Ray cluster
- Incus container host(s) for web server environments
- SGLang inference servers

## Installation

```bash
uv sync
```

## Running RL Training (GRPO)

### 1. Convert HF Checkpoint to Megatron Format

```bash
source dependencies/slime/scripts/models/qwen3-4B-Instruct-2507.sh && \
uv run python dependencies/slime/tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint /path/to/Qwen3-4B/ \
    --save /path/to/Qwen3-4B_torch_dist/
```

### 2. Set Up Incus and Web Server Environments

WebServ uses [Incus](https://linuxcontainers.org/incus/) containers with ZFS copy-on-write to manage isolated web server instances for RL rollouts.

#### 2.1 Install Incus

```bash
# Ubuntu/Debian
sudo apt install incus

# Initialize Incus with ZFS storage backend
sudo incus admin init
# When prompted:
#   - Storage backend: zfs
#   - Create a new ZFS pool: yes
#   - Use an existing block device or loop file (your choice)
```

#### 2.2 Import WebArena Docker Images into Incus

Follow the [WebArena documentation](https://github.com/web-arena-x/webarena) to pull the Docker images for the web environments you need (shopping, CMS, GitLab). Then import them into Incus:

```bash
# Export Docker image to tarball
docker save <webarena_image_name> -o webarena_shopping.tar

# Import into Incus as an image
incus image import webarena_shopping.tar --alias webarena-shopping

# Create a base container from the image
incus launch webarena-shopping shopping-base

# Wait for the container to fully start and services to initialize,
# then snapshot it (this snapshot is what gets cloned during rollouts)
incus snapshot shopping-base ready
incus stop shopping-base
```

Repeat for each site (CMS, GitLab, etc.).

#### 2.3 Start the Incus Server

The Incus server is an HTTP API that manages container lifecycle (launch, clone, reset, delete) for the training loop:

```bash
uv run python dependencies/rl_web_agent/incus_server.py
```

By default it listens on port 8001. Key environment variables:

| Variable | Purpose | Default |
|---|---|---|
| `INCUS_POOL` | ZFS storage pool name | `default` |
| `MIN_FREE_MEMORY_RATIO` | Minimum free memory before rejecting launches | `0.1` |
| `THROTTLE_INTERVAL` | Minimum seconds between container launches | `1.0` |

#### 2.4 Other Infrastructure

Ensure the following services are also running:

- **Ray cluster**: Head node + worker nodes with GPUs
- **Proxy server**: Host-rewriting proxy for container routing (set `PROXY_SERVER`)

### 3. Launch GRPO Training

```bash
./web_agent/run_grpo_async.sh <model_config> <run_name>
```

**Arguments:**
- `model_config`: Name of a file under `web_agent/models/` (without `.sh`), e.g., `qwen3-4B`, `qwen3-30B-A3B`
- `run_name`: Experiment name (used for wandb group and checkpoint directory)

**Example:**
```bash
./web_agent/run_grpo_async.sh qwen3-4B my_rl_experiment
```

This will:
1. Source model-specific config from `web_agent/models/qwen3-4B.sh`
2. Submit a Ray job that runs async GRPO training
3. Launch up to 512 parallel browser-server rollout instances
4. Train on 64 GPUs (4 actor nodes × 8 GPUs) with 32 SGLang inference workers

### 4. Model Configs

Model configs are in `web_agent/models/`. Each sets:
- `MODEL_ARGS`: Architecture parameters (from slime model scripts)
- `CKPT_ARGS`: Checkpoint paths (load/save)
- `PERF_ARGS`: Parallelism (TP, PP, CP, recompute)
- `SGLANG_ARGS`: Inference engine config
- `MISC_ARGS`: Training options
- Cluster size: `ACTOR_NUM_NODES`, `ROLLOUT_NUM_GPUS`

### 5. Key Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `INCUS_SERVER_URL` | Incus container orchestrator endpoint | `http://127.0.0.1:8001` |
| `PROXY_SERVER` | Host-rewriting proxy for container routing | `http://localhost:8080` |
| `MAX_CONCURRENT_CONTAINER_LAUNCHES` | Max parallel container launches | `32` |
| `MAX_CONCURRENT_CONTAINERS_RUNNING` | Max total running containers | `512` |
| `BROWSER_HEADLESS` | Run browsers in headless mode | `true` |
| `TOOL_CALL_PARSER` | SGLang tool call parser | `qwen25` |
| `NUM_ASYNC_ROLLOUT_WORKERS` | Number of async rollout workers | `16` |

## Training Hyperparameters

Default GRPO configuration (from `web_agent/run_grpo_async.sh`):

- **Algorithm**: GRPO with dynamic sampling filtering
- **KL loss**: Low-variance KL, coefficient 0.001
- **PPO clipping**: ε = 0.2, high clip 0.28
- **Optimizer**: Adam, lr = 1e-6, constant schedule, weight decay 0.01
- **Rollouts**: 200 per step, batch size 16, 12 samples per prompt, max response 4096 tokens, temperature 1.0
- **Cluster**: 64 GPUs (4 actor nodes × 8 GPUs), 32 SGLang workers

## SFT Data

Pre-processed SFT training data is provided in `data/sft_training.jsonl` (726 examples). This is used to create the SFT checkpoint that initializes RL training.

## Evaluation

The evaluation framework is in `dependencies/rl_web_agent/`. To evaluate a checkpoint:

```bash
uv run python dependencies/rl_web_agent/run_sglang_eval.py \
    --model /path/to/checkpoint \
    --dataset-dir dependencies/rl_web_agent/dataset/test_webarena_lite \
    --output-dir /path/to/results
```

## Converting Checkpoints to HF Format

```bash
bash scripts/convert_all_checkpoints.sh \
    --input-dir /path/to/training_checkpoints \
    --output-dir /path/to/hf_checkpoints \
    --origin-hf-dir /path/to/original_hf_model
```
