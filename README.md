# Flex AI — Serverless LLM Fine-Tuning & Inference

<p align="center">
  <img src="https://getflex.ai/og.png" alt="Flex AI" width="600"/>
</p>

<p align="center">
  <a href="https://pypi.org/project/flex_ai/"><img src="https://img.shields.io/pypi/v/flex_ai.svg" alt="PyPI version"/></a>
  <a href="https://pypi.org/project/flex_ai/"><img src="https://img.shields.io/pypi/pyversions/flex_ai.svg" alt="Python versions"/></a>
  <a href="https://github.com/arielcohen4/flex_ai/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"/></a>
  <a href="https://docs.getflex.ai"><img src="https://img.shields.io/badge/docs-getflex.ai-brightgreen" alt="Docs"/></a>
</p>

> **Fine-tune, serve and use open-source LLMs — no GPU setup required.**  
> Works on Linux, macOS, and Windows.

---

## What is Flex AI?

**[Flex AI](https://getflex.ai)** is a developer-first platform for training and serving open-source LLMs entirely serverlessly. Upload your dataset, kick off a fine-tune job, and deploy a production-ready OpenAI-compatible endpoint — all through a single Python client or CLI, from any machine.

### Architecture

```
Your Machine (any OS)
        │
        │  pip install flex_ai
        ▼
┌─────────────────────────────────┐
│   Flex AI Python Client / CLI   │
│   https://api.getflex.ai        │
└────────────┬────────────────────┘
             │  REST API
             ▼
┌─────────────────────────────────┐
│        Flex AI Backend          │
│  (Orchestration & Storage)      │
└────────────┬────────────────────┘
             │  Serverless dispatch
             ▼
┌─────────────────────────────────┐
│          Modal.com              │
│  Serverless GPU Endpoints       │
│  T4 · A10G · L4 · A100-40/80GB │
│  Training + Inference workloads │
└─────────────────────────────────┘
```

The heavy lifting — model loading, training loops, checkpoint saving, and serving — all runs on **[Modal](https://modal.com)** serverless GPU endpoints. You never provision, configure, or pay for idle GPUs. CPU-bound work (dataset upload, checkpoint download) happens outside GPU time to keep costs minimal.

### Why Flex AI?

| Pain point | How we solve it |
|------------|-----------------|
| Expensive GPU instances | Serverless — pay only for active compute |
| Multi-GPU complexity (FSDP / DeepSpeed) | Automatically configured based on model size |
| Wrong prompt templates wasting runs | Dataset validation before training starts |
| Too many checkpoints consuming storage | `save_only_best_checkpoint` option |
| Slow inference with long few-shot prompts | Fine-tune once, infer without giant prompts |
| Vendor lock-in | OpenAI-compatible endpoints, open-source models |

---

## Installation

```bash
pip install flex_ai
```

Requires Python ≥ 3.6. Works on **Linux**, **macOS**, and **Windows** with no local GPU needed.

---

## Authentication

> **Security note:** Your API key is a secret credential — treat it like a password.  
> Never hard-code it in source files or commit it to version control.

Get your API key from the [Flex AI dashboard → Settings → API Keys](https://app.getflex.ai/settings/api-keys).

### Option 1 — Environment variable (recommended)

```bash
# Linux / macOS
export FLEX_AI_API_KEY=flex_xxxxxxxxxxxxxxxxxxxx

# Windows (Command Prompt)
set FLEX_AI_API_KEY=flex_xxxxxxxxxxxxxxxxxxxx

# Windows (PowerShell)
$env:FLEX_AI_API_KEY="flex_xxxxxxxxxxxxxxxxxxxx"
```

The client picks it up automatically:

```python
from flex_ai import FlexAI

client = FlexAI()  # reads FLEX_AI_API_KEY from environment
```

### Option 2 — Pass directly (use only in trusted notebooks)

```python
client = FlexAI(api_key="flex_xxxxxxxxxxxxxxxxxxxx")
```

> ⚠️ **Never commit a real API key.** Add `.env` to your `.gitignore` and use environment variables or a secrets manager.

---

## Quick Start (4 lines of code)

```python
from flex_ai import FlexAI

client = FlexAI()

# 1. Upload dataset
dataset = client.create_dataset("my-dataset", "train.jsonl", "eval.jsonl")

# 2. Fine-tune
task = client.create_finetune(
    name="my-finetune",
    dataset_id=dataset["id"],
    model="meta-llama/Llama-3.2-1B-Instruct",
    n_epochs=3,
    train_with_lora=True,
)

# 3. Wait for training
client.wait_for_task_completion(task_id=task["id"])

# 4. Deploy & use
checkpoints = client.get_task_checkpoints(task_id=task["id"])
endpoint_id = client.create_multi_lora_endpoint(
    name="my-endpoint",
    lora_checkpoints=[{"id": c["id"], "name": f"step_{c['step']}"} for c in checkpoints],
    compute="A100-40GB",
)
endpoint = client.wait_for_endpoint_ready(endpoint_id=endpoint_id)
```

Then call your fine-tuned model via the OpenAI SDK:

```python
from openai import OpenAI

oai = OpenAI(api_key=client.api_key, base_url=f"{endpoint['url']}/v1")
response = oai.chat.completions.create(
    model="meta-llama/Llama-3.2-1B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

---

## Dataset Formats

All datasets are **JSONL files** (one JSON object per line). Flex AI auto-detects the format from the first row.

### Instruction format

Best for single-turn question/answer or task completion:

```jsonl
{"instruction": "Summarize this paragraph in one sentence.", "output": "The paragraph discusses..."}
{"instruction": "Translate to Spanish: Hello world", "output": "Hola mundo"}
```

Optional system prompt:

```jsonl
{"instruction": "What is 2+2?", "output": "4", "system": "You are a math tutor."}
```

### Chat format

Best for multi-turn conversations:

```jsonl
[{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "Paris."}]
[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hi!"}, {"role": "assistant", "content": "Hello! How can I help?"}]
```

### Text format

Best for generative / completion tasks:

```jsonl
{"text": "The quick brown fox jumps over the lazy dog."}
{"text": "In a galaxy far, far away..."}
```

> You need a `train.jsonl` file (required) and an `eval.jsonl` file (optional but strongly recommended).

### Validate before uploading

```python
client.validate_dataset("train.jsonl", "eval.jsonl")
```

This tokenizes your dataset locally, checks for format errors, and prints a token-length report — before any GPU time is spent.

---

## Fine-Tuning Options

```python
task = client.create_finetune(
    name="my-finetune",
    dataset_id=dataset["id"],
    model="meta-llama/Llama-3.2-1B-Instruct",
    n_epochs=5,
    train_with_lora=True,

    # Hyperparameters (all optional)
    batch_size=4,
    learning_rate=0.0001,
    n_checkpoints_and_evaluations_per_epoch=2,
    save_only_best_checkpoint=True,

    # LoRA config (optional, used when train_with_lora=True)
    lora_config={"lora_r": 64, "lora_alpha": 16, "lora_dropout": 0.05},

    # Early stopping (optional)
    early_stopping_config={"patience": 3, "threshold": 0.01},

    # Weights & Biases integration (optional)
    # wandb_key="wandb_xxxxxxxxxxxxxxxxxxxx",  # ⚠️ secret — use env var
)
```

### Available models

View all supported models at [app.getflex.ai/models](https://app.getflex.ai/models) or via the SDK:

```python
client.get_models()
```

### Full fine-tune vs LoRA

| Mode | `train_with_lora` | Use case |
|------|-------------------|----------|
| Full fine-tune | `False` | Maximum accuracy, higher cost |
| LoRA adapter | `True` | Fast, cheap, supports Multi-LoRA serving |

---

## Checkpoints

```python
# List all checkpoints for a task
checkpoints = client.get_task_checkpoints(task_id=task["id"])

# Get a single checkpoint by ID
checkpoint = client.get_checkpoint(id="<checkpoint_id>")

# Download checkpoint (PyTorch / Safetensors)
client.download_checkpoint(checkpoint_id="<checkpoint_id>")
# → extracted to ./<checkpoint_id>-checkpoint-step-<step>/

# Download GGUF checkpoint (for llama.cpp, Ollama, etc.)
client.download_checkpoint_gguf(checkpoint_id="<checkpoint_id>")
# → extracted to ./<checkpoint_id>-checkpoint-gguf-step-<step>/
```

---

## Multi-LoRA Inference Endpoints

Deploy multiple LoRA adapters behind a single OpenAI-compatible endpoint. The endpoint scales automatically and supports thousands of fine-tuned adapters.

```python
from flex_ai.common.classes import LoraCheckpoint

checkpoints = client.get_task_checkpoints(task_id=task["id"])

endpoint_id = client.create_multi_lora_endpoint(
    name="production-endpoint",
    lora_checkpoints=[
        {"id": c["id"], "name": f"step_{c['step']}"}
        for c in checkpoints
    ],
    compute="A100-40GB",          # GPU type
    idle_timeout_seconds=300,     # scale-to-zero after inactivity
)

endpoint = client.wait_for_endpoint_ready(endpoint_id=endpoint_id)
print(endpoint["url"])  # OpenAI-compatible base URL
```

### Available GPU options

| Compute | VRAM | Best for |
|---------|------|----------|
| `T4` | 16 GB | Small models (≤7B), cost-sensitive |
| `L4` | 24 GB | Mid-size models |
| `A10G` | 24 GB | Mid-size models, good throughput |
| `A100-40GB` | 40 GB | Large models (13B–34B) |
| `A100-80GB` | 80 GB | Very large models (70B+) |

---

## Python API Reference

| Method | Description |
|--------|-------------|
| `FlexAI(api_key=None)` | Init client. Reads `FLEX_AI_API_KEY` if not passed. |
| `validate_dataset(train_path, eval_path)` | Tokenize & validate locally before upload |
| `create_dataset(name, train_path, eval_path)` | Validate, upload, and register dataset |
| `get_datasets()` | List all datasets |
| `get_models()` | List all supported base models |
| `create_finetune(...)` | Start a fine-tuning job |
| `wait_for_task_completion(task_id, interval=1)` | Block until training completes |
| `get_task(id)` | Get task status and metadata |
| `get_task_checkpoints(task_id)` | List checkpoints for a task |
| `get_checkpoint(id)` | Get a single checkpoint |
| `download_checkpoint(checkpoint_id)` | Download & extract PyTorch checkpoint |
| `download_checkpoint_gguf(checkpoint_id)` | Download & extract GGUF checkpoint |
| `create_multi_lora_endpoint(...)` | Deploy a Multi-LoRA inference endpoint |
| `wait_for_endpoint_ready(endpoint_id, interval=1)` | Block until endpoint is `LIVE` |

---

## CLI Reference

The `flex_ai` CLI is installed automatically with the package.

```bash
# Datasets
flex_ai datasets list
flex_ai datasets create --name "My Dataset" --train-path train.jsonl --eval-path eval.jsonl

# Models
flex_ai models list

# Fine-tuning
flex_ai fine_tuning create \
  --name "My Fine-Tune" \
  --dataset-id <dataset_id> \
  --model "meta-llama/Llama-3.2-1B-Instruct" \
  --n-epochs 3 \
  --train-with-lora True \
  --lora-r 64 --lora-alpha 16 --lora-dropout 0.1 \
  --batch-size 8 \
  --learning-rate 2e-5 \
  --n-checkpoints-and-evaluations-per-epoch 2 \
  --save-only-best-checkpoint False

# Tasks
flex_ai tasks --id <task_id>

# Checkpoints
flex_ai checkpoints download --checkpoint-id <checkpoint_id>
flex_ai checkpoints download-gguf --checkpoint-id <checkpoint_id>
```

---

## Full End-to-End Example

```python
from flex_ai import FlexAI
from openai import OpenAI

# ── Auth ──────────────────────────────────────────────────────────────────────
# Set FLEX_AI_API_KEY in your environment — never hard-code secrets!
client = FlexAI()

# ── Dataset ───────────────────────────────────────────────────────────────────
dataset = client.create_dataset(
    name="support-tickets",
    train_path="data/train.jsonl",
    eval_path="data/eval.jsonl",
)
print("Dataset ID:", dataset["id"])

# ── Fine-tune ─────────────────────────────────────────────────────────────────
task = client.create_finetune(
    name="support-tickets-v1",
    dataset_id=dataset["id"],
    model="meta-llama/Llama-3.2-1B-Instruct",
    n_epochs=5,
    train_with_lora=True,
    lora_config={"lora_r": 64, "lora_alpha": 16, "lora_dropout": 0.05},
    batch_size=4,
    learning_rate=1e-4,
    n_checkpoints_and_evaluations_per_epoch=2,
    save_only_best_checkpoint=True,
)
print("Task ID:", task["id"])

# ── Wait for training ─────────────────────────────────────────────────────────
status = client.wait_for_task_completion(task_id=task["id"])
print("Final status:", status)  # COMPLETED

# ── Deploy endpoint ───────────────────────────────────────────────────────────
checkpoints = client.get_task_checkpoints(task_id=task["id"])
endpoint_id = client.create_multi_lora_endpoint(
    name="support-endpoint",
    lora_checkpoints=[{"id": c["id"], "name": f"step_{c['step']}"} for c in checkpoints],
    compute="A100-40GB",
    idle_timeout_seconds=300,
)
endpoint = client.wait_for_endpoint_ready(endpoint_id=endpoint_id)

# ── Inference ─────────────────────────────────────────────────────────────────
oai = OpenAI(api_key=client.api_key, base_url=f"{endpoint['url']}/v1")
response = oai.chat.completions.create(
    model="meta-llama/Llama-3.2-1B-Instruct",
    messages=[{"role": "user", "content": "How do I reset my password?"}],
)
print(response.choices[0].message.content)
```

---

## Security

| Secret | Where to store it | Where NOT to store it |
|--------|-------------------|-----------------------|
| `FLEX_AI_API_KEY` | Environment variable, secrets manager (AWS Secrets Manager, Doppler, etc.) | Source code, `.env` committed to git |
| `wandb_key` | Environment variable | Source code |

Add this to your `.gitignore`:

```
.env
*.env
```

---

## Links

| Resource | URL |
|----------|-----|
| Website | [getflex.ai](https://getflex.ai) |
| Dashboard | [app.getflex.ai](https://app.getflex.ai) |
| Documentation | [docs.getflex.ai](https://docs.getflex.ai) |
| API Keys | [app.getflex.ai/settings/api-keys](https://app.getflex.ai/settings/api-keys) |
| Colab Notebook | [Open in Colab](https://colab.research.google.com/drive/1nSFRjrHpbz372h7W-2G7jmTuh_Dhiv9x) |
| PyPI | [pypi.org/project/flex_ai](https://pypi.org/project/flex_ai/) |

---

## License

MIT © [Ariel Cohen](https://github.com/arielcohen4)
