# Fine-Tuning & Serving Gemma 4 31B on Google Cloud TPU

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2605.25645-b31b1b.svg)](https://arxiv.org/abs/2605.25645)

**Authors:** Jatin Kishnani · Mayank Goyel · Amit Singh · Pulkit Agrawal — *H2LooP AI*

📄 **Technical paper:** [arXiv:2605.25645](https://arxiv.org/abs/2605.25645)

---

## 1. About

A **production-ready, reproducible recipe** for **LoRA supervised fine-tuning of Gemma 4 31B on Google Cloud TPU** (v5p-8), with **vLLM-based inference serving** on TPU (v6e-8 / Trillium).

Most LLM fine-tuning tooling targets GPUs (PyTorch + HuggingFace + FSDP). This repo documents and packages the **JAX + Tunix/Qwix** path on TPU — including the non-obvious porting work (mesh/sharding configuration, LoRA module-name mapping, gradient checkpointing, an Orbax→safetensors merge step, and the vLLM-TPU serving setup) — so you can train and serve a 31B model on TPU end-to-end.

**What you get:**
- LoRA SFT on a frozen 31B base (only ~1.5% of params train) — tensor-parallel sharded across TPU chips.
- Bring-your-own-dataset support (JSONL, HuggingFace, or the included CodeV Verilog pipeline).
- Spot-preemption-safe checkpointing to GCS.
- A merge step that folds LoRA back into the base → a standard HuggingFace-format model.
- vLLM-TPU serving + an OpenAI-compatible benchmark client.

> **Stack:** JAX · Tunix · Qwix · Flax/NNX · Orbax · vLLM-TPU. No PyTorch / torch_xla.

---

## 2. Prerequisites

| Requirement | Notes |
|---|---|
| **Google Cloud project** with TPU quota | `v5p-8` (4 chips) for training; `v6e-8` (8 chips) for inference |
| **`gcloud` CLI** authenticated | `gcloud auth login` + project set |
| **Python 3.11+** | TPU VMs ship with **3.10** — install 3.11 (Tunix requires it) |
| **HuggingFace account** | Accept the Gemma license, create an access token (`$HF_TOKEN`) |
| **GCS bucket** in the **same region** as the TPU | Stores the model weights + checkpoints |
| **Docker** | Required for vLLM-TPU inference (and optional for training) |

---

## 3. Quick Start

> ⚠️ **Spot instances cannot be stopped — only deleted — so keep your model and checkpoints in GCS, not on the VM disk.**

### Provision a TPU
```bash
# Create a GCS bucket in the TPU's region (model + checkpoints live here)
gcloud storage buckets create gs://YOUR_BUCKET --location=us-central1 --uniform-bucket-level-access

# Provision the training TPU (spot is 60–70% cheaper)
gcloud compute tpus tpu-vm create my-tpu \
  --zone=us-central1-a --accelerator-type=v5p-8 --version=v2-alpha-tpuv5 --spot
```

### Install dependencies
TPU VMs ship Python 3.10; install 3.11 first, then the stack:
```bash
# Python 3.11 (Tunix requires it)
sudo add-apt-repository ppa:deadsnakes/ppa -y && sudo apt-get update -qq
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev

python3.11 -m venv ~/.venv311 && source ~/.venv311/bin/activate

# JAX for TPU
pip install 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Project deps + Tunix/Qwix/Flax from source
pip install -r training/requirements.txt
pip install git+https://github.com/google/tunix git+https://github.com/google/qwix
pip uninstall -y flax && pip install git+https://github.com/google/flax
```

### Configure paths and train
Set your model, checkpoint, and tokenizer paths at the top of
[`training/gemma4_lora_sft.py`](training/gemma4_lora_sft.py):
```python
MODEL_GCS_PATH = "gs://YOUR_BUCKET/models/gemma-4-31b-it"   # or a local dir
CKPT_DIR       = "gs://YOUR_BUCKET/checkpoints/lora-run-001"
TOKENIZER_PATH = "gs://gemma-data/tokenizers/tokenizer_gemma3.model"
```
Then launch (route model staging to RAM-backed `/dev/shm` so the boot disk doesn't fill):
```bash
PYTHONUNBUFFERED=1 PJRT_DEVICE=TPU TMPDIR=/dev/shm \
  python3 -u training/gemma4_lora_sft.py
```
The first step compiles for 3–8 min (XLA), then per-step loss is logged. The mesh auto-selects
tensor-parallelism from the chip count (`tp=4` on v5p-8).

> **Tip:** a `Dockerfile` is provided for a reproducible container — `docker build -t gemma-tpu .`,
> then run with `--privileged --network host -v /dev/shm:/dev/shm -v /path/to/model:/mnt/data`.

---

## 4. Bring Your Own Dataset

The pipeline works with any instruction/response data. Point a config at your source and map the columns:

```yaml
# configs/my_dataset.yaml
dataset:
  type: "jsonl"            # "jsonl" | "huggingface" | "codev"
  path: "gs://my-bucket/data/train.jsonl"   # or "username/dataset-name" for HF
  columns:
    prompt:   "instruction"   # your input column
    response: "output"        # your target column
  system_prompt: "You are a helpful assistant."   # optional
```

**Supported formats:**
- **`jsonl`** — line-delimited `{"prompt": "...", "response": "..."}` (local path or `gs://`). Chat-format `{"messages": [...]}` is also accepted.
- **`huggingface`** — any Hub dataset, with configurable column mapping.
- **`codev`** — the built-in [CodeV-R1](https://huggingface.co/datasets/zhuyaoyu/CodeV-R1-dataset) Verilog pipeline (`training/codev_dataset.py`): wraps each example with a Verilog system prompt, strips `<think>…</think>` reasoning, keeps the ` ```verilog ` block, and computes an **assistant-only loss mask**.

See [`configs/default.yaml`](configs/default.yaml) for all options.

---

## 5. Checkpoints & Resume

Checkpoints are written (Orbax format) to the configured `CKPT_DIR` / `output_dir` — **use a `gs://` path on spot instances** so they survive preemption.

```yaml
checkpointing:
  output_dir: "gs://YOUR_BUCKET/checkpoints/run-001"
  save_every_steps: 100      # checkpoint cadence
  keep_last_n: 3             # retain last N, older ones pruned
  resume_from: "latest"      # or a specific step number, e.g. 500
```

To resume after a preemption, point the run at the same `output_dir` with `resume_from: latest`.

---

## 6. Merge LoRA → Standalone Model

LoRA adapters are merged back into the base weights (`W + (A·B)·α/r`) to produce a plain
HuggingFace-format model — directly loadable by Tunix or vLLM, with no qwix wrappers.

The merge math runs on **CPU**, so it does **not** need the TPU (set `JAX_PLATFORMS=cpu` to
run it alongside a training job or on a cheap VM):

```bash
# Merge a checkpoint into the base model
JAX_PLATFORMS=cpu python3 training/orbax_to_peft.py \
  --base-model /path/to/gemma-4-31b-it \
  --ckpt-dir   gs://YOUR_BUCKET/checkpoints/run-001 \
  --ckpt-step  latest \
  --output-dir ./merged_model
```

Push straight to the HuggingFace Hub (creates the repo if needed):
```bash
python3 training/orbax_to_peft.py \
  --base-model /path/to/gemma-4-31b-it \
  --ckpt-dir   gs://YOUR_BUCKET/checkpoints/run-001 \
  --ckpt-step  latest \
  --output-dir ./merged_model \
  --push-to-hub your-username/my-finetuned-gemma --private \
  --hf-token $HF_TOKEN
```

> Use a **local** `--base-model` path if you've already downloaded the weights (avoids
> re-downloading the gated model). `--ckpt-step latest` resolves to the newest checkpoint.

---

## 7. Serving & Inference (vLLM on TPU)

Serve the merged model with the official `vllm/vllm-tpu` Docker image. **Set
`--tensor-parallel-size` to the chip count** (4 on v5p-8, 8 on v6e-8):

```bash
sudo docker run -itd --name gemma4-serve --privileged --network host \
  --shm-size 16G -v /dev/shm:/dev/shm -v /path/to/merged_model:/model \
  --entrypoint vllm vllm/vllm-tpu:gemma4 \
  serve /model \
  --tensor-parallel-size 4 \
  --max-model-len 4096 \
  --disable_chunked_mm_input \
  --host 0.0.0.0 --port 8000
```

Startup takes ~8–10 min (model load + per-token-bucket XLA compilation). It then exposes an
OpenAI-compatible API:
```bash
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "/model",
  "messages": [{"role": "user", "content": "Write a Verilog 4-bit counter with enable and reset."}]
}'
```

Benchmark it with the included client ([`inference/benchmark_inference.py`](inference/benchmark_inference.py)):
```bash
python3 inference/benchmark_inference.py vllm --url http://localhost:8000 \
  --model /model --concurrency 8 --runs 3 --max-tokens 256
```

See [`inference/tpu_v6e8_inference_setup.md`](inference/tpu_v6e8_inference_setup.md) for the full v6e-8 setup and gotchas.

---

## 8. Benchmarks

Gemma 4 31B, CodeV-R1 dataset, TPU vs **2× H100 80GB** baseline (identical hyperparameters):

| Workload | Result |
|---|---|
| **Training** — v5p-8 vs 2× H100 | **1.61× faster**, **2.12× cheaper** |
| **Inference** — v6e-8 vs 2× H100, 4096-token context | **66% higher throughput**, **23.6× faster TTFT** (61 ms vs 1,443 ms at QPS=4) |
| **End-to-end** — train + serve, short context | **1.82× cheaper** on TPU |

At long context the TPU advantage grows — driven by higher aggregate compute and 2× more KV-cache
capacity (250 GB vs 160 GB total HBM). Full methodology, QPS sweeps (512 → 16k tokens), and cost
analysis are in the **technical paper**: 📄 [arXiv:2605.25645](https://arxiv.org/abs/2605.25645).

*Dataset: CodeV-R1 ([arXiv:2407.10424](https://arxiv.org/abs/2407.10424)).*

---

## 9. Repository Structure

```
gemma-tpu/
├── configs/
│   ├── default.yaml             # Template config with all options
│   └── codev_verilog.yaml       # Config used in paper benchmarks
├── training/
│   ├── gemma4_lora_sft.py       # Core LoRA SFT script (configure paths at top)
│   ├── train.py                 # CLI wrapper
│   ├── orbax_to_peft.py         # Merge LoRA → base + optional Hub push
│   ├── codev_dataset.py         # CodeV-R1 Verilog data pipeline
│   ├── tpu_vm_setup.md          # Step-by-step TPU VM setup guide
│   └── requirements.txt
├── inference/
│   ├── benchmark_inference.py   # Tunix + vLLM benchmark client
│   ├── tpu_v6e8_inference_setup.md
│   └── inference_results.txt    # Measured TPU vs GPU numbers
├── eval/evaluate_tpu.py         # verilog-eval benchmark
├── Dockerfile
└── LICENSE                      # Apache 2.0
```

---

## 10. Citation

If you use this work, please cite the technical report:

```bibtex
@article{kishnani2026tpugemma4,
  title         = {Fine-Tuning and Serving Gemma 4 31B on Google Cloud TPU:
                   A Technical Comparison with GPU Baselines},
  author        = {Kishnani, Jatin and Goyel, Mayank and Singh, Amit and Agrawal, Pulkit and Mishra, Sairanjan},
  year          = {2026},
  month         = may,
  eprint        = {2605.25645},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2605.25645}
}
```

---

## Contributing

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Areas we'd love help with:
- Dataset adapters (ShareGPT, Alpaca formats)
- Benchmarks on other TPU types (v5e, v4)
- Support for additional Gemma sizes (9B/12B — currently 31B only)
- Improved documentation

---

## License

[Apache 2.0](LICENSE)

**Disclaimer:** Gemma models are developed by Google DeepMind. We claim no ownership or affiliation. This is independent research.
