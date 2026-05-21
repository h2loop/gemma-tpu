# TPU v6e-8 Inference Setup — Gemma 4 31B via vLLM

## Instance

| Field | Value |
|-------|-------|
| Name | `gemma4-inference` |
| Zone | `asia-northeast1-b` |
| Type | `v6e-8` (8 chips × 31.25 GiB HBM = 250 GiB) |
| Runtime | **`v2-alpha-tpuv6e`** (not `tpu-ubuntu2204-base`) |
| Preemptible | yes |
| Project | `YOUR_PROJECT` |

```bash
gcloud compute tpus tpu-vm create gemma4-inference \
  --zone=asia-northeast1-b \
  --accelerator-type=v6e-8 \
  --version=v2-alpha-tpuv6e \
  --preemptible
```

## SSH Config

```
Host gemma4-inference
  HostName <EXTERNAL_IP>
  User azek
  IdentityFile ~/.ssh/google_compute_engine
  StrictHostKeyChecking no
```

Get IP:
```bash
gcloud compute tpus tpu-vm describe gemma4-inference \
  --zone=asia-northeast1-b \
  --format="value(networkEndpoints[0].accessConfig.externalIp)"
```

## Bootstrap (run once after instance creation)

```bash
# Enable hugepages (improves TPU startup)
ssh gemma4-inference 'sudo sh -c "echo always > /sys/kernel/mm/transparent_hugepage/enabled"'

# Install Python 3.11
ssh gemma4-inference 'sudo add-apt-repository ppa:deadsnakes/ppa -y && \
  sudo apt-get update -qq && \
  sudo apt-get install -y python3.11 python3.11-venv python3.11-dev -qq'

# Create venv and verify TPU
ssh gemma4-inference 'python3.11 -m venv ~/.venv311 && \
  source ~/.venv311/bin/activate && \
  pip install -q --upgrade pip && \
  pip install -q "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && \
  python3 -c "import jax; print(jax.devices())"'
# Should print 8 TpuDevice entries

# Install vllm-tpu (for bench client) and tpu-info
ssh gemma4-inference 'source ~/.venv311/bin/activate && \
  pip install -q vllm-tpu tpu-info'
```

## Serving (Docker)

The only working approach for Gemma 4 on TPU is the official Docker image:

```bash
ssh gemma4-inference 'sudo docker pull vllm/vllm-tpu:gemma4'

ssh gemma4-inference 'sudo docker run -itd --name gemma4-tpu \
  --privileged --network host \
  --shm-size 16G -v /dev/shm:/dev/shm \
  -e HF_TOKEN=$HF_TOKEN \
  --entrypoint vllm vllm/vllm-tpu:gemma4 \
  serve google/gemma-4-31B-it \
  --tensor-parallel-size 8 \
  --max-model-len 4096 \
  --disable_chunked_mm_input \
  --download-dir /dev/shm \
  --host 0.0.0.0 --port 8000'
```

Startup takes ~10 minutes (model download + XLA compilation for bucket sizes 16→2048).

Check readiness:
```bash
ssh gemma4-inference 'curl -s http://localhost:8000/v1/models'
```

Monitor TPU during serving:
```bash
ssh gemma4-inference 'source ~/.venv311/bin/activate && tpu-info'
```

## Benchmarking

### Max throughput config (sweet spot)
```bash
ssh gemma4-inference 'source ~/.venv311/bin/activate && \
  vllm bench serve --backend vllm --base-url http://localhost:8000 \
  --model google/gemma-4-31B-it --dataset-name random \
  --num-prompts 128 --random-input-len 512 --random-output-len 512 \
  --request-rate inf'
```

### Low-latency config (matches original baseline)
```bash
ssh gemma4-inference 'source ~/.venv311/bin/activate && \
  vllm bench serve --backend vllm --base-url http://localhost:8000 \
  --model google/gemma-4-31B-it --dataset-name random \
  --num-prompts 64 --random-input-len 128 --random-output-len 256 \
  --request-rate 10000 --temperature 0'
```

## Benchmark Results

### Run configurations

| Run | Prompts | Input len | Output len | Rate | Purpose |
|-----|---------|-----------|------------|------|---------|
| 1 | 64 | 256 | 256 | inf | Baseline |
| 2 | 128 | 512 | 512 | inf | **Max throughput** |
| 3 | 256 | 1024 | 1024 | inf | Saturation test |
| 4 | 64 | 128 | 256 | 10000 | Low-latency baseline |

### Results

| Metric | Run 1 | Run 2 (max) | Run 3 | Run 4 (low-lat) |
|--------|-------|-------------|-------|------------------|
| Output tok/s | 2,245 | **3,085** | 2,339 | 3,253 |
| Peak output tok/s | 2,560 | **3,456** | 3,136 | 3,520 |
| Total tok/s | 4,490 | **6,169** | 4,679 | 4,879 |
| TTFT median (ms) | 601 | 793 | 2,936 | 235 |
| TPOT median (ms) | 26 | 39 | 40 | **18.7** |
| P99 ITL (ms) | 71 | 83 | 86 | 22 |

### TPU utilization at peak (Run 3, 256 concurrent)

| Metric | Value |
|--------|-------|
| HBM usage | 28.69 / 31.25 GiB per chip (92%) |
| Duty cycle | 98% all chips |
| TensorCore utilization | 19.3% (memory-bound, expected for LLM decode) |

## Key Findings

1. **Peak sustained throughput: 3,456 output tok/s** at 128 concurrent requests
2. **Best total throughput: 6,169 tok/s** (input + output combined)
3. **Lowest latency: 18.7ms TPOT** at 64 concurrent, shorter sequences
4. **Saturation point:** >128 concurrent requests causes KV cache pressure → TTFT spikes
5. **TensorCore at 19%** is normal — autoregressive decode is memory-bandwidth bound, not compute bound

## Gotchas

1. **Runtime must be `v2-alpha-tpuv6e`** — `tpu-ubuntu2204-base` gives "Failed to get global TPU topology"
2. **Docker required for Gemma 4** — pip vllm-tpu has version conflicts (jax 0.9.2 not on PyPI, transformers/hub breaks local paths)
3. **Use `--entrypoint vllm`** — default entrypoint in the image expects args in a different format
4. **Model via HF repo ID only** — local path (`/dev/shm/...`) is rejected by huggingface_hub validator
5. **Preemptible = will be deleted** — model lives in GCS (`gs://YOUR_BUCKET/models/gemma-4-31b-it/`), not on VM
6. **v6e-8 capacity is scarce** — asia-northeast1-b was the only zone with availability across all tested regions
7. **XLA compilation on first start: ~8 min** — compiles for each padded token bucket size
8. **`tpu-info` shows PID=None** for Docker workloads — use duty cycle/HBM metrics instead

## Cleanup

```bash
gcloud compute tpus tpu-vm delete gemma4-inference --zone=asia-northeast1-b --quiet
```
