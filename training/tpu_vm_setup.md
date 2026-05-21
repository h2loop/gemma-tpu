# TPU VM Setup Guide — JAX + Tunix + Gemma 4

Lessons learned from hands-on setup of a v6e-4 (Trillium, 1 worker, 4 chips × 32 GB = 128 GB HBM) for Tunix/Qwix/Flax-based LoRA fine-tuning.

---

## Key Facts Before You Start

| Thing | Reality |
|---|---|
| Python on TPU VM | Ships with **3.10** (both v5e and v6e) |
| Tunix requirement | **3.11+** |
| Fix | Install 3.11 via deadsnakes PPA — do this before anything else |
| `--worker=all` + `pkill` | Kills the SSH session itself (bash args match the pattern) — never combine |
| `--worker=all` for launch | Unreliable under parallel load; use a background `for` loop instead |
| Log visibility | Python uses block buffering when writing to file — always use `PYTHONUNBUFFERED=1` + `python3 -u` |
| GCS bucket | Must be in the same region as the TPU (`us-central1` for `us-central1-a`) |
| Spot instances | Cannot be stopped, only deleted — model should live in GCS, not on VM disk |
| Model loading | Tunix downloads model from GCS to `/tmp/models` before loading into HBM — needs disk space or `TMPDIR=/dev/shm` |
| `/dev/shm` | 355 GB RAM-backed tmpfs on v6e-4 — safe target for model staging; avoids disk pressure |
| Cross-project bucket | TPU service account from `YOUR_PROJECT` needs explicit IAM grant to read buckets in other projects |
| Underlying GCE instance | Named `t1v-n-XXXXXXXX-w-0` — lives in Google-managed project, cannot attach disks directly |
| Vision tower keys | 356 skipped keys on load (vision tower weights) — expected when loading text-only LoRA from multimodal checkpoint |
| `qwix.apply_lora_to_model` | Takes `(model, provider)` only — no `get_model_input()` call needed; auto-injects `rngs` for NNX models |

---

## Step 1 — Create GCS Bucket (skip if exists)

```bash
gcloud storage buckets create gs://YOUR_BUCKET \
  --project=YOUR_PROJECT \
  --location=us-central1 \
  --uniform-bucket-level-access
```

---

## Step 2 — Create TPU Instance

```bash
gcloud compute tpus tpu-vm create YOUR_TPU_NAME \
  --zone=us-central1-a \
  --project=YOUR_PROJECT \
  --accelerator-type=v5p-8 \
  --version=v2-alpha-tpuv5 \
  --spot
```

Wait for `Created tpu [YOUR_TPU_NAME]` (~3–5 min).

> Spot cannot be stopped — `delete` is the only way to deallocate. Model in GCS survives deletion.
> v6e-4 is a single-worker slice — no multi-host coordination needed.

---

## Step 3 — Add SSH Key

```bash
ssh-add ~/.ssh/google_compute_engine
```

Run this at the start of every local session. If SSH fails repeatedly, run it again.

---

## Step 4 — Install Python 3.11 (required — VM ships with 3.10)

```bash
gcloud compute tpus tpu-vm ssh YOUR_TPU_NAME \
  --zone=us-central1-a --project=YOUR_PROJECT --worker=0 \
  --ssh-flag="-o StrictHostKeyChecking=no -o ConnectTimeout=30" \
  --command="sudo add-apt-repository ppa:deadsnakes/ppa -y 2>/dev/null && \
             sudo apt-get update -qq && \
             sudo apt-get install -y python3.11 python3.11-venv python3.11-dev -qq && \
             python3.11 --version && echo py311-ok"
```

Should print `Python 3.11.x` and `py311-ok`.

---

## Step 5 — Install JAX + Tunix Stack

```bash
gcloud compute tpus tpu-vm ssh YOUR_TPU_NAME \
  --zone=us-central1-a --project=YOUR_PROJECT --worker=0 \
  --ssh-flag="-o StrictHostKeyChecking=no -o ConnectTimeout=30" \
  --command="python3.11 -m venv ~/.venv311 && \
             source ~/.venv311/bin/activate && \
             pip install -q --upgrade pip && \
             pip install -q 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && \
             pip install -q safetensors tensorflow tensorflow_datasets \
                            tensorboardX transformers grain datasets huggingface_hub \
                            'numpy>2' optax orbax-checkpoint gcsfs && \
             pip install -q git+https://github.com/google/tunix && \
             pip install -q git+https://github.com/google/qwix && \
             pip uninstall -q flax -y && \
             pip install -q git+https://github.com/google/flax && \
             python3.11 -c 'import jax; print(jax.__version__); print(len(jax.devices()), \"devices\")' && \
             echo pkg-ok"
```

Should print `jax x.x.x`, `4 devices`, and `pkg-ok`. Takes ~5 min.

---

## Step 6 — Push Training Script

```bash
gcloud compute tpus tpu-vm scp ~/gemma4_lora_sft.py YOUR_TPU_NAME:~/gemma4_lora_sft.py \
  --zone=us-central1-a --project=YOUR_PROJECT --worker=0
```

---

## Step 7 — Launch Training (single worker)

v6e-4 is a single-worker slice — no loop needed. Never put `pkill` and the launch command in the same SSH call.

`TMPDIR=/dev/shm` routes the model staging download to RAM (355 GB available) instead of the boot disk (which fills up at 97 GB with the 62.5 GB model download).

```bash
gcloud compute tpus tpu-vm ssh YOUR_TPU_NAME \
  --zone=us-central1-a --project=YOUR_PROJECT --worker=0 \
  --ssh-flag="-o StrictHostKeyChecking=no -o ConnectTimeout=30" \
  --command="source ~/.venv311/bin/activate && \
             PYTHONUNBUFFERED=1 PJRT_DEVICE=TPU TMPDIR=/dev/shm \
             nohup python3 -u ~/gemma4_lora_sft.py \
             </dev/null > ~/gemma4_lora_sft.log 2>&1 & echo launched-0"
```

Should print `launched-0`.

---

## Step 8 — Monitor

```bash
# Worker 0 log (JAX compiles for 3–8 min before step 1 appears)
gcloud compute tpus tpu-vm ssh YOUR_TPU_NAME \
  --zone=us-central1-a --project=YOUR_PROJECT --worker=0 \
  --ssh-flag="-o StrictHostKeyChecking=no" \
  --command="tail -40 ~/gemma4_lora_sft.log"
```

Expected log sequence:
```
JAX sees 4 devices: [TpuDevice(id=0, ...) ...]
Mesh: tp=4
Loading Gemma 4 31B from gs://YOUR_BUCKET/models/gemma-4-31b-it ...
Model loaded.
trainable params: ~50M / 31B (0.16%)
Compiling step fn (takes 3–8 min on first step) ...
step=1  loss=2.XXXX
step=20 loss=1.XXXX
...
Done. LoRA adapters saved to: gs://YOUR_BUCKET/checkpoints/lora-run-001
```

---

## Kill Training

Use a detached subshell — do NOT use `pkill` in the same command as other logic, it will match the current bash args and kill the session.

```bash
gcloud compute tpus tpu-vm ssh YOUR_TPU_NAME \
  --zone=us-central1-a --project=YOUR_PROJECT --worker=0 \
  --ssh-flag="-o StrictHostKeyChecking=no -o ConnectTimeout=20" \
  --command="nohup sh -c 'sleep 2 && pkill -9 -f gemma4_lora_sft' \
             </dev/null >/dev/null 2>&1 & echo sched-0"
```

---

## Delete TPU

```bash
gcloud compute tpus tpu-vm delete YOUR_TPU_NAME \
  --zone=us-central1-a --project=YOUR_PROJECT --quiet
```

GCS bucket and model survive. On next run, skip Steps 1–2 (bucket + TPU) if bucket exists, and skip model download (Step 4 in `gemma4_tpu_lora.md`) if model is already in GCS.

---

## Quick Reference

| Action | Command |
|---|---|
| Check TPU state | `gcloud compute tpus tpu-vm describe YOUR_TPU_NAME --zone=us-central1-a --project=YOUR_PROJECT --format="value(state,health)"` |
| Re-add SSH key | `ssh-add ~/.ssh/google_compute_engine` |
| Check running process | `... --worker=0 --command="ps aux \| grep '[g]emma4'"` |
| Check all workers clear | `... --worker=all --command="ps aux \| grep '[p]ython3' \| grep -v system"` |
| Tail log | `... --worker=0 --command="tail -40 ~/gemma4_lora_sft.log"` |
| Check GCS model | `gsutil ls gs://YOUR_BUCKET/models/gemma-4-31b-it/` |
| Check GCS checkpoints | `gsutil ls gs://YOUR_BUCKET/checkpoints/lora-run-001/` |

---

## Gotchas

1. **`--worker=all` + `pkill`** — `pkill -f script_name` matches the bash subprocess whose args contain the script name. It kills the SSH session. Use `nohup sh -c 'sleep 2 && pkill'` to detach, or run as a separate SSH call after the launch.

2. **0-byte log file** — Python block-buffers stdout when writing to a file. Always pass `PYTHONUNBUFFERED=1` and `python3 -u`. Without this you'll see nothing until the buffer fills.

3. **"TPU already in use"** — A prior process is holding `libtpu`. Kill it with the detached pkill pattern above, wait ~5s, then relaunch.

4. **Multi-host rendezvous** — Not applicable for v6e-4 (single worker). No parallel launch loop needed.

5. **Spot preemption** — Spot TPUs give no warning. Keep checkpoints writing to GCS (`orbax-checkpoint` with a GCS path) so you can resume from the last step.

6. **Boot disk fills on model load** — Tunix stages the model in `/tmp/models` before loading into HBM. For a 62.5 GB model on a 97 GB boot disk (85 GB already used), this fills the disk and crashes mid-traceback. Fix: `TMPDIR=/dev/shm` at launch. Also clear `/tmp/models` before retrying: `sudo rm -rf /tmp/models`.

7. **Cross-project GCS access** — The TPU service account is scoped to `YOUR_PROJECT`. If the model bucket lives in another project, grant access explicitly:
   ```bash
   gsutil iam ch serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com:roles/storage.objectViewer gs://BUCKET
   ```
   Get the project number from the error message (`... does not have storage.objects.list access`).

8. **`qwix.apply_lora_to_model` API** — Call as `apply_lora_to_model(model, provider)` with no extra args. There is no `get_model_input()` on the Gemma4 object. The function auto-injects `rngs=nnx.Rngs(10003)` for NNX models.
