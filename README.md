<p align="center">
  <strong>sera-tpu</strong>
</p>

## Fine-Tuning and Serving Gemma 4 31B on Google Cloud TPU: _A Technical Comparison with GPU Baselines_

Jatin Kishnani &nbsp;&nbsp; Mayank Goyel &nbsp;&nbsp; Amit Singh &nbsp;&nbsp; Pulkit Agrawal

**Paper:** [`paper/technical_paper.pdf`](paper/technical_paper.pdf) | **License:** [H2LooP Research Only (ROL)](LICENSE)

----

This repository contains the complete source code to replicate the findings in our technical report on LoRA fine-tuning Gemma 4 31B on TPU v5p-8 and serving with vLLM on TPU v6e-8 (Trillium), benchmarked against a 2xH100 GPU baseline.

**Key results:**
- Training: **1.61x faster**, **2.12x cheaper** on TPU v5p-8 vs 2xH100
- Inference (4096-token context): **66% higher throughput**, **23.6x faster TTFT** on TPU v6e-8
- Eval: pass@1 = 0.641 (TPU) vs 0.697 (GPU) on verilog-eval spec-to-RTL (156 problems)

----

## Repository Structure

```
sera-tpu/
├── training/                   # TPU training recipe
│   ├── gemma4_lora_sft.py      # Main training script (JAX + Tunix + Qwix)
│   ├── codev_dataset.py        # CodeV-R1 data pipeline (Grain-based)
│   ├── orbax_to_peft.py        # Orbax checkpoint → merged safetensors
│   ├── trajectory_dataset.py   # Dataset utilities
│   ├── analyze_seqlen.py       # Sequence length analysis for max_seq_len selection
│   ├── tpu_vm_setup.md         # TPU VM provisioning & environment guide
│   └── requirements.txt        # Python dependencies
├── eval/
│   └── evaluate_tpu.py         # verilog-eval pass@k evaluation on TPU
├── inference/
│   ├── benchmark_inference.py  # vLLM bench serve wrapper
│   ├── inference_results.txt   # Raw benchmark outputs (TPU + GPU)
│   └── tpu_v6e8_inference_setup.md  # v6e-8 Trillium vLLM setup guide
├── paper/
│   ├── technical_paper.tex     # Full LaTeX source
│   ├── technical_paper.pdf     # Compiled PDF (17 pages)
│   ├── plot_report.py          # Generate all figures
│   └── plot_curves.py          # Training curve plots
├── LICENSE                     # H2LooP Research Only License
└── README.md
```

## Quick Start

### 1. Provision TPU VM

```bash
gcloud compute tpus tpu-vm create sera-tpu \
  --zone=us-central1-a \
  --project=YOUR_PROJECT \
  --accelerator-type=v5p-8 \
  --version=v2-alpha-tpuv5 \
  --spot
```

See [`training/tpu_vm_setup.md`](training/tpu_vm_setup.md) for full setup (Python 3.11, JAX, Tunix stack).

### 2. Train

```bash
PYTHONUNBUFFERED=1 PJRT_DEVICE=TPU TMPDIR=/dev/shm \
  python3 -u training/gemma4_lora_sft.py
```

Checkpoints save to GCS every 100 steps. Training completes in ~3.3 hours on v5p-8.

### 3. Merge Checkpoint

```bash
python3 training/orbax_to_peft.py \
  --base-model gs://h2loop-gemma4/models/gemma-4-31b-it \
  --ckpt-dir gs://h2loop-gemma4/checkpoints/lora-run-001 \
  --ckpt-step 1244 \
  --output-dir /dev/shm/merged_gemma4_31b
```

### 4. Evaluate

```bash
python3 eval/evaluate_tpu.py \
  --model-dir /dev/shm/merged_gemma4_31b \
  --n-samples 5 --temperature 0.8
```

### 5. Serve with vLLM (TPU v6e-8)

See [`inference/tpu_v6e8_inference_setup.md`](inference/tpu_v6e8_inference_setup.md) for the full Docker setup.

```bash
sudo docker run -itd --name gemma4-tpu \
  --privileged --network host \
  --shm-size 16G -v /dev/shm:/dev/shm \
  --entrypoint vllm vllm/vllm-tpu:gemma4 \
  serve google/gemma-4-31B-it \
  --tensor-parallel-size 8 \
  --max-model-len 16384 \
  --disable_chunked_mm_input
```

----

## Installation

```bash
# JAX for TPU (install first)
pip install 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Dependencies
pip install -r training/requirements.txt

# Tunix / Qwix / Flax (from source)
pip install git+https://github.com/google/tunix
pip install git+https://github.com/google/qwix
pip uninstall flax -y && pip install git+https://github.com/google/flax
```

----

## Citation & Reading More

If you use this work in a research paper, please cite:

```bibtex
@techreport{kishnani2026tpugemma4,
  title={Fine-Tuning and Serving Gemma 4 31B on Google Cloud TPU: A Technical Comparison with GPU Baselines},
  author={Kishnani, Jatin and Goyel, Mayank and Singh, Amit and Agrawal, Pulkit},
  institution={H2LooP AI},
  year={2026},
  month={May},
  note={LoRA SFT, Checkpoint Conversion, and vLLM Inference on TPU v5p and v6e (Trillium)},
  url={https://github.com/h2loop/gemma-tpu}
}
```

**Related:**
- [Gemma 4 Technical Report](https://ai.google.dev/gemma) — Google DeepMind, 2025
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al., ICLR 2022
- [vLLM: Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — Kwon et al., SOSP 2023
- [CodeV: Empowering LLMs for Verilog Generation](https://arxiv.org/abs/2407.10424) — Zhu et al., 2024
- [VerilogEval: Evaluating LLMs for Verilog Code Generation](https://arxiv.org/abs/2309.07544) — Liu et al., ICCAD 2023

----

## License

Released under the [H2LooP Research Only License (ROL)](LICENSE). See LICENSE for full terms.

**Disclaimer:** Google Gemma 4 31B is developed by Google DeepMind. JAX, Tunix, Qwix, and Flax are Google open-source projects. vLLM is developed by UC Berkeley et al. We do not claim ownership or affiliation with these projects. This work is independent research and should not be interpreted as endorsement or collaboration with any model provider.
