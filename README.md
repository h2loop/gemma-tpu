## Fine-Tuning Gemma on Google Cloud TPU

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Jatin Kishnani &nbsp;&nbsp; Mayank Goyel &nbsp;&nbsp; Amit Singh &nbsp;&nbsp; Pulkit Agrawal

**Paper:** [`paper/technical_paper.pdf`](paper/technical_paper.pdf)

----

A production-ready recipe for LoRA fine-tuning Gemma models on Google Cloud TPU, with vLLM inference support. Bring your own dataset — works with any JSONL, HuggingFace dataset, or chat-format data.

**Benchmark results (Gemma 4 31B, CodeV-R1 dataset):**
- Training: **1.61x faster**, **2.12x cheaper** on TPU v5p-8 vs 2xH100
- Inference: **66% higher throughput**, **23.6x faster TTFT** at 4096-token context on TPU v6e-8

----

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Build
docker build -t gemma-tpu .

# Run on TPU VM
docker run --privileged --network host \
  -v /dev/shm:/dev/shm \
  -v $(pwd)/configs:/app/configs \
  gemma-tpu python training/train.py --config configs/your_config.yaml
```

### Option 2: Manual Setup

```bash
# Provision TPU
gcloud compute tpus tpu-vm create my-tpu \
  --zone=us-central1-a \
  --accelerator-type=v5p-8 \
  --version=v2-alpha-tpuv5

# Install dependencies
pip install 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -r training/requirements.txt
pip install git+https://github.com/google/tunix git+https://github.com/google/qwix

# Train with config
python training/train.py --config configs/default.yaml
```

----

## Bring Your Own Dataset

Create a YAML config pointing to your data:

```yaml
# configs/my_dataset.yaml
dataset:
  type: "jsonl"  # or "huggingface"
  path: "gs://my-bucket/data/train.jsonl"  # or "username/dataset-name"
  columns:
    prompt: "instruction"   # your prompt column
    response: "output"      # your response column
  system_prompt: "You are a helpful assistant."  # optional
```

**Supported formats:**
- **JSONL**: `{"prompt": "...", "response": "..."}`
- **HuggingFace**: Any dataset with configurable column mapping
- **Chat format**: `{"messages": [{"role": "user", "content": "..."}, ...]}`

See [`configs/default.yaml`](configs/default.yaml) for all options.

----

## TPU Cost & Time Estimates (Gemma 4 31B)

| TPU Type | Chips | HBM | Training Time | On-Demand $/hr |
|----------|-------|-----|---------------|----------------|
| v5p-8 | 4 | 411 GB | ~3.3 hr | $16.80 |
| v6e-8 | 8 | 250 GB | ~2.8 hr | $21.52 |

*Times for 10K samples, seq_len=3072, batch=8. Spot instances are 60-70% cheaper. Script currently supports Gemma 4 31B only.*

----

## Checkpoint Resume (Spot Preemption)

Checkpoints save to GCS automatically. To resume after preemption:

```yaml
# In your config
checkpointing:
  output_dir: "gs://my-bucket/checkpoints/run-001"
  resume_from: "latest"  # or specific step number like 500
```

Or via CLI:
```bash
python training/train.py --config configs/my_config.yaml --resume latest
```

----

## Merge & Push to HuggingFace Hub

```bash
# Merge LoRA into base model
python training/orbax_to_peft.py \
  --base-model google/gemma-4-31b-it \
  --ckpt-dir gs://my-bucket/checkpoints/run-001 \
  --ckpt-step 1244 \
  --output-dir ./merged_model

# Push to Hub
python training/orbax_to_peft.py \
  --base-model google/gemma-4-31b-it \
  --ckpt-dir gs://my-bucket/checkpoints/run-001 \
  --ckpt-step 1244 \
  --push-to-hub your-username/my-finetuned-gemma \
  --hf-token $HF_TOKEN
```

----

## Example Output

**Prompt:** Write a Verilog module for a 4-bit counter with enable and reset.

**Response (fine-tuned on CodeV-R1):**
```verilog
module counter_4bit (
    input wire clk,
    input wire rst_n,
    input wire enable,
    output reg [3:0] count
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        count <= 4'b0000;
    else if (enable)
        count <= count + 1'b1;
end

endmodule
```

----

## Repository Structure

```
gemma-tpu/
├── configs/                    # YAML configuration files
│   ├── default.yaml            # Template config with all options
│   └── codev_verilog.yaml      # Config used in paper benchmarks
├── training/
│   ├── train.py                # Main entry point (CLI)
│   ├── gemma4_lora_sft.py      # Core training logic
│   ├── orbax_to_peft.py        # Checkpoint merge + Hub upload
│   └── requirements.txt
├── eval/
│   └── evaluate_tpu.py         # verilog-eval benchmark
├── inference/
│   └── tpu_v6e8_inference_setup.md
├── paper/
│   └── technical_paper.pdf     # Full technical report
├── Dockerfile
├── CONTRIBUTING.md
└── LICENSE                     # Apache 2.0
```

----

## Citation

```bibtex
@techreport{kishnani2026tpugemma4,
  title={Fine-Tuning and Serving Gemma 4 on Google Cloud TPU},
  author={Jatin Kishnani and Mayank Goyel and Amit Singh and Pulkit Agrawal},
  institution={H2LooP AI},
  year={2026},
  url={https://github.com/h2loop/gemma-tpu}
}
```

----

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Areas we'd love help with:**
- Dataset adapters (ShareGPT, Alpaca formats)
- Benchmarks on other TPU types (v5e, v4)
- Gemma 9B/12B configurations
- Improved documentation

----

## License

[Apache 2.0](LICENSE)

**Disclaimer:** Gemma models are developed by Google DeepMind. We do not claim ownership or affiliation. This is independent research.
