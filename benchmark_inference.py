"""
Gemma 4 31B — Inference Benchmark (TPU vs GPU comparison).

Two-phase approach:
  Phase 1: Tunix sampler (JAX-native, already works on your TPU setup)
           Quick functional test + baseline TPU numbers.
  Phase 2: vLLM OpenAI-compatible server benchmark client.
           Use this against vLLM running on EITHER TPU or GPU for
           a fair apples-to-apples hardware comparison.

Usage:
  # Phase 1 — Tunix sampler on TPU (quick, no extra deps)
  python benchmark_inference.py tunix

  # Phase 2 — benchmark against a running vLLM server
  python benchmark_inference.py vllm --url http://localhost:8000 --concurrency 1
  python benchmark_inference.py vllm --url http://localhost:8000 --concurrency 8
  python benchmark_inference.py vllm --url http://localhost:8000 --concurrency 64

  # Compare results
  python benchmark_inference.py compare --tpu results_tpu.json --gpu results_gpu.json
"""

import argparse
import json
import os
import statistics
import time
import datetime
import sys

# ═══════════════════════════════════════════════════════════════════════════════
# Shared utilities
# ═══════════════════════════════════════════════════════════════════════════════

_T0 = time.time()

def log(msg: str) -> None:
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


BENCH_PROMPTS = {
    "short": "Explain what a TPU is in one paragraph.",
    "medium": (
        "You are a senior software engineer. Write a detailed code review "
        "for a Python function that implements binary search on a sorted list. "
        "Cover edge cases, time complexity, and suggest improvements."
    ),
    "long": (
        "You are an expert systems architect. Design a complete microservices "
        "architecture for a real-time collaborative document editor similar to "
        "Google Docs. Include: service decomposition, data flow diagrams described "
        "in text, conflict resolution strategy using CRDTs, WebSocket gateway design, "
        "authentication and authorization flow, database choices for each service, "
        "caching strategy, deployment topology on Kubernetes, observability stack, "
        "and a phased rollout plan. Be thorough and specific. "
        "Also discuss trade-offs between operational transforms and CRDTs, "
        "how to handle offline editing and sync, cursor presence broadcasting, "
        "and rate limiting strategies for the API gateway."
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Tunix sampler (JAX-native TPU baseline)
# ═══════════════════════════════════════════════════════════════════════════════

def run_tunix_benchmark(args):
    """Benchmark using Tunix sampler directly on TPU. No server needed."""
    import threading
    os.environ["TMPDIR"] = "/dev/shm"

    import jax
    import jax.numpy as jnp
    from flax import nnx
    from tunix.models.gemma4 import model as gemma4_model_lib
    from tunix.models.gemma4 import params_safetensors as params_lib
    from tunix.generate import sampler as sampler_lib
    from tunix.generate import tokenizer_adapter as tokenizer_lib

    MODEL_GCS_PATH = "gs://h2loop-gemma4/models/gemma-4-31b-it"
    TOKENIZER_PATH = "gs://gemma-data/tokenizers/tokenizer_gemma3.model"

    log("=" * 60)
    log("Phase 1: Tunix Sampler — TPU Baseline")
    log("=" * 60)

    num_tpus = len(jax.devices())
    log(f"JAX {jax.__version__} — {num_tpus}x {jax.devices()[0].device_kind}")

    if num_tpus == 8:
        mesh_counts = (2, 4)
    elif num_tpus == 4:
        mesh_counts = (1, 4)
    elif num_tpus == 1:
        mesh_counts = (1, 1)
    else:
        mesh_counts = (2, num_tpus // 2)

    mesh = jax.make_mesh(
        mesh_counts, ("fsdp", "tp"),
        axis_types=(jax.sharding.AxisType.Auto,) * 2,
    )

    model_config = gemma4_model_lib.ModelConfig.gemma4_31b()
    log("Loading model...")
    t_load_start = time.perf_counter()
    with mesh:
        model = params_lib.create_model_from_safe_tensors(
            MODEL_GCS_PATH, model_config, mesh, dtype=jnp.bfloat16
        )
    t_load = time.perf_counter() - t_load_start
    log(f"Model loaded in {t_load:.1f}s")

    tokenizer = tokenizer_lib.Tokenizer(tokenizer_path=TOKENIZER_PATH)

    log("Initializing sampler...")
    with mesh:
        sampler = sampler_lib.Sampler(model=model, tokenizer=tokenizer)

    results = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "engine": "tunix_sampler",
        "hardware": {
            "type": "TPU",
            "num_devices": num_tpus,
            "device_kind": str(jax.devices()[0].device_kind),
            "hbm_per_chip_gb": round(
                (jax.local_devices()[0].memory_stats() or {}).get("bytes_limit", 0) / 1e9, 1
            ),
            "mesh": {"fsdp": mesh_counts[0], "tp": mesh_counts[1]},
            "jax_version": jax.__version__,
        },
        "config": {
            "model": MODEL_GCS_PATH,
            "dtype": "bfloat16",
            "max_new_tokens": args.max_tokens,
            "temperature": 0.7,
            "top_k": 50,
            "warmup_runs": args.warmup,
            "bench_runs": args.runs,
        },
        "model_load_time_s": round(t_load, 2),
        "benchmarks": [],
    }

    for prompt_name, prompt_text in BENCH_PROMPTS.items():
        log(f"\n--- Prompt: {prompt_name} ({len(prompt_text)} chars) ---")

        # Warmup
        for i in range(args.warmup):
            with mesh:
                out = sampler(
                    input_strings=[prompt_text],
                    total_generation_steps=args.max_tokens,
                    temperature=0.7, top_k=50,
                )
            jax.block_until_ready(out)
            log(f"  warmup {i+1}/{args.warmup}")

        # Timed runs
        latencies = []
        token_counts = []
        for i in range(args.runs):
            jax.block_until_ready(jax.devices())

            t0 = time.perf_counter()
            with mesh:
                out = sampler(
                    input_strings=[prompt_text],
                    total_generation_steps=args.max_tokens,
                    temperature=0.7, top_k=50,
                )
            jax.block_until_ready(out)
            elapsed = time.perf_counter() - t0

            out_text = out.text[0] if hasattr(out, 'text') else str(out)
            n_tokens = len(tokenizer.encode(out_text))
            latencies.append(elapsed)
            token_counts.append(n_tokens)
            log(f"  run {i+1}: {elapsed:.3f}s, {n_tokens} tokens, "
                f"{n_tokens/elapsed:.1f} tok/s")

        avg_lat = statistics.mean(latencies)
        avg_tok = statistics.mean(token_counts)
        results["benchmarks"].append({
            "prompt": prompt_name,
            "prompt_chars": len(prompt_text),
            "avg_latency_s": round(avg_lat, 4),
            "std_latency_s": round(statistics.stdev(latencies), 4) if len(latencies) > 1 else 0,
            "avg_output_tokens": round(avg_tok, 1),
            "throughput_tok_per_s": round(avg_tok / avg_lat, 2),
            "all_latencies_s": [round(l, 4) for l in latencies],
        })

    out_path = args.output or "results_tpu_tunix.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: vLLM server benchmark client (works for both TPU and GPU)
# ═══════════════════════════════════════════════════════════════════════════════

def run_vllm_benchmark(args):
    """
    Benchmark against a running vLLM OpenAI-compatible server.
    Works identically whether vLLM is running on TPU or GPU.

    Start vLLM on TPU:
      vllm serve google/gemma-4-31B-it --dtype bfloat16 --tensor-parallel-size 8

    Start vLLM on GPU:
      vllm serve google/gemma-4-31B-it --dtype bfloat16 --tensor-parallel-size 2
    """
    import asyncio
    import aiohttp

    base_url = args.url.rstrip("/")
    concurrency = args.concurrency
    max_tokens = args.max_tokens

    log("=" * 60)
    log(f"Phase 2: vLLM Server Benchmark — {base_url}")
    log(f"concurrency={concurrency}  max_tokens={max_tokens}  "
        f"warmup={args.warmup}  runs={args.runs}")
    log("=" * 60)

    async def single_request(session, prompt, max_tok):
        """Send one completion request, measure TTFT and total time."""
        payload = {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": max_tok,
            "temperature": 0.7,
            "stream": True,
        }
        ttft = None
        total_tokens = 0
        t0 = time.perf_counter()

        async with session.post(
            f"{base_url}/v1/completions",
            json=payload,
        ) as resp:
            async for line in resp.content:
                decoded = line.decode("utf-8").strip()
                if not decoded.startswith("data: "):
                    continue
                data_str = decoded[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    if chunk.get("choices") and chunk["choices"][0].get("text"):
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        total_tokens += 1
                except json.JSONDecodeError:
                    continue

        total_time = time.perf_counter() - t0
        decode_time = total_time - (ttft or 0)
        decode_throughput = (total_tokens - 1) / decode_time if decode_time > 0 and total_tokens > 1 else 0

        return {
            "ttft_s": ttft or total_time,
            "total_time_s": total_time,
            "output_tokens": total_tokens,
            "decode_tok_per_s": decode_throughput,
        }

    async def run_concurrent(prompt, max_tok, n_requests):
        """Run n_requests concurrently."""
        async with aiohttp.ClientSession() as session:
            tasks = [single_request(session, prompt, max_tok) for _ in range(n_requests)]
            return await asyncio.gather(*tasks)

    # Check server is up
    import urllib.request
    try:
        urllib.request.urlopen(f"{base_url}/v1/models", timeout=5)
        log("Server is reachable")
    except Exception as e:
        log(f"Cannot reach server at {base_url}: {e}")
        log("Start vLLM first:")
        log("  TPU: vllm serve google/gemma-4-31B-it --dtype bfloat16 --tensor-parallel-size 8")
        log("  GPU: vllm serve google/gemma-4-31B-it --dtype bfloat16 --tensor-parallel-size 2")
        sys.exit(1)

    results = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "engine": "vllm",
        "server_url": base_url,
        "config": {
            "model": args.model,
            "concurrency": concurrency,
            "max_new_tokens": max_tokens,
            "temperature": 0.7,
            "warmup_runs": args.warmup,
            "bench_runs": args.runs,
        },
        "benchmarks": [],
    }

    for prompt_name, prompt_text in BENCH_PROMPTS.items():
        log(f"\n--- Prompt: {prompt_name} (concurrency={concurrency}) ---")

        # Warmup
        for i in range(args.warmup):
            asyncio.run(run_concurrent(prompt_text, max_tokens, 1))
            log(f"  warmup {i+1}/{args.warmup}")

        # Timed runs — each run sends `concurrency` parallel requests
        all_results = []
        for i in range(args.runs):
            batch = asyncio.run(run_concurrent(prompt_text, max_tokens, concurrency))
            all_results.extend(batch)

            avg_ttft = statistics.mean(r["ttft_s"] for r in batch)
            avg_tps = statistics.mean(r["decode_tok_per_s"] for r in batch)
            log(f"  run {i+1}: avg_ttft={avg_ttft:.3f}s, avg_decode={avg_tps:.1f} tok/s")

        # Aggregate
        ttfts = [r["ttft_s"] for r in all_results]
        total_times = [r["total_time_s"] for r in all_results]
        decode_speeds = [r["decode_tok_per_s"] for r in all_results if r["decode_tok_per_s"] > 0]
        output_tokens = [r["output_tokens"] for r in all_results]

        # Total throughput = total tokens generated / wall clock of one run
        total_tok_per_run = sum(r["output_tokens"] for r in all_results) / args.runs
        wall_per_run = statistics.mean(
            max(r["total_time_s"] for r in all_results[i*concurrency:(i+1)*concurrency])
            for i in range(args.runs)
        )
        aggregate_throughput = total_tok_per_run / wall_per_run if wall_per_run > 0 else 0

        entry = {
            "prompt": prompt_name,
            "concurrency": concurrency,
            "num_requests": len(all_results),
            "ttft_mean_s": round(statistics.mean(ttfts), 4),
            "ttft_p50_s": round(sorted(ttfts)[len(ttfts)//2], 4),
            "ttft_p99_s": round(sorted(ttfts)[int(len(ttfts)*0.99)], 4),
            "decode_tok_per_s_mean": round(statistics.mean(decode_speeds), 2) if decode_speeds else 0,
            "total_time_mean_s": round(statistics.mean(total_times), 4),
            "avg_output_tokens": round(statistics.mean(output_tokens), 1),
            "aggregate_throughput_tok_per_s": round(aggregate_throughput, 2),
        }
        results["benchmarks"].append(entry)
        log(f"  → TTFT={entry['ttft_mean_s']:.3f}s, "
            f"decode={entry['decode_tok_per_s_mean']:.1f} tok/s, "
            f"aggregate={entry['aggregate_throughput_tok_per_s']:.1f} tok/s")

    out_path = args.output or f"results_vllm_c{concurrency}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Compare TPU vs GPU results
# ═══════════════════════════════════════════════════════════════════════════════

def run_compare(args):
    """Load two result JSONs and print a side-by-side comparison."""
    with open(args.tpu) as f:
        tpu = json.load(f)
    with open(args.gpu) as f:
        gpu = json.load(f)

    print("\n" + "=" * 72)
    print("  TPU vs GPU Inference Comparison")
    print("=" * 72)

    print(f"\n  TPU: {tpu.get('engine', '?')}  |  "
          f"GPU: {gpu.get('engine', '?')}")
    if "hardware" in tpu:
        h = tpu["hardware"]
        print(f"  TPU hw: {h.get('num_devices', '?')}x {h.get('device_kind', '?')}")
    if "hardware" in gpu:
        h = gpu["hardware"]
        print(f"  GPU hw: {h.get('num_devices', '?')}x {h.get('device_kind', '?')}")

    print(f"\n  {'Prompt':<12} {'Metric':<28} {'TPU':>12} {'GPU':>12} {'Ratio':>10}")
    print("  " + "-" * 74)

    tpu_bench = {b["prompt"]: b for b in tpu["benchmarks"]}
    gpu_bench = {b["prompt"]: b for b in gpu["benchmarks"]}

    for prompt_name in BENCH_PROMPTS:
        tb = tpu_bench.get(prompt_name, {})
        gb = gpu_bench.get(prompt_name, {})

        metrics = []
        # Adapt to whichever fields are present
        for key, label, unit, lower_better in [
            ("ttft_mean_s", "TTFT", "s", True),
            ("avg_latency_s", "Avg Latency", "s", True),
            ("total_time_mean_s", "Total Time", "s", True),
            ("throughput_tok_per_s", "Throughput", "tok/s", False),
            ("decode_tok_per_s_mean", "Decode Speed", "tok/s", False),
            ("aggregate_throughput_tok_per_s", "Aggregate Throughput", "tok/s", False),
        ]:
            tv = tb.get(key)
            gv = gb.get(key)
            if tv is not None or gv is not None:
                metrics.append((label, unit, tv, gv, lower_better))

        for i, (label, unit, tv, gv, lower_better) in enumerate(metrics):
            pname = prompt_name if i == 0 else ""
            tv_str = f"{tv:.3f} {unit}" if tv is not None else "n/a"
            gv_str = f"{gv:.3f} {unit}" if gv is not None else "n/a"

            if tv is not None and gv is not None and gv > 0:
                ratio = tv / gv
                if lower_better:
                    winner = "TPU" if ratio < 1 else "GPU"
                else:
                    winner = "TPU" if ratio > 1 else "GPU"
                ratio_str = f"{ratio:.2f}x ({winner})"
            else:
                ratio_str = "—"

            print(f"  {pname:<12} {label:<28} {tv_str:>12} {gv_str:>12} {ratio_str:>10}")

        if metrics:
            print()

    print("=" * 72)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Gemma 4 31B inference benchmark — TPU vs GPU"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # tunix subcommand
    p_tunix = sub.add_parser("tunix", help="Tunix sampler on TPU (Phase 1)")
    p_tunix.add_argument("--warmup", type=int, default=2)
    p_tunix.add_argument("--runs", type=int, default=5)
    p_tunix.add_argument("--max-tokens", type=int, default=256)
    p_tunix.add_argument("--output", type=str, default=None)

    # vllm subcommand
    p_vllm = sub.add_parser("vllm", help="Benchmark vLLM server (Phase 2)")
    p_vllm.add_argument("--url", type=str, default="http://localhost:8000")
    p_vllm.add_argument("--model", type=str, default="google/gemma-4-31B-it")
    p_vllm.add_argument("--concurrency", type=int, default=1)
    p_vllm.add_argument("--warmup", type=int, default=2)
    p_vllm.add_argument("--runs", type=int, default=5)
    p_vllm.add_argument("--max-tokens", type=int, default=256)
    p_vllm.add_argument("--output", type=str, default=None)

    # compare subcommand
    p_cmp = sub.add_parser("compare", help="Compare TPU vs GPU results")
    p_cmp.add_argument("--tpu", type=str, required=True, help="TPU results JSON")
    p_cmp.add_argument("--gpu", type=str, required=True, help="GPU results JSON")

    args = parser.parse_args()

    if args.mode == "tunix":
        run_tunix_benchmark(args)
    elif args.mode == "vllm":
        run_vllm_benchmark(args)
    elif args.mode == "compare":
        run_compare(args)


if __name__ == "__main__":
    main()
