#!/usr/bin/env python
"""
TPU evaluator for Gemma 4 31B + LoRA on NVlabs/verilog-eval.

Inference path (correct, per tunix/models/gemma4/sampling_example.ipynb):
  1. Merge LoRA adapters into base weights via orbax_to_peft.py
     → saves a merged model.safetensors to /dev/shm/merged_gemma4_31b/
  2. Load the merged model as a plain base model (no qwix, no LoRA wrappers)
  3. Run tunix Sampler (with KV cache, proper duty cycle)

Requires `iverilog` on PATH.
"""

import argparse
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("TMPDIR", "/dev/shm")
os.environ.setdefault("XLA_FLAGS", "--xla_llvm_disable_expensive_passes=true")

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import numpy as np
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer
from tunix.generate import sampler as sampler_lib
from tunix.models.gemma4 import model as gemma4_model_lib
from tunix.models.gemma4 import params_safetensors as params_lib

import tunix.oss.utils as _tunix_oss_utils
import tunix.models.safetensors_loader as _tunix_sl
_orig_load = _tunix_oss_utils.load_file_from_gcs


def _cached_load(file_dir, target_dir=None):
    import tempfile as _t
    if target_dir is None:
        target_dir = _t.gettempdir()
    _, prefix = file_dir[5:].split("/", 1)
    local_dir = os.path.join(target_dir, prefix)
    if os.path.isdir(local_dir) and any(
        f.endswith(".safetensors") for f in os.listdir(local_dir)
    ):
        return local_dir
    return _orig_load(file_dir, target_dir)


_tunix_oss_utils.load_file_from_gcs = _cached_load
_tunix_sl.load_file_from_gcs = _cached_load


SYSTEM_PROMPT = (
    "You are an expert Verilog hardware design engineer. "
    "Given a natural-language specification, write correct, synthesizable "
    "Verilog code that implements the requested module. Output only the "
    "Verilog code inside a ```verilog ... ``` fenced block."
)
VERILOG_FENCE_RE = re.compile(
    r"```(?:verilog|systemverilog|sv)?\s*(.*?)```", re.DOTALL
)


def discover_problems(dataset_dir: Path):
    prompts = sorted(dataset_dir.glob("Prob*_prompt.txt"))
    problems = []
    for p in prompts:
        pid = p.name[: -len("_prompt.txt")]
        if (dataset_dir / f"{pid}_test.sv").exists() and \
           (dataset_dir / f"{pid}_ref.sv").exists():
            problems.append(pid)
    return problems


def load_problem(dataset_dir: Path, pid: str):
    return {
        "id": pid,
        "prompt": (dataset_dir / f"{pid}_prompt.txt").read_text(),
        "test": (dataset_dir / f"{pid}_test.sv").read_text(),
        "ref": (dataset_dir / f"{pid}_ref.sv").read_text(),
    }


def extract_verilog(text: str) -> str:
    # Try to find a complete fenced block first
    m = VERILOG_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    # If the model hit max_new_tokens mid-block, the closing ``` may be missing.
    # Strip the opening fence and return whatever code follows.
    t = text.strip()
    for prefix in ("```verilog", "```systemverilog", "```sv", "```"):
        if t.startswith(prefix):
            t = t[len(prefix):].lstrip("\n")
            # Remove trailing incomplete fence if present
            if t.rstrip().endswith("```"):
                t = t.rstrip()[:-3].rstrip()
            return t.strip()
    return t


def build_prompt(tokenizer, problem) -> str:
    """Prompt matching training distribution (no system role) + targeted hints:
    - Declare outputs driven by always blocks as reg (wire/reg confusion)
    - Use exact port names from the spec (hallucinated names cause testbench mismatch)
    - Use reg for variables assigned inside always blocks (not wire)
    - Assign array elements individually, not the whole array at once
    """
    user_content = (
        f"{problem['prompt'].strip()}\n\n"
        f"Name the module `TopModule`. "
        f"Use the exact port names specified above. "
        f"Declare any signal assigned inside an always block as `reg`, not `wire`. "
        f"Assign array elements individually (e.g. arr[i] <= val), not the whole array. "
        f"Output only the Verilog code inside a ```verilog ... ``` fenced block."
    )
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ── Simulation (identical to upstream evaluate.py) ───────────────────────────
def run_iverilog(work_dir: Path, files, timeout: int = 60):
    out_bin = work_dir / "a.out"
    try:
        c = subprocess.run(
            ["iverilog", "-g2012", "-o", str(out_bin), *map(str, files)],
            capture_output=True, text=True, timeout=timeout, cwd=work_dir,
        )
    except subprocess.TimeoutExpired:
        return False, "compile_timeout"
    if c.returncode != 0:
        return False, f"compile_error:\n{c.stderr}"
    try:
        r = subprocess.run(
            ["vvp", str(out_bin)],
            capture_output=True, text=True, timeout=timeout, cwd=work_dir,
        )
    except subprocess.TimeoutExpired:
        return False, "run_timeout"
    log = r.stdout + r.stderr
    m = re.search(r"Mismatches:\s*(\d+)\s+in\s+(\d+)\s+samples", log)
    if m:
        return int(m.group(1)) == 0 and int(m.group(2)) > 0, log
    if r.returncode != 0:
        return False, f"run_nonzero:\n{log}"
    if re.search(r"\b(ERROR|FAIL|Mismatch)\b", log, re.IGNORECASE):
        return False, log
    return True, log


def evaluate_sample(args_tuple):
    pid, sample_idx, generated_code, ref_text, test_text, work_root = args_tuple
    sample_dir = (Path(work_root) / pid / f"s{sample_idx}").resolve()
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "TopModule.sv").write_text(generated_code)
    (sample_dir / "RefModule.sv").write_text(ref_text)
    (sample_dir / "test.sv").write_text(test_text)
    passed, log = run_iverilog(sample_dir, [
        sample_dir / "RefModule.sv",
        sample_dir / "TopModule.sv",
        sample_dir / "test.sv",
    ])
    (sample_dir / "result.log").write_text(log)
    return pid, sample_idx, passed


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def build_mesh():
    n = len(jax.devices())
    if n == 16:   counts = (2, 8)
    elif n == 8:  counts = (2, 4)
    elif n == 4:  counts = (1, 4)
    else:         counts = (1, 1)
    return jax.make_mesh(
        counts, ("fsdp", "tp"),
        axis_types=(jax.sharding.AxisType.Auto,) * 2,
    )


def merge_and_load(base_model_path, ckpt_dir, ckpt_step, rank, alpha,
                   merged_dir, force_remerge):
    """Run orbax_to_peft.py to merge LoRA into base, then load merged model."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "orbax_to_peft.py")
    if not os.path.exists(script):
        script = os.path.expanduser("~/orbax_to_peft.py")

    merged_safetensors = os.path.join(merged_dir, "model.safetensors")
    if force_remerge or not os.path.exists(merged_safetensors):
        print(f"[eval] Merging LoRA adapters → {merged_dir}")
        cmd = [
            sys.executable, script,
            "--base-model", base_model_path,
            "--ckpt-dir", ckpt_dir,
            "--output-dir", merged_dir,
            "--rank", str(rank),
            "--alpha", str(alpha),
        ]
        if ckpt_step is not None:
            cmd += ["--ckpt-step", str(ckpt_step)]
        ret = subprocess.run(cmd, check=True)
        if ret.returncode != 0:
            raise RuntimeError("orbax_to_peft.py failed")
    else:
        print(f"[eval] Using cached merged model at {merged_dir}")

    # Load merged model as plain base model — no qwix, no LoRA wrappers
    mesh = build_mesh()
    model_config = gemma4_model_lib.ModelConfig.gemma4_31b()
    print(f"[eval] Loading merged model from {merged_dir}")
    with mesh:
        merged_model = params_lib.create_model_from_safe_tensors(
            merged_dir, model_config, mesh, dtype=jnp.bfloat16
        )

    # Cache config per sampling_example.ipynb for gemma4_31b:
    #   num_kv_heads=16, head_dim=512 (not 256 — that's per-head, not global)
    cache_config = sampler_lib.CacheConfig(
        cache_size=4096,   # prompt_budget + gen_budget; must exceed longest prompt+gen
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        head_dim=model_config.head_dim,
    )
    return merged_model, cache_config, mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="gs://YOUR_BUCKET/models/gemma-4-31b-it")
    ap.add_argument("--ckpt-dir",   default="gs://YOUR_BUCKET/checkpoints/lora-run-001")
    ap.add_argument("--ckpt-step",  type=int, default=None)
    ap.add_argument("--lora-rank",  type=int,   default=64)
    ap.add_argument("--lora-alpha", type=float, default=64.0)
    ap.add_argument("--merged-dir", default="/dev/shm/merged_gemma4_31b",
                    help="Where to save/cache the merged model weights.")
    ap.add_argument("--force-remerge", action="store_true",
                    help="Re-run the merge even if merged-dir already exists.")
    ap.add_argument("--rerun-failed", action="store_true",
                    help="Keep passing results from progress.json, re-run only failed problems.")
    ap.add_argument("--from-gcs-samples", action="store_true",
                    help="Download sample JSONs from GCS and re-simulate (skip generation). "
                         "Reconstructs ground-truth scores from stored outputs.")

    ap.add_argument("--verilog-eval-dir", required=True)
    ap.add_argument("--task", choices=["spec-to-rtl", "code-complete-iccad2023"],
                    default="spec-to-rtl")
    ap.add_argument("--problems", nargs="*", default=None)
    ap.add_argument("--n-samples",    type=int,   default=5)
    ap.add_argument("--temperature",  type=float, default=0.8)
    ap.add_argument("--top-p",        type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--output-dir",   default="eval_out")
    ap.add_argument("--ks",           nargs="+", type=int, default=[1, 5])
    ap.add_argument("--sim-workers",  type=int, default=8)
    ap.add_argument("--limit",        type=int, default=None)
    ap.add_argument("--seed",         type=int, default=0)
    args = ap.parse_args()

    if shutil.which("iverilog") is None:
        sys.exit("error: iverilog not on PATH")

    dataset_dir = Path(args.verilog_eval_dir) / f"dataset_{args.task}"
    if not dataset_dir.exists():
        sys.exit(f"error: {dataset_dir} not found")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples"; samples_dir.mkdir(exist_ok=True)
    work_root = out_dir / "sim"; work_root.mkdir(exist_ok=True)

    pids = args.problems or discover_problems(dataset_dir)
    if args.limit:
        pids = pids[: args.limit]
    print(f"[eval] {len(pids)} problems, {args.n_samples} samples, "
          f"T={args.temperature} top_p={args.top_p}")

    # ── Merge + load (skipped when --from-gcs-samples) ───────────────────────
    gen_sampler = None
    tokenizer = None
    mesh = None

    if not args.from_gcs_samples:
        t0 = time.time()
        merged_model, cache_config, mesh = merge_and_load(
            args.base_model, args.ckpt_dir, args.ckpt_step,
            args.lora_rank, args.lora_alpha,
            args.merged_dir, args.force_remerge,
        )
        local_model = _cached_load(args.base_model) if args.base_model.startswith("gs://") \
            else args.base_model
        import json as _json
        _tok_cfg_path = os.path.join(local_model, "tokenizer_config.json")
        if os.path.exists(_tok_cfg_path):
            _tok_cfg = _json.load(open(_tok_cfg_path))
            if isinstance(_tok_cfg.get("extra_special_tokens"), list):
                _tok_cfg["extra_special_tokens"] = {}
                with open(_tok_cfg_path, "w") as _f:
                    _json.dump(_tok_cfg, _f)
        tokenizer = AutoTokenizer.from_pretrained(local_model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        gen_sampler = sampler_lib.Sampler(
            transformer=merged_model,
            tokenizer=tokenizer,
            cache_config=cache_config,
        )
        print(f"[eval] Model + sampler ready in {time.time() - t0:.1f}s")

    # GCS path for eval artifacts
    eval_gcs_dir = args.ckpt_dir.rstrip("/") + f"/eval/{Path(args.output_dir).name}"

    n = args.n_samples
    per_problem = {}
    running_pass1 = []

    # ── Load prior results ────────────────────────────────────────────────────
    # --from-gcs-samples: download sample JSONs from GCS, skip generation,
    #   re-simulate everything (reconstructs ground-truth from stored outputs).
    # --rerun-failed: keep passing results from progress.json, re-generate
    #   and re-simulate only the failed problems.
    # default resume: skip all problems already in progress.json.

    if args.from_gcs_samples:
        print(f"[eval] Downloading sample JSONs from {eval_gcs_dir}/samples/ …")
        import gcsfs as _gcsfs
        fs = _gcsfs.GCSFileSystem()
        remote_samples = f"{eval_gcs_dir}/samples"
        for pid in pids:
            remote = f"{remote_samples}/{pid}.json"
            local = samples_dir / f"{pid}.json"
            if not local.exists():
                try:
                    fs.get(remote, str(local))
                except Exception as e:
                    print(f"  [GCS] {pid}: {e}")
        print(f"[eval] Downloaded samples for re-simulation")

    progress_path = out_dir / "progress.json"
    if progress_path.exists() and not args.from_gcs_samples:
        prev = json.load(open(progress_path))
        prev_results = prev.get("per_problem", {})
        if args.rerun_failed:
            per_problem = {pid: res for pid, res in prev_results.items()
                          if res.get("passed", 0) > 0}
            skipped_failed = len(prev_results) - len(per_problem)
            print(f"[eval] Rerun-failed mode: keeping {len(per_problem)} passed, "
                  f"re-running {skipped_failed} failed problems")
        else:
            per_problem = prev_results
        for pid, res in per_problem.items():
            running_pass1.append(res.get("pass@1", float(res.get("passed", 0) > 0)))
        if per_problem:
            print(f"[eval] Loaded {len(per_problem)} prior results, "
                  f"running_pass@1={np.mean(running_pass1):.3f}")

    # Backup existing GCS progress.json before this run overwrites it
    import datetime as _dt
    _run_ts = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    try:
        import gcsfs as _gcsfs
        _fs_bak = _gcsfs.GCSFileSystem()
        for _fname in ("progress.json", "summary.json"):
            _src = f"{eval_gcs_dir}/{_fname}"
            _dst = f"{eval_gcs_dir}/history/{_run_ts}_{_fname}"
            try:
                _fs_bak.copy(_src, _dst)
                print(f"[GCS] backed up {_fname} → history/{_run_ts}_{_fname}")
            except Exception:
                pass  # file may not exist yet
    except Exception:
        pass

    def _flush_progress():
        done = len(per_problem)
        progress = {
            "problems_done": done,
            "problems_total": len(pids),
            "running_pass@1": round(float(np.mean(running_pass1)), 4) if running_pass1 else None,
            "per_problem": per_problem,
        }
        local_path = out_dir / "progress.json"
        local_path.write_text(json.dumps(progress, indent=2))
        try:
            import gcsfs as _gcsfs
            _gcsfs.GCSFileSystem().put(str(local_path), f"{eval_gcs_dir}/progress.json")
        except Exception as e:
            print(f"[GCS] upload failed: {e}")

    # ── Generate + simulate per-problem ──────────────────────────────────────
    for i, pid in enumerate(pids):
        if pid in per_problem:
            continue   # already done in a prior run

        problem = load_problem(dataset_dir, pid)
        sample_file = samples_dir / f"{pid}.json"

        t_gen = time.time()
        if args.from_gcs_samples and sample_file.exists():
            # Use previously generated outputs — skip generation entirely
            saved = json.load(open(sample_file))
            raws = saved.get("raw", saved.get("codes", []))
        else:
            prompt_text = build_prompt(tokenizer, problem)
            with mesh:
                out = gen_sampler(
                    input_strings=[prompt_text] * n,
                    max_generation_steps=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=args.seed + i,
                    echo=False,
                    eos_tokens=[tokenizer.eos_token_id, 106, 50],
                )
            raws = out.text
        t_gen = time.time() - t_gen
        codes = [extract_verilog(r) for r in raws]
        gen_time = time.time() - t_gen

        sample_file = samples_dir / f"{pid}.json"
        sample_file.write_text(
            json.dumps({"pid": pid, "raw": raws, "codes": codes}, indent=2)
        )
        try:
            import gcsfs as _gcsfs
            _gcsfs.GCSFileSystem().put(str(sample_file),
                                       f"{eval_gcs_dir}/samples/{pid}.json")
        except Exception:
            pass

        # Simulate
        tasks = [
            (pid, j, code, problem["ref"], problem["test"], str(work_root))
            for j, code in enumerate(codes)
        ]
        sim_results = [False] * n
        with ProcessPoolExecutor(max_workers=args.sim_workers) as ex:
            for fut in as_completed([ex.submit(evaluate_sample, t) for t in tasks]):
                _, j, passed = fut.result()
                sim_results[j] = passed

        c = sum(sim_results)
        per_problem[pid] = {
            "passed": c, "total": n,
            **{f"pass@{k}": pass_at_k(n, c, k) for k in args.ks if k <= n},
        }
        running_pass1.append(per_problem[pid].get("pass@1", float(c > 0)))
        _flush_progress()

        print(f"[{i+1}/{len(pids)}] {pid}  "
              f"passed={c}/{n}  pass@1={per_problem[pid].get('pass@1',0):.3f}  "
              f"running_pass@1={np.mean(running_pass1):.3f}  "
              f"gen={gen_time:.0f}s")

    if gen_sampler is not None:
        del gen_sampler
    if 'merged_model' in dir() and merged_model is not None:
        del merged_model
    gc.collect()
    if not args.from_gcs_samples:
        jax.clear_caches()

    # ── Final aggregate ───────────────────────────────────────────────────────
    aggregate = {
        f"pass@{k}": float(np.mean([
            per_problem[p][f"pass@{k}"] for p in pids
            if f"pass@{k}" in per_problem[p]
        ]))
        for k in args.ks if k <= n
    }

    summary = {
        "base_model": args.base_model,
        "ckpt_dir": args.ckpt_dir,
        "ckpt_step": args.ckpt_step,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "task": args.task,
        "n_samples": n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_problems": len(pids),
        "aggregate": aggregate,
        "per_problem": per_problem,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    try:
        import gcsfs as _gcsfs
        _gcsfs.GCSFileSystem().put(str(summary_path), f"{eval_gcs_dir}/summary.json")
        print(f"[GCS] summary → {eval_gcs_dir}/summary.json")
    except Exception as e:
        print(f"[GCS] summary upload failed: {e}")

    print("\n=== Final Results ===")
    for k, v in aggregate.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
