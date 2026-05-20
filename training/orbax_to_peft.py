"""
Merge LoRA adapters from an orbax checkpoint into the Gemma4 base safetensors
weights and save a merged model that can be loaded as a plain base model.

The merged model has no qwix wrappers, so Tunix's Sampler (with KV cache,
proper duty cycle) works on it without trace-level issues.

Usage:
    python3 orbax_to_peft.py \
        --base-model gs://h2loop-gemma4/models/gemma-4-31b-it \
        --ckpt-dir   gs://h2loop-gemma4/checkpoints/lora-run-001 \
        --ckpt-step  1244 \
        --output-dir /dev/shm/merged_gemma4_31b \
        --rank 64 --alpha 64.0

The merged model is saved to --output-dir and can be loaded with:
    params_safetensors.create_model_from_safe_tensors(output_dir, ...)
"""

import argparse
import gc
import json
import os
import shutil

os.environ.setdefault("TMPDIR", "/dev/shm")
os.environ.setdefault("XLA_FLAGS", "--xla_llvm_disable_expensive_passes=true")

import numpy as np
import jax
import jax.numpy as jnp
import safetensors.numpy as safe_np
import qwix
from flax import nnx
from tunix.models.gemma4 import model as gemma4_model_lib
from tunix.models.gemma4 import params_safetensors as params_lib
from tunix.sft import checkpoint_manager as ckpt_mgr_lib

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

_LORA_MODULES = (
    ".*q_einsum|.*kv_einsum|.*attn_vec_einsum|.*gate_proj|.*down_proj|.*up_proj"
)


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


def _join(path) -> str:
    return ".".join(str(x) for x in path)


def _safetensors_key(tunix_path: str) -> str | list[tuple[str, slice]]:
    """Map a tunix LoRA path to Gemma4 safetensors key(s).

    tunix_path examples:
        layers.0.attn.q_einsum
        layers.0.attn.kv_einsum      → two keys (k_proj, v_proj)
        layers.0.attn.attn_vec_einsum
        layers.0.mlp.gate_proj
        layers.0.mlp.up_proj
        layers.0.mlp.down_proj
    """
    # layers.N.attn.X  or  layers.N.mlp.X
    parts = tunix_path.split(".")
    # parts: ['layers', N, 'attn'/'mlp', module_name]
    layer_idx = parts[1]
    module_type = parts[2]  # attn or mlp
    module_name = parts[3]  # q_einsum, kv_einsum, gate_proj, ...
    prefix = f"model.language_model.layers.{layer_idx}"

    if module_type == "attn":
        _map = {
            "q_einsum":        f"{prefix}.self_attn.q_proj.weight",
            "attn_vec_einsum": f"{prefix}.self_attn.o_proj.weight",
            # kv_einsum is handled separately (returns two keys)
        }
        if module_name == "kv_einsum":
            return [
                (f"{prefix}.self_attn.k_proj.weight", 0),  # index 0 = K
                (f"{prefix}.self_attn.v_proj.weight", 1),  # index 1 = V
            ]
        return _map[module_name]

    if module_type == "mlp":
        _map = {
            "gate_proj": f"{prefix}.mlp.gate_proj.weight",
            "up_proj":   f"{prefix}.mlp.up_proj.weight",
            "down_proj": f"{prefix}.mlp.down_proj.weight",
        }
        return _map[module_name]

    raise ValueError(f"Unknown tunix path: {tunix_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="gs://h2loop-gemma4/models/gemma-4-31b-it")
    ap.add_argument("--ckpt-dir",   default="gs://h2loop-gemma4/checkpoints/lora-run-001")
    ap.add_argument("--ckpt-step",  type=int, default=None)
    ap.add_argument("--output-dir", default="/dev/shm/merged_gemma4_31b")
    ap.add_argument("--rank",  type=int,   default=64)
    ap.add_argument("--alpha", type=float, default=64.0)
    args = ap.parse_args()

    mesh = build_mesh()
    model_config = gemma4_model_lib.ModelConfig.gemma4_31b()

    # ── Load base model + inject LoRA + restore checkpoint ───────────────────
    print(f"[merge] Loading base model from {args.base_model}")
    with mesh:
        base_model = params_lib.create_model_from_safe_tensors(
            args.base_model, model_config, mesh, dtype=jnp.bfloat16
        )

    print(f"[merge] Injecting LoRA r={args.rank} α={args.alpha}")
    lora_provider = qwix.LoraProvider(
        module_path=_LORA_MODULES, rank=args.rank, alpha=args.alpha,
    )
    dummy = jnp.zeros((1, 1), dtype=jnp.int32)
    dummy_m = jnp.ones((1, 1, 1), dtype=jnp.bool_)
    lora_model = qwix.apply_lora_to_model(
        base_model, lora_provider, dummy, dummy, None, dummy_m,
    )
    del base_model; gc.collect()

    print(f"[merge] Restoring checkpoint from {args.ckpt_dir} step={args.ckpt_step or 'latest'}")
    with mesh:
        cm = ckpt_mgr_lib.CheckpointManager(root_directory=args.ckpt_dir)
        step, _ = cm.maybe_restore(
            lora_model, optimizer=None, step=args.ckpt_step,
            restore_only_lora_params=True,
        )
    if step == 0:
        raise RuntimeError(f"No checkpoint found in {args.ckpt_dir}")
    print(f"[merge] Restored step {step}")

    # ── Collect LoRA params ──────────────────────────────────────────────────
    # nnx.iter_graph yields (path_tuple, variable). For LoRAParam:
    # path looks like: (..., 'layers', 0, 'attn', 'q_einsum', 'lora_a')
    lora_pairs: dict[str, list] = {}
    for path, value in nnx.iter_graph(lora_model):
        if not isinstance(value, nnx.LoRAParam):
            continue
        param_name = str(path[-1])   # 'w_lora_a' or 'w_lora_b'
        layer_path = _join(path[:-1])  # e.g. 'layers.0.attn.q_einsum'
        if layer_path not in lora_pairs:
            lora_pairs[layer_path] = {}
        # Normalise key to 'lora_a' / 'lora_b' regardless of prefix
        key = "lora_a" if "lora_a" in param_name else "lora_b"
        lora_pairs[layer_path][key] = np.array(value[...])

    print(f"[merge] Found {len(lora_pairs)} LoRA modules")

    # ── Load base safetensors into a mutable dict ────────────────────────────
    # Model may be sharded across multiple shard files — load index.
    base_local = _cached_load(args.base_model) if args.base_model.startswith("gs://") \
        else args.base_model

    index_path = os.path.join(base_local, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        # Load all unique shard files
        shard_files = sorted(set(weight_map.values()))
        base_state = {}
        for sf in shard_files:
            print(f"[merge] Loading shard {sf}")
            base_state.update(safe_np.load_file(os.path.join(base_local, sf)))
    else:
        base_state = dict(safe_np.load_file(os.path.join(base_local, "model.safetensors")))

    print(f"[merge] Loaded {len(base_state)} base weights")

    # ── Apply LoRA deltas W_merged = W + (lora_a @ lora_b) * (alpha/rank) ────
    # Shapes confirmed from Gemma4 31b:
    #   q_einsum:        lora_a (in, r), lora_b (r, n_heads, head_dim)
    #                    → delta (in, n_heads*head_dim) T → (out, in) = q_proj
    #   kv_einsum:       lora_a (in, r), lora_b (r, 2, n_kv_heads, head_dim)
    #                    → split on dim-1 for K and V → each (n_kv_heads*head_dim, in)
    #   attn_vec_einsum: lora_a (n_heads, head_dim, r), lora_b (r, in)
    #                    → delta (n_heads*head_dim, in) T → (in, out) = o_proj
    #   gate/up_proj:    lora_a (in, r), lora_b (r, out)  → delta (in,out) T → (out,in)
    #   down_proj:       lora_a (in, r), lora_b (r, out)  → delta (in,out) T → (out,in)
    merged = 0
    scale = args.alpha / args.rank

    for layer_path, params in lora_pairs.items():
        lora_a = params["lora_a"].astype(np.float32)
        lora_b = params["lora_b"].astype(np.float32)
        module_name = layer_path.split(".")[-1]  # q_einsum, kv_einsum, etc.
        keys = _safetensors_key(layer_path)

        if module_name == "kv_einsum":
            # lora_a: (in, r),  lora_b: (r, 2, n_kv_heads, head_dim)
            # K uses lora_b[:,0,:,:], V uses lora_b[:,1,:,:]
            for safetensors_key, kv_idx in keys:
                assert safetensors_key in base_state, f"Key not found: {safetensors_key}"
                b_slice = lora_b[:, kv_idx, :, :].reshape(lora_b.shape[0], -1)  # (r, kv_heads*head_dim)
                delta = (lora_a @ b_slice) * scale          # (in, kv_heads*head_dim)
                delta = delta.T                              # (kv_heads*head_dim, in) = (out, in)
                w = base_state[safetensors_key].astype(np.float32)
                assert delta.shape == w.shape, f"{safetensors_key}: delta {delta.shape} vs w {w.shape}"
                base_state[safetensors_key] = (w + delta).astype(base_state[safetensors_key].dtype)
                merged += 1

        elif module_name == "attn_vec_einsum":
            # lora_a: (n_heads, head_dim, r) → flatten → (n_heads*head_dim, r)
            # lora_b: (r, in)
            # safetensors o_proj: (in, n_heads*head_dim) — TRANSPOSED layout in HF
            a2 = lora_a.reshape(-1, lora_a.shape[-1])       # (n_heads*head_dim, r)
            delta = (a2 @ lora_b) * scale                   # (n_heads*head_dim, in)
            delta = delta.T                                  # (in, n_heads*head_dim) = o_proj shape
            assert keys in base_state, f"Key not found: {keys}"
            w = base_state[keys].astype(np.float32)
            assert delta.shape == w.shape, f"{keys}: delta {delta.shape} vs w {w.shape}"
            base_state[keys] = (w + delta).astype(base_state[keys].dtype)
            merged += 1

        else:
            # q_einsum / gate_proj / up_proj / down_proj
            # lora_a: (in, r),  lora_b: (r, ...)  → flatten lora_b to (r, out)
            b2 = lora_b.reshape(lora_b.shape[0], -1)        # (r, out)
            delta = (lora_a @ b2) * scale                   # (in, out)
            delta = delta.T                                  # (out, in) = HF weight layout
            assert keys in base_state, f"Key not found: {keys}"
            w = base_state[keys].astype(np.float32)
            assert delta.shape == w.shape, f"{keys}: delta {delta.shape} vs w {w.shape}"
            base_state[keys] = (w + delta).astype(base_state[keys].dtype)
            merged += 1

    print(f"[merge] Applied {merged} LoRA deltas")

    # ── Save merged model ─────────────────────────────────────────────────────
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    # Save all weights into a single file (easier to load with Tunix)
    merged_path = os.path.join(args.output_dir, "model.safetensors")
    print(f"[merge] Saving merged weights to {merged_path}")
    safe_np.save_file(base_state, merged_path)

    # Copy config, tokenizer, etc.
    for fname in os.listdir(base_local):
        if not fname.endswith(".safetensors") and not fname.endswith(".index.json"):
            src = os.path.join(base_local, fname)
            if os.path.isfile(src):
                shutil.copy(src, os.path.join(args.output_dir, fname))

    size_gb = os.path.getsize(merged_path) / 1e9
    print(f"[merge] Done. Merged model at {args.output_dir} ({size_gb:.1f} GB)")
    print(f"[merge] Load with: params_safetensors.create_model_from_safe_tensors('{args.output_dir}', ...)")


if __name__ == "__main__":
    main()
