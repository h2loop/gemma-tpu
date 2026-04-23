"""
Gemma 4 31B LoRA SFT on TPU.
Adapted from tunix qlora_gemma.ipynb (Gemma3 → Gemma4) +
sampling_example.ipynb for correct model-load kwargs.
Stack: JAX + Tunix (qwix LoRA). No PyTorch/torch_xla.
"""

import gc
import os
import threading
import time

# Redirect model cache to /dev/shm (tmpfs) to avoid filling local disk.
os.environ["TMPDIR"] = "/dev/shm"
# Persist XLA compilation cache so restarts skip the long first-step compile.
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"

import jax
import jax.numpy as jnp
import optax
import qwix
import gcsfs
from flax import nnx
from tunix.models.gemma4 import model as gemma4_model_lib
from tunix.models.gemma4 import params_safetensors as params_lib
from tunix.examples.data import translation_dataset as data_lib
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.sft import peft_trainer, metrics_logger, utils
import trajectory_dataset as traj_data_lib

# ── Monkey-patch tunix GCS loader to use local cache ──────────────────────────
import tunix.oss.utils as _tunix_oss_utils
_orig_load_file_from_gcs = _tunix_oss_utils.load_file_from_gcs

def _cached_load_file_from_gcs(file_dir, target_dir=None):
    import tempfile as _tmpmod
    if target_dir is None:
        target_dir = _tmpmod.gettempdir()
    _, prefix = file_dir[5:].split("/", 1)
    local_dir = os.path.join(target_dir, prefix)
    if os.path.isdir(local_dir) and any(f.endswith(".safetensors") for f in os.listdir(local_dir)):
        log(f"Using cached model at {local_dir}")
        return local_dir
    return _orig_load_file_from_gcs(file_dir, target_dir)

_tunix_oss_utils.load_file_from_gcs = _cached_load_file_from_gcs

# ── Logging helpers ───────────────────────────────────────────────────────────
_RUN_START = time.time()

def log(msg: str) -> None:
    elapsed = time.time() - _RUN_START
    print(f"[{elapsed:7.1f}s] {msg}", flush=True)

class _Phase:
    """Context manager: logs start/done with elapsed time; ticks every 20s."""
    def __init__(self, label: str, tick: int = 20):
        self.label = label
        self.tick  = tick
        self._stop = threading.Event()
        self._t0   = None

    def _ticker(self):
        while not self._stop.wait(self.tick):
            log(f"  {self.label} … {time.time() - self._t0:.0f}s")

    def __enter__(self):
        self._t0 = time.time()
        log(f"{self.label} — started")
        threading.Thread(target=self._ticker, daemon=True).start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        log(f"{self.label} — done ({time.time() - self._t0:.1f}s)")

def _device_memory_summary() -> None:
    for d in jax.local_devices():
        s = d.memory_stats() or {}
        used  = s.get("bytes_in_use", 0) / 1e9
        limit = s.get("bytes_limit",  0) / 1e9
        if limit:
            log(f"  {d}: {used:.1f} / {limit:.1f} GB used ({100*used/limit:.0f}%)")
        else:
            log(f"  {d}: memory_stats unavailable")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_GCS_PATH   = "gs://h2loop-gemma4/models/gemma-4-31b-it"
CKPT_DIR         = "gs://h2loop-gemma4/checkpoints/lora-run-001"
LOG_DIR          = "/tmp/tensorboard/lora"
# tokenizer_gemma3.model is the same SentencePiece vocab used by Gemma 4.
# If this 403s, see docs § "Tokenizer path may 403" for fallback.
TOKENIZER_PATH   = "gs://gemma-data/tokenizers/tokenizer_gemma3.model"

TRAJECTORY_DIR   = os.path.expanduser("~/t1_data")

USE_QUANTIZATION = False   # True → QLoRA (NF4 weights), False → LoRA
RANK             = 16
ALPHA            = float(2 * RANK)   # 16.0

MAX_SEQ_LEN      = 12288
BATCH_SIZE       = 4
MAX_STEPS        = 2000    # 4x more steps to compensate for bs 8→2
EVAL_EVERY       = 500     # evaluate at quarter points
CKPT_EVERY       = 250     # checkpoint every 250 steps
LR               = 2e-4

# ── Startup banner ────────────────────────────────────────────────────────────
import datetime
log("=" * 60)
log(f"Gemma 4 31B LoRA SFT — {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
log(f"JAX {jax.__version__}  |  model: {MODEL_GCS_PATH}")
log(f"rank={RANK}  alpha={ALPHA}  lr={LR}  steps={MAX_STEPS}  bs={BATCH_SIZE}  seq={MAX_SEQ_LEN}")
log(f"quantization: {'QLoRA (NF4)' if USE_QUANTIZATION else 'LoRA (bf16)'}")
log("=" * 60)

# ── Mesh ─────────────────────────────────────────────────────────────────────
NUM_TPUS = len(jax.devices())
log(f"JAX sees {NUM_TPUS} devices: {jax.devices()}")

if NUM_TPUS == 16:
    MESH_COUNTS = (2, 8)   # fsdp=2, tp=8
elif NUM_TPUS == 8:
    MESH_COUNTS = (2, 4)   # fsdp=2, tp=4 — uses all 8 chips
elif NUM_TPUS == 4:
    MESH_COUNTS = (1, 4)   # tp=4 — tensor parallel shards weights across all devices
elif NUM_TPUS == 1:
    MESH_COUNTS = (1, 1)
else:
    raise ValueError(f"No mesh preset for {NUM_TPUS} TPUs — add one above")

MESH = [MESH_COUNTS, ("fsdp", "tp")]
mesh = jax.make_mesh(*MESH, axis_types=(jax.sharding.AxisType.Auto,) * len(MESH[0]))
log(f"Mesh: fsdp={MESH_COUNTS[0]}, tp={MESH_COUNTS[1]}")

# ── Load model ────────────────────────────────────────────────────────────────
if MODEL_GCS_PATH.startswith("gs://"):
    _fs = gcsfs.GCSFileSystem()
    _bucket_path = MODEL_GCS_PATH.replace("gs://", "")
    _shards = [f for f in _fs.ls(_bucket_path) if f.endswith(".safetensors")]
    _total_bytes = sum(_fs.info(f)["size"] for f in _shards)
else:
    import glob as _glob
    _shards = _glob.glob(os.path.join(MODEL_GCS_PATH, "*.safetensors"))
    _total_bytes = sum(os.path.getsize(f) for f in _shards)
log(f"Model: {len(_shards)} safetensor shards, {_total_bytes/1e9:.1f} GB total")
log(f"Loading from {MODEL_GCS_PATH} ...")

model_config = gemma4_model_lib.ModelConfig.gemma4_31b()

with _Phase("Model download + load", tick=30):
    with mesh:
        # dtype=jnp.bfloat16 required — confirmed in sampling_example.ipynb
        base_model = params_lib.create_model_from_safe_tensors(
            MODEL_GCS_PATH, model_config, mesh, dtype=jnp.bfloat16
        )

log("Device memory after model load:")
_device_memory_summary()

# ── Apply LoRA (or QLoRA) ─────────────────────────────────────────────────────
# Module regex verified in QLoRA notebook — Gemma4 flax impl uses same einsum names.
# If LoraProvider raises on path regex, run: nnx.display(base_model) to inspect names.
if USE_QUANTIZATION:
    lora_provider = qwix.LoraProvider(
        module_path=".*q_einsum|.*kv_einsum|.*attn_vec_einsum|.*gate_proj|.*down_proj|.*up_proj",
        rank=RANK,
        alpha=ALPHA,
        weight_qtype="nf4",
        tile_size=128,
    )
else:
    lora_provider = qwix.LoraProvider(
        module_path=".*q_einsum|.*kv_einsum|.*attn_vec_einsum|.*gate_proj|.*down_proj|.*up_proj",
        rank=RANK,
        alpha=ALPHA,
    )

with _Phase("LoRA injection + shard"):
    # qwix needs a forward pass to trace the model and inject LoRA weights.
    # Use tiny dummy inputs (1 sample, 1 token) to minimise memory during tracing.
    _dummy_tokens = jnp.zeros((1, 1), dtype=jnp.int32)
    _dummy_positions = jnp.zeros((1, 1), dtype=jnp.int32)
    _dummy_mask = jnp.ones((1, 1, 1), dtype=jnp.bool_)
    lora_model = qwix.apply_lora_to_model(
        base_model, lora_provider,
        _dummy_tokens, _dummy_positions, None, _dummy_mask,
    )

    # The base model is already sharded from load. LoRA params are small and
    # don't need explicit re-sharding (partition specs from the original weights
    # are incompatible with LoRA's reshaped tensors).
    # Fix sharding annotations on LoRA params whose rank doesn't match their spec.
    # qwix copies the sharding tuple from the original weight (e.g. rank-4
    # kv_weight_cndh) onto rank-2 lora_a/lora_b params. The trainer's optimizer
    # sharding step then fails. Reset any mismatched annotations to replicated.
    _fixed = 0
    for _, module in nnx.iter_modules(lora_model):
        for attr_name in list(vars(module)):
            v = getattr(module, attr_name, None)
            if not isinstance(v, nnx.Variable):
                continue
            shd_val = getattr(v, 'sharding', None)
            if shd_val is None:
                continue
            if isinstance(shd_val, tuple):
                spec_len = len(shd_val)
            elif isinstance(shd_val, jax.sharding.NamedSharding):
                spec_len = len(shd_val.spec)
            elif isinstance(shd_val, jax.sharding.PartitionSpec):
                spec_len = len(shd_val)
            else:
                continue
            if spec_len != v[...].ndim:
                v.set_metadata('out_sharding', (None,) * v[...].ndim)
                _fixed += 1
            else:
                # Check divisibility against mesh for tuple-style sharding
                if isinstance(shd_val, tuple):
                    safe = list(shd_val)
                    changed = False
                    for i, axis_name in enumerate(shd_val):
                        if axis_name is not None and i < v[...].ndim:
                            mesh_size = MESH_COUNTS[["fsdp", "tp"].index(axis_name)] if axis_name in ("fsdp", "tp") else 1
                            if v[...].shape[i] % mesh_size != 0:
                                safe[i] = None
                                changed = True
                    if changed:
                        v.set_metadata('out_sharding', tuple(safe))
                        _fixed += 1
    if _fixed:
        log(f"Fixed sharding annotations on {_fixed} LoRA params")

del base_model
gc.collect()

trainable = sum(p.size for p in jax.tree.leaves(nnx.state(lora_model, nnx.LoRAParam)))
total     = sum(p.size for p in jax.tree.leaves(nnx.state(lora_model)))
method    = "QLoRA" if USE_QUANTIZATION else "LoRA"
log(f"{method} applied — trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# ── Apply gradient checkpointing (remat) to each decoder layer ────────────────
# Use nnx.remat on the unbound DecoderLayer.__call__ method.
from tunix.models.gemma4 import model as _gemma4_mod
_gemma4_mod.DecoderLayer.__call__ = nnx.remat(_gemma4_mod.DecoderLayer.__call__)
log("Applied nnx.remat to DecoderLayer.__call__")

# ── Tokenizer & dataset ───────────────────────────────────────────────────────
with _Phase("Tokenizer load"):
    tokenizer = tokenizer_lib.Tokenizer(tokenizer_path=TOKENIZER_PATH)

with _Phase("Dataset build"):
    train_ds, val_ds = traj_data_lib.create_trajectory_datasets(
        data_dir=TRAJECTORY_DIR,
        global_batch_size=BATCH_SIZE,
        max_target_length=MAX_SEQ_LEN,
        num_train_epochs=3,
        tokenizer=tokenizer,
    )

def gen_model_input_fn(x: peft_trainer.TrainingInput):
    pad_mask = x.input_tokens != tokenizer.pad_id()
    positions = utils.build_positions_from_mask(pad_mask)
    attention_mask = utils.make_causal_attn_mask(pad_mask)
    return {
        "input_tokens": x.input_tokens,
        "input_mask": x.input_mask,
        "positions": positions,
        "attention_mask": attention_mask,
    }

# ── Optimizer ────────────────────────────────────────────────────────────────
# Cosine decay with 10% warmup (500 steps → 50 warmup steps)
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=LR,
    warmup_steps=int(MAX_STEPS * 0.1),
    decay_steps=MAX_STEPS,
    end_value=LR * 0.1,
)
optimizer = optax.adamw(learning_rate=schedule, weight_decay=0.01)

# ── Train ─────────────────────────────────────────────────────────────────────
logging_options = metrics_logger.MetricsLoggerOptions(
    log_dir=LOG_DIR,
    flush_every_n_steps=10,
)

training_config = peft_trainer.TrainingConfig(
    eval_every_n_steps=0,
    max_steps=MAX_STEPS,
    metrics_logging_options=logging_options,
    checkpoint_root_directory=None,
)

trainer = peft_trainer.PeftTrainer(
    lora_model, optimizer, training_config
).with_gen_model_input_fn(gen_model_input_fn)

# Monkey-patch _shard_optimizer to handle LoRA rank mismatches.
# The optimizer state inherits partition specs from LoRA params whose sharding
# annotations don't match their actual rank (e.g. rank-4 spec on rank-2 param).
import jax.sharding as _shd
_orig_shard_optimizer = peft_trainer.PeftTrainer._shard_optimizer

def _patched_shard_optimizer(self, mesh):
    if mesh.empty:
        return
    optimizer_state = nnx.state(self.optimizer, nnx.optimizer.OptState)
    optimizer_pspecs = nnx.get_partition_spec(optimizer_state)

    def _fix(leaf, spec):
        if not isinstance(spec, _shd.PartitionSpec):
            return spec
        if len(spec) != leaf.ndim:
            return _shd.PartitionSpec(*((None,) * leaf.ndim))
        # Also check divisibility: if any axis can't be evenly split, replicate it.
        safe_axes = []
        for i, axis_name in enumerate(spec):
            if axis_name is not None and axis_name in mesh.shape:
                if leaf.shape[i] % mesh.shape[axis_name] != 0:
                    safe_axes.append(None)
                else:
                    safe_axes.append(axis_name)
            else:
                safe_axes.append(axis_name)
        return _shd.PartitionSpec(*safe_axes)

    optimizer_pspecs = jax.tree.map(_fix, optimizer_state, optimizer_pspecs)
    optimizer_sharded_state = jax.lax.with_sharding_constraint(
        optimizer_state, optimizer_pspecs
    )
    nnx.update(self.optimizer, optimizer_sharded_state)

peft_trainer.PeftTrainer._shard_optimizer = _patched_shard_optimizer

log(f"Starting {method} training — step 1 XLA compile takes 3–8 min before loss appears")
# Free fragmented memory before XLA compilation
gc.collect()
jax.clear_caches()
log("Device memory before training:")
_device_memory_summary()
with mesh:
    trainer.train(train_ds, val_ds)

# ── Dump comprehensive run metrics to JSON ────────────────────────────────────
import json as _json

_run_metrics = {
    "run_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    "wall_time_seconds": round(time.time() - _RUN_START, 1),
    "config": {
        "model": MODEL_GCS_PATH,
        "method": method,
        "use_quantization": USE_QUANTIZATION,
        "lora_rank": RANK,
        "lora_alpha": ALPHA,
        "learning_rate": LR,
        "max_steps": MAX_STEPS,
        "batch_size": BATCH_SIZE,
        "max_seq_len": MAX_SEQ_LEN,
        "eval_every_n_steps": EVAL_EVERY,
        "optimizer": "adamw",
        "weight_decay": 0.01,
        "warmup_steps": int(MAX_STEPS * 0.1),
        "checkpoint_dir": CKPT_DIR,
        "data_dir": TRAJECTORY_DIR,
    },
    "hardware": {
        "num_devices": NUM_TPUS,
        "device_kind": str(jax.devices()[0].device_kind),
        "hbm_per_chip_gb": round(
            (jax.local_devices()[0].memory_stats() or {}).get("bytes_limit", 0) / 1e9, 1
        ),
        "mesh": {"fsdp": MESH_COUNTS[0], "tp": MESH_COUNTS[1]},
        "jax_version": jax.__version__,
    },
    "model_info": {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_pct": round(100 * trainable / total, 4),
        "checkpoint_size_gb": round(_total_bytes / 1e9, 2),
    },
    "training": {},
    "eval": {},
}

# Extract metric histories from the trainer's metrics logger
_ml = trainer.metrics_logger
_prefix = trainer.metrics_prefix

for mode_name in ("train", "eval"):
    section = {}
    for metric_name in ("loss", "perplexity", "grad_norm"):
        try:
            if _ml.metric_exists(_prefix, metric_name, mode_name):
                history = _ml.get_metric_history(_prefix, metric_name, mode_name)
                section[metric_name] = [round(float(v), 6) for v in history]
                section[f"{metric_name}_mean"] = round(float(history.mean()), 6)
                section[f"{metric_name}_final"] = round(float(history[-1]), 6)
        except Exception:
            pass
    _run_metrics[mode_name if mode_name == "eval" else "training"] = section

# Device memory snapshot
_run_metrics["device_memory_final"] = []
for d in jax.local_devices():
    s = d.memory_stats() or {}
    _run_metrics["device_memory_final"].append({
        "device": str(d),
        "used_gb": round(s.get("bytes_in_use", 0) / 1e9, 2),
        "limit_gb": round(s.get("bytes_limit", 0) / 1e9, 2),
    })

_json_path = "training_run.json"
with open(_json_path, "w") as _jf:
    _json.dump(_run_metrics, _jf, indent=2)
log(f"Run metrics saved to {_json_path}")

log(f"Done. {method} adapters saved to: {CKPT_DIR}")
log(f"Total wall time: {time.time() - _RUN_START:.0f}s")
