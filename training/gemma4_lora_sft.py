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
# Skip expensive LLVM optimization passes — trades small runtime perf for much faster compile.
os.environ["XLA_FLAGS"] = "--xla_llvm_disable_expensive_passes=true"

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
from tunix.sft import peft_trainer, metrics_logger, utils, hooks as sft_hooks
import codev_dataset as codev_data_lib

# ── Monkey-patch tunix GCS loader to use local cache ──────────────────────────
import tunix.oss.utils as _tunix_oss_utils
import tunix.models.safetensors_loader as _tunix_sl
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

# Patch both the utils module and safetensors_loader's local copy (set at import time)
_tunix_oss_utils.load_file_from_gcs = _cached_load_file_from_gcs
_tunix_sl.load_file_from_gcs = _cached_load_file_from_gcs

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

# ── Recipe mirrors h2loop/gemma4_sft_h100/train_2.py (commit 59f2e67) ────────
USE_QUANTIZATION       = False   # True → QLoRA (NF4 weights), False → LoRA
RANK                   = 64
ALPHA                  = 64.0    # α = r
LORA_DROPOUT           = 0.05    # train_2.py@59f2e67: 0.0 → 0.05

MAX_SEQ_LEN            = 3072   # CodeV-R1 10k: mean=922, p99=2553, p99.5=2894; 3072 covers 99.6%
# Effective batch = 8 (train_2.py@59f2e67: per_device=1, grad_accum=8, 2 GPUs ⇒ 16 scaled to 8 per-replica).
# Upstream new recipe: effective 8 on 2×H100 (1·8·1 if interpreting per-device × grad_accum × world, but
# actual effective batch depends on world size; we match the "micro·accum" product = 8).
MICRO_BATCH_SIZE       = 1       # per optimizer substep
GRAD_ACCUM_STEPS       = 8
EFFECTIVE_BATCH        = MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS

NUM_TRAIN_EPOCHS       = 1       # train_2.py: num_train_epochs=1
LR                     = 1e-4    # train_2.py@59f2e67: 2e-4 → 1e-4
WEIGHT_DECAY           = 0.001
WARMUP_STEPS           = 100
LR_END_VALUE           = 0.0

# Subsample: train_2.py@59f2e67 does raw.select(range(10000)). Match it.
DATASET_LIMIT          = 10000   # None for full 87,321 rows

# Checkpointing + eval cadence. train_2.py@59f2e67 saves every 100 steps (keeps 3).
EVAL_EVERY             = 0
CKPT_EVERY             = 100
CKPT_KEEP              = 3
LOG_EVERY              = 10      # train_2.py@59f2e67: logging_steps=10

# ── Startup banner ────────────────────────────────────────────────────────────
import datetime
log("=" * 60)
log(f"Gemma 4 31B LoRA SFT — {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
log(f"JAX {jax.__version__}  |  model: {MODEL_GCS_PATH}")
log(f"rank={RANK}  alpha={ALPHA}  dropout={LORA_DROPOUT}  lr={LR}  "
    f"epochs={NUM_TRAIN_EPOCHS}  micro_bs={MICRO_BATCH_SIZE}  "
    f"grad_accum={GRAD_ACCUM_STEPS}  eff_bs={EFFECTIVE_BATCH}  seq={MAX_SEQ_LEN}")
log(f"quantization: {'QLoRA (NF4)' if USE_QUANTIZATION else 'LoRA (bf16)'}")
log("=" * 60)

# ── Mesh ─────────────────────────────────────────────────────────────────────
NUM_TPUS = len(jax.devices())
log(f"JAX sees {NUM_TPUS} devices: {jax.devices()}")

if NUM_TPUS == 16:
    MESH_COUNTS = (2, 8)   # fsdp=2, tp=8
elif NUM_TPUS == 8:
    MESH_COUNTS = (2, 4)   # fsdp=2, tp=4 — tp=8 breaks on Gemma4 heads not divisible by 8
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
# Target modules mirror train_2.py {q,k,v,o,gate,up,down}_proj. In Tunix's Gemma4
# flax port: q_einsum = q_proj, kv_einsum = k_proj+v_proj (fused), and
# attn_vec_einsum = o_proj. The model has no vision tower, so no exclusion regex
# is required (train_2.py needs `^(?!.*vision)` to filter the HF vision stack).
_LORA_MODULES = (
    ".*q_einsum|.*kv_einsum|.*attn_vec_einsum|.*gate_proj|.*down_proj|.*up_proj"
)
if USE_QUANTIZATION:
    lora_provider = qwix.LoraProvider(
        module_path=_LORA_MODULES,
        rank=RANK,
        alpha=ALPHA,
        dropout=LORA_DROPOUT,
        weight_qtype="nf4",
        tile_size=128,
    )
else:
    lora_provider = qwix.LoraProvider(
        module_path=_LORA_MODULES,
        rank=RANK,
        alpha=ALPHA,
        dropout=LORA_DROPOUT,
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

# Register an RNG stream for qwix's dropout-in-LoRA path. Required when
# LoraProvider(dropout > 0); qwix reads it via `module.qwix_rngs[...]()`.
if LORA_DROPOUT > 0:
    lora_model.set_attributes(qwix_rngs=nnx.Rngs(dropout=42))
    log(f"Registered qwix_rngs for LoRA dropout={LORA_DROPOUT}")

# ── Apply gradient checkpointing (remat) to each decoder layer ────────────────
# Use nnx.remat on the unbound DecoderLayer.__call__ method.
from tunix.models.gemma4 import model as _gemma4_mod
_gemma4_mod.DecoderLayer.__call__ = nnx.remat(_gemma4_mod.DecoderLayer.__call__)
log("Applied nnx.remat to DecoderLayer.__call__")

# ── Tokenizer & dataset ───────────────────────────────────────────────────────
with _Phase("Tokenizer load"):
    tokenizer = tokenizer_lib.Tokenizer(tokenizer_path=TOKENIZER_PATH)

with _Phase("Dataset build"):
    train_ds, val_ds, ds_stats = codev_data_lib.create_codev_datasets(
        tokenizer=tokenizer,
        global_batch_size=MICRO_BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        val_fraction=0.0,  # train_2.py: no eval split
        dev_subsample=DATASET_LIMIT,
    )
log(f"CodeV-R1: raw={ds_stats['raw']}  kept={ds_stats['kept']}  "
    f"dropped_long={ds_stats['dropped_long']}  dropped_empty={ds_stats['dropped_empty']}")

# Derive max_steps from epoch count for logging/profiling (trainer also stops on
# StopIteration from the data iterator).
_STEPS_PER_EPOCH = ds_stats["train_rows"] // MICRO_BATCH_SIZE
MAX_STEPS = max(1, _STEPS_PER_EPOCH * NUM_TRAIN_EPOCHS // GRAD_ACCUM_STEPS)
log(f"steps_per_epoch={_STEPS_PER_EPOCH}  max_optim_steps={MAX_STEPS}")

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
# Cosine schedule matches train_2.py: 100-step warmup, decay to 0.
# Clamp warmup if the run is shorter than WARMUP_STEPS (dev/subsample mode).
_warmup = min(WARMUP_STEPS, max(1, MAX_STEPS // 10))
if _warmup != WARMUP_STEPS:
    log(f"Warmup clamped: {WARMUP_STEPS} → {_warmup} (max_steps={MAX_STEPS})")
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=LR,
    warmup_steps=_warmup,
    decay_steps=MAX_STEPS,
    end_value=LR_END_VALUE,
)
optimizer = optax.adamw(learning_rate=schedule, weight_decay=WEIGHT_DECAY)

# ── Train ─────────────────────────────────────────────────────────────────────
logging_options = metrics_logger.MetricsLoggerOptions(
    log_dir=LOG_DIR,
    flush_every_n_steps=10,
)

# Orbax checkpoint options: save every CKPT_EVERY optimizer steps, retain
# CKPT_KEEP rollouts (mirrors train_2.py@59f2e67 save_strategy="steps",
# save_steps=100, save_total_limit=3).
import orbax.checkpoint as _ocp
ckpt_options = _ocp.CheckpointManagerOptions(
    save_interval_steps=CKPT_EVERY,
    max_to_keep=CKPT_KEEP,
)

training_config = peft_trainer.TrainingConfig(
    eval_every_n_steps=EVAL_EVERY,
    max_steps=MAX_STEPS,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    metrics_logging_options=logging_options,
    checkpoint_root_directory=CKPT_DIR,
    checkpointing_options=ckpt_options,
)

LOG_FILE_PATH = os.path.expanduser("~/gemma4_lora_sft.log")
RUN_JSON_PATH = "training_run.json"


def _build_run_metrics(*, final: bool, current_step: int | None = None) -> dict:
    """Assemble the training_run.json payload.

    Safe to call mid-run; unavailable metrics (e.g. eval when not configured)
    are silently skipped. Relies on module-level names defined above:
    ds_stats, total, trainable, trainer, _total_bytes, and recipe constants.
    """
    payload = {
        "run_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "wall_time_seconds": round(time.time() - _RUN_START, 1),
        "is_final": bool(final),
        "current_step": current_step,
        "config": {
            "model": MODEL_GCS_PATH,
            "method": method,
            "use_quantization": USE_QUANTIZATION,
            "lora_rank": RANK,
            "lora_alpha": ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "learning_rate": LR,
            "max_optim_steps": MAX_STEPS,
            "micro_batch_size": MICRO_BATCH_SIZE,
            "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
            "effective_batch": EFFECTIVE_BATCH,
            "max_seq_len": MAX_SEQ_LEN,
            "num_train_epochs": NUM_TRAIN_EPOCHS,
            "eval_every_n_steps": EVAL_EVERY,
            "optimizer": "adamw",
            "weight_decay": WEIGHT_DECAY,
            "warmup_steps": WARMUP_STEPS,
            "lr_end_value": LR_END_VALUE,
            "checkpoint_dir": CKPT_DIR,
            "ckpt_every_n_steps": CKPT_EVERY,
            "ckpt_keep": CKPT_KEEP,
            "log_every_n_steps": LOG_EVERY,
            "dataset": "zhuyaoyu/CodeV-R1-dataset:codev_r1_sft.jsonl",
            "dataset_stats": ds_stats,
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

    try:
        ml = trainer.metrics_logger
        prefix = trainer.metrics_prefix
        import numpy as _np
        for mode_name in ("train", "eval"):
            section = {}
            for metric_name in ("loss", "perplexity", "grad_norm"):
                try:
                    if ml.metric_exists(prefix, metric_name, mode_name):
                        history = ml.get_metric_history(prefix, metric_name, mode_name)
                        arr = _np.asarray(history, dtype=float)
                        section[metric_name] = [round(float(v), 6) for v in arr]
                        if arr.size > 0:
                            section[f"{metric_name}_mean"] = round(float(arr.mean()), 6)
                            section[f"{metric_name}_final"] = round(float(arr[-1]), 6)
                except Exception:
                    continue
            payload["training" if mode_name == "train" else "eval"] = section
    except Exception:
        pass

    payload["device_memory_current"] = []
    for d in jax.local_devices():
        s = d.memory_stats() or {}
        payload["device_memory_current"].append({
            "device": str(d),
            "used_gb": round(s.get("bytes_in_use", 0) / 1e9, 2),
            "limit_gb": round(s.get("bytes_limit", 0) / 1e9, 2),
        })
    return payload


def _upload_artifacts_to_gcs(step: int) -> None:
    """Snapshot log + fresh training_run.json to GCS alongside the checkpoint.

    Files land in <CKPT_DIR>/artifacts/step-<N>/ so they travel with the
    orbax checkpoint that was just written. Non-fatal on failure.
    """
    try:
        import json as _json_mod
        payload = _build_run_metrics(final=False, current_step=step)
        with open(RUN_JSON_PATH, "w") as _f:
            _json_mod.dump(payload, _f, indent=2)
    except Exception as e:
        log(f"artifact upload: skip run JSON ({e})")
        return
    try:
        import gcsfs as _gcsfs
        fs = _gcsfs.GCSFileSystem()
        base = CKPT_DIR.rstrip("/") + f"/artifacts/step-{step}"
        remote_json = f"{base}/training_run.json"
        remote_log  = f"{base}/gemma4_lora_sft.log"
        fs.put(RUN_JSON_PATH, remote_json)
        if os.path.exists(LOG_FILE_PATH):
            fs.put(LOG_FILE_PATH, remote_log)
        log(f"artifact upload @step{step}: gs://{base.removeprefix('gs://')}/")
    except Exception as e:
        log(f"artifact upload @step{step} failed: {e}")


# Per-step stdout logger. Tunix's progress bar auto-disables when stderr is
# not a TTY (e.g. under nohup → logfile), so without this the run produces no
# step-level output until the final JSON dump.
class _StdoutStepLogger(sft_hooks.TrainingHooks):
    def __init__(self, every: int = 1):
        self._every = every

    def on_train_start(self, train_ctx): pass
    def on_train_end(self, train_ctx): pass
    def on_train_step_start(self, train_ctx): pass
    def on_eval_step_start(self, train_ctx): pass
    def on_eval_step_end(self, train_ctx, eval_loss):
        log(f"eval_loss={float(eval_loss):.4f}")

    def on_train_step_end(self, train_ctx, train_step, train_loss):
        # Upload log + JSON to GCS on the same cadence as orbax checkpoints.
        # orbax saves when train_step % save_interval_steps == 0 (non-zero),
        # so we mirror that condition.
        if CKPT_EVERY and train_step > 0 and train_step % CKPT_EVERY == 0:
            _upload_artifacts_to_gcs(train_step)

        if train_step % self._every != 0:
            return
        try:
            lr = float(schedule(train_step))
        except Exception:
            lr = float("nan")
        gn = None
        try:
            gh = train_ctx.metrics_logger.get_metric_history(
                train_ctx.metrics_prefix, "grad_norm", "train"
            )
            if len(gh) > 0:
                gn = float(gh[-1])
        except Exception:
            pass
        parts = [f"step={train_step:>4d}/{MAX_STEPS}", f"loss={float(train_loss):.4f}", f"lr={lr:.2e}"]
        if gn is not None:
            parts.append(f"grad_norm={gn:.3f}")
        log("  ".join(parts))

trainer = peft_trainer.PeftTrainer(
    lora_model, optimizer, training_config
).with_gen_model_input_fn(gen_model_input_fn)
# with_training_hooks returns None (no self-return), so call it separately.
trainer.with_training_hooks(_StdoutStepLogger(every=LOG_EVERY))

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

# ── Dump final run metrics + upload to GCS ───────────────────────────────────
import json as _json

_run_metrics = _build_run_metrics(final=True, current_step=None)
with open(RUN_JSON_PATH, "w") as _jf:
    _json.dump(_run_metrics, _jf, indent=2)
log(f"Run metrics saved to {RUN_JSON_PATH}")

# Final artifact snapshot to <CKPT_DIR>/artifacts/final/
try:
    import gcsfs as _gcsfs
    _fs = _gcsfs.GCSFileSystem()
    _final_base = CKPT_DIR.rstrip("/") + "/artifacts/final"
    _fs.put(RUN_JSON_PATH, f"{_final_base}/training_run.json")
    if os.path.exists(LOG_FILE_PATH):
        _fs.put(LOG_FILE_PATH, f"{_final_base}/gemma4_lora_sft.log")
    log(f"Final artifacts uploaded to gs://{_final_base.removeprefix('gs://')}/")
except Exception as _e:
    log(f"Final artifact upload failed: {_e}")

log(f"Done. {method} adapters saved to: {CKPT_DIR}")
log(f"Total wall time: {time.time() - _RUN_START:.0f}s")
