"""
CodeV-R1 SFT data pipeline for Tunix (TPU).

Mirrors the recipe in h2loop/gemma4_sft_h100/train_2.py:
  - load zhuyaoyu/CodeV-R1-dataset (codev_r1_sft.jsonl)
  - wrap with a Verilog system prompt
  - strip <think>…</think> reasoning; keep only the ```verilog …``` block
  - normalize to alternating user/model turns (Gemma has no system role)
  - drop (NOT truncate) rows longer than max_seq_len
  - compute an assistant-only loss mask (only model-turn tokens contribute)

Outputs TrainingInput batches via grain, compatible with tunix PeftTrainer.
"""

import re

import grain.python as grain
import numpy as np
from datasets import load_dataset
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.sft.peft_trainer import TrainingInput


SYSTEM_PROMPT = (
    "You are an expert Verilog hardware design engineer. "
    "Given a natural-language specification, write correct, synthesizable "
    "Verilog code that implements the requested module. Output only the "
    "Verilog code inside a ```verilog ... ``` fenced block."
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_VERILOG_RE = re.compile(r"```verilog\s*(.*?)```", re.DOTALL)


def _strip_reasoning(response: str) -> str:
    m = _ANSWER_RE.search(response)
    body = m.group(1).strip() if m else _THINK_RE.sub("", response).strip()
    v = _VERILOG_RE.search(body)
    if v:
        return f"```verilog\n{v.group(1).strip()}\n```"
    return body


def _to_gemma_messages(example: dict) -> list[dict]:
    """Build a conversation in Gemma's native role vocabulary (user/model).

    Match upstream train_2.py behavior exactly: upstream builds
    [system, user, assistant], but its Gemma chat template only handles user
    and assistant branches — the system message is silently dropped by the
    template. SYSTEM_PROMPT is therefore defined but unused in the training
    input, just like upstream. We mirror that here by emitting only the user
    and model turns.
    """
    user = example["prompt"].strip()
    model = _strip_reasoning(example["response"])
    return [
        {"role": "user", "content": user},
        {"role": "model", "content": model},
    ]


class CodeVDataSource(grain.RandomAccessDataSource):
    """Loads and pre-filters the CodeV-R1 SFT split.

    Preprocessing happens once at construction time so length-filtering can
    drop over-long rows (mirroring train_2.py, which drops rather than
    truncates). Each element is a pre-tokenized (tokens, mask) pair.
    """

    def __init__(
        self,
        tokenizer: tokenizer_lib.Tokenizer,
        max_seq_len: int,
        dev_subsample: int | None = None,
    ):
        raw = load_dataset(
            "zhuyaoyu/CodeV-R1-dataset",
            data_files="codev_r1_sft.jsonl",
            split="train",
        )
        if dev_subsample is not None:
            raw = raw.select(range(min(dev_subsample, len(raw))))

        kept: list[tuple[np.ndarray, np.ndarray]] = []
        dropped_long = 0
        dropped_empty = 0
        for row in raw:
            messages = _to_gemma_messages(row)
            if len(messages) < 2 or messages[-1]["role"] != "model":
                dropped_empty += 1
                continue

            tokens, mask = _tokenize_with_assistant_mask(tokenizer, messages)
            if tokens.size == 0 or int(mask.sum()) == 0:
                dropped_empty += 1
                continue
            if tokens.size > max_seq_len:
                dropped_long += 1
                continue
            kept.append((tokens, mask))

        self._rows = kept
        self._stats = {
            "raw": len(raw),
            "kept": len(kept),
            "dropped_long": dropped_long,
            "dropped_empty": dropped_empty,
        }

    @property
    def stats(self) -> dict:
        return self._stats

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return self._rows[idx]


def _tokenize_with_assistant_mask(
    tokenizer: tokenizer_lib.Tokenizer, messages: list[dict]
) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize the full chat and build a mask that is True only on model-turn
    tokens (assistant-only loss, matching SFTConfig(assistant_only_loss=True)).

    Upstream train_2.py uses {% generation %} tags in the chat template plus
    trl's assistant_only_loss=True to compute the mask. Tunix's tokenizer
    adapter doesn't emit those markers, so we derive the mask from the already-
    rendered chat string by tokenizing each segment with BOS/EOS disabled and
    stitching offsets together.
    """
    full_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=False
    )
    full_tokens = np.asarray(tokenizer.encode(full_text), dtype=np.int32)

    # Encode raw pieces without BOS/EOS so concatenated lengths line up with
    # what apply_chat_template produced. SentencePiece gets its BOS/EOS from
    # SetEncodeExtraOptions("bos:eos"); the underlying processor can still
    # encode a string without those options.
    sp = getattr(tokenizer, "_tokenizer", None)

    def _enc_clean(text: str) -> list[int]:
        if not text:
            return []
        if sp is not None and hasattr(sp, "EncodeAsIds"):
            return sp.EncodeAsIds(text, add_bos=False, add_eos=False)
        return tokenizer.encode(text)

    # Tunix renders as: <start_of_turn>{role}\n{content}<end_of_turn>\n
    # The assistant span we want to mask is {content}<end_of_turn>\n — i.e.
    # everything that comes *after* the turn header. (We include <end_of_turn>
    # so the model is trained to emit the turn terminator.)
    bos_len = 1 if bool(getattr(tokenizer, "_tokenizer", None)) and \
        hasattr(tokenizer, "bos_id") else 0
    cursor = bos_len  # full_tokens starts with BOS
    mask = np.zeros(len(full_tokens), dtype=np.bool_)
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        header = f"<start_of_turn>{role}\n"
        body   = f"{content}<end_of_turn>\n"

        header_ids = _enc_clean(header)
        body_ids   = _enc_clean(body)

        header_end = cursor + len(header_ids)
        body_end   = header_end + len(body_ids)

        if role == "model":
            mask[header_end:min(body_end, len(mask))] = True

        cursor = body_end
        if cursor >= len(mask):
            break

    return full_tokens, mask


class _PadToLen(grain.MapTransform):
    def __init__(self, max_seq_len: int, pad_id: int):
        self._max_seq_len = max_seq_len
        self._pad_id = pad_id

    def map(self, tokens_and_mask: tuple[np.ndarray, np.ndarray]) -> TrainingInput:
        tokens, mask = tokens_and_mask
        pad_len = self._max_seq_len - tokens.size
        if pad_len > 0:
            tokens = np.pad(tokens, (0, pad_len), constant_values=self._pad_id)
            mask = np.pad(mask, (0, pad_len), constant_values=False)
        elif pad_len < 0:
            tokens = tokens[: self._max_seq_len]
            mask = mask[: self._max_seq_len]
        return TrainingInput(input_tokens=tokens, input_mask=mask)


class _SlicedSource(grain.RandomAccessDataSource):
    def __init__(self, source, start: int, end: int):
        self._source = source
        self._start = start
        self._end = end

    def __len__(self) -> int:
        return self._end - self._start

    def __getitem__(self, idx: int):
        return self._source[self._start + idx]


def create_codev_datasets(
    tokenizer: tokenizer_lib.Tokenizer,
    global_batch_size: int,
    max_seq_len: int,
    num_train_epochs: int = 1,
    val_fraction: float = 0.0,
    dev_subsample: int | None = None,
) -> tuple[grain.DataLoader, grain.DataLoader | None, dict]:
    """Build train (and optionally eval) loaders over CodeV-R1.

    Returns (train_loader, val_loader_or_None, stats_dict). train_2.py uses no
    eval set; we keep val_fraction=0.0 for parity but support a held-out slice.
    """
    source = CodeVDataSource(tokenizer, max_seq_len, dev_subsample=dev_subsample)
    n = len(source)
    n_val = int(n * val_fraction)
    n_train = n - n_val

    train_src = _SlicedSource(source, 0, n_train)
    val_src = _SlicedSource(source, n_train, n) if n_val > 0 else None

    ops = [
        _PadToLen(max_seq_len, tokenizer.pad_id()),
        grain.Batch(batch_size=global_batch_size, drop_remainder=True),
    ]

    train_loader = grain.DataLoader(
        data_source=train_src,
        sampler=grain.IndexSampler(
            num_records=n_train,
            num_epochs=num_train_epochs,
            shard_options=grain.NoSharding(),
            shuffle=True,
            seed=42,
        ),
        operations=ops,
    )

    val_loader = None
    if val_src is not None:
        val_loader = grain.DataLoader(
            data_source=val_src,
            sampler=grain.IndexSampler(
                num_records=n_val,
                num_epochs=1,
                shard_options=grain.NoSharding(),
            ),
            operations=ops,
        )

    stats = dict(source.stats)
    stats.update({"train_rows": n_train, "val_rows": n_val})
    return train_loader, val_loader, stats
