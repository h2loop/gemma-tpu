"""
Custom Grain data loader for agent trajectory JSONL files.
Converts multi-turn (system/user/assistant/tool) trajectories into
Gemma chat-template format for SFT.
"""

import glob
import json
import os
import random

import grain.python as grain
import jax
import numpy as np
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.sft.peft_trainer import TrainingInput


def _load_trajectories(data_dir: str) -> list[list[dict]]:
    """Load all JSONL trajectory files, each as a list of messages."""
    files = sorted(glob.glob(os.path.join(data_dir, "**/*.jsonl"), recursive=True))
    trajectories = []
    for f in files:
        msgs = []
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    msgs.append(json.loads(line))
        if msgs:
            trajectories.append(msgs)
    return trajectories


def _trajectory_to_chat(messages: list[dict]) -> list[dict[str, str]]:
    """Convert a trajectory into Gemma-compatible chat messages.

    - 'assistant' role → 'model'
    - tool_calls are serialized into the model's content
    - 'tool' role responses are folded into a 'user' turn (tool output)
    - 'system' messages become a 'user' preamble
    """
    chat = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "") or ""

        if role == "system":
            # Gemma doesn't have a system role; prepend as user context
            chat.append({"role": "user", "content": f"[System]\n{content}"})

        elif role == "user":
            chat.append({"role": "user", "content": content})

        elif role == "assistant":
            # Serialize tool_calls into the content so the model learns them
            parts = [content] if content else []
            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", {})
                parts.append(
                    f"[tool_call] {fn.get('name', '')}({fn.get('arguments', '')})"
                )
            chat.append({"role": "model", "content": "\n".join(parts)})

        elif role == "tool":
            # Tool output → user turn so model sees the result
            tool_name = msg.get("tool_name", "tool")
            chat.append({
                "role": "user",
                "content": f"[tool_result: {tool_name}]\n{content}",
            })

    # Merge consecutive same-role messages (Gemma requires alternating)
    merged = []
    for m in chat:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] += "\n" + m["content"]
        else:
            merged.append(m)

    # Ensure conversation starts with user and alternates
    if merged and merged[0]["role"] == "model":
        merged.insert(0, {"role": "user", "content": "(start)"})

    # Ensure it ends with model turn (what we want the model to learn)
    if merged and merged[-1]["role"] == "user":
        merged = merged[:-1]

    return merged


class TrajectoryDataSource(grain.RandomAccessDataSource):
    """Grain-compatible random access data source for trajectory files."""

    def __init__(self, data_dir: str, seed: int = 42):
        self._trajectories = _load_trajectories(data_dir)
        rng = random.Random(seed)
        rng.shuffle(self._trajectories)

    def __len__(self):
        return len(self._trajectories)

    def __getitem__(self, idx):
        return self._trajectories[idx]


class TokenizeTrajectory(grain.MapTransform):
    """Tokenize a trajectory into (input_tokens, target_mask) using chat template."""

    def __init__(self, tokenizer: tokenizer_lib.Tokenizer):
        self._tokenizer = tokenizer

    def map(self, trajectory: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        chat = _trajectory_to_chat(trajectory)
        if len(chat) < 2:
            return np.array([], dtype=np.int32), np.array([], dtype=np.bool_)

        # Full conversation
        full_text = self._tokenizer.apply_chat_template(
            chat, add_generation_prompt=False, tokenize=False
        )
        full_tokens = np.array(
            self._tokenizer.encode(full_text), dtype=np.int32
        )

        # Build mask: 1 for all model-turn tokens, 0 for user-turn tokens.
        # We do this by encoding incrementally and marking model segments.
        mask = np.zeros(len(full_tokens), dtype=np.bool_)
        prompt_so_far = ""
        for i, msg in enumerate(chat):
            # Text up to and including this message
            text_through = self._tokenizer.apply_chat_template(
                chat[: i + 1], add_generation_prompt=False, tokenize=False
            )
            if msg["role"] == "model":
                start_len = len(self._tokenizer.encode(prompt_so_far))
                end_len = len(self._tokenizer.encode(text_through))
                if end_len <= len(mask):
                    mask[start_len:end_len] = True
            prompt_so_far = text_through

        return full_tokens, mask


class BuildTrainInput(grain.MapTransform):
    """Pad/truncate and wrap into TrainingInput."""

    def __init__(self, max_seq_len: int, pad_id: int):
        self._max_seq_len = max_seq_len
        self._pad_id = pad_id

    def map(self, tokens_and_mask: tuple[np.ndarray, np.ndarray]) -> TrainingInput:
        tokens, mask = tokens_and_mask
        seq_len = len(tokens)

        # Truncate if needed
        if seq_len > self._max_seq_len:
            tokens = tokens[: self._max_seq_len]
            mask = mask[: self._max_seq_len]
            seq_len = self._max_seq_len

        # Pad if needed
        pad_len = self._max_seq_len - seq_len
        if pad_len > 0:
            tokens = np.pad(tokens, (0, pad_len), constant_values=self._pad_id)
            mask = np.pad(mask, (0, pad_len), constant_values=False)

        return TrainingInput(input_tokens=tokens, input_mask=mask)


class FilterEmpty(grain.FilterTransform):
    """Drop degenerate trajectories."""

    def __init__(self, min_tokens: int = 4):
        self._min_tokens = min_tokens

    def filter(self, element: TrainingInput) -> bool:
        return int(element.input_mask.sum()) >= self._min_tokens


class _SlicedSource(grain.RandomAccessDataSource):
    """A slice of another RandomAccessDataSource."""

    def __init__(self, source, start: int, end: int):
        self._source = source
        self._start = start
        self._end = end

    def __len__(self):
        return self._end - self._start

    def __getitem__(self, idx):
        return self._source[self._start + idx]


def create_trajectory_datasets(
    data_dir: str,
    global_batch_size: int,
    max_target_length: int,
    num_train_epochs: int | None,
    tokenizer: tokenizer_lib.Tokenizer,
    val_fraction: float = 0.10,
) -> tuple[grain.DataLoader, grain.DataLoader | None]:
    """Build train (and optionally val) data loaders from trajectory JSONL files."""
    source = TrajectoryDataSource(data_dir)
    n = len(source)
    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val

    train_source = _SlicedSource(source, 0, n_train)
    val_source = _SlicedSource(source, n_train, n)

    ops = [
        TokenizeTrajectory(tokenizer),
        BuildTrainInput(max_target_length, tokenizer.pad_id()),
        FilterEmpty(),
        grain.Batch(batch_size=global_batch_size, drop_remainder=True),
    ]

    train_loader = grain.DataLoader(
        data_source=train_source,
        sampler=grain.IndexSampler(
            num_records=n_train,
            num_epochs=num_train_epochs,
            shard_options=grain.NoSharding(),
        ),
        operations=ops,
    )

    val_loader = grain.DataLoader(
        data_source=val_source,
        sampler=grain.IndexSampler(
            num_records=n_val,
            num_epochs=1,
            shard_options=grain.NoSharding(),
        ),
        operations=ops,
    )

    return train_loader, val_loader
