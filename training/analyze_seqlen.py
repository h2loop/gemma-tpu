"""Analyze CodeV-R1 sequence-length distribution under the Gemma chat template.

Parallelized via datasets.map(num_proc=...). Uses the exact same system
prompt, reasoning-stripping, chat template, and tokenizer as codev_dataset.py.
"""
import argparse
import os
import numpy as np
from datasets import load_dataset
from tunix.generate import tokenizer_adapter as tokenizer_lib

import codev_dataset as cd


_TOKENIZER = None  # per-worker lazy singleton


def _init_worker(tokenizer_path: str):
    global _TOKENIZER
    _TOKENIZER = tokenizer_lib.Tokenizer(tokenizer_path=tokenizer_path)


def _measure(row, tokenizer_path: str):
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = tokenizer_lib.Tokenizer(tokenizer_path=tokenizer_path)
    msgs = cd._to_gemma_messages(row)
    if len(msgs) < 2 or msgs[-1]["role"] != "model":
        return {"full_len": 0, "asst_len": 0}
    tokens, mask = cd._tokenize_with_assistant_mask(_TOKENIZER, msgs)
    return {"full_len": int(tokens.size), "asst_len": int(mask.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer",
                    default="gs://gemma-data/tokenizers/tokenizer_gemma3.model")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num-proc", type=int, default=max(1, os.cpu_count() // 2))
    args = ap.parse_args()

    raw = load_dataset(
        "zhuyaoyu/CodeV-R1-dataset",
        data_files="codev_r1_sft.jsonl",
        split="train",
    )
    if args.limit:
        raw = raw.select(range(min(args.limit, len(raw))))

    measured = raw.map(
        _measure,
        fn_kwargs={"tokenizer_path": args.tokenizer},
        num_proc=args.num_proc,
        remove_columns=raw.column_names,
        desc="tokenize",
    )

    total = np.asarray(measured["full_len"])
    asst = np.asarray(measured["asst_len"])
    keep = total > 0
    total = total[keep]
    asst = asst[keep]

    qs = [50, 75, 90, 95, 99, 100]

    def row(name, a):
        pcts = np.percentile(a, qs).astype(int).tolist()
        print(f"{name:12s}  n={a.size:>6d}  "
              f"mean={a.mean():>7.1f}  std={a.std():>7.1f}  "
              f"min={a.min():>5d}  " +
              "  ".join(f"p{q}={v}" for q, v in zip(qs, pcts)))

    print(f"\nDataset: CodeV-R1 (rows scanned: {total.size}) workers={args.num_proc}")
    row("full_seq", total)
    row("asst_only", asst)

    for cap in (1024, 2048, 4096, 8192, 12288, 16384):
        frac_kept = (total <= cap).mean()
        print(f"max_seq_len={cap:>5d}: keeps {frac_kept*100:5.1f}% "
              f"({int((total<=cap).sum())}/{total.size})")


if __name__ == "__main__":
    main()
