#!/usr/bin/env python3
"""Tokenize FineWeb-Edu sample-10BT (ODC-By) with GPT-2 BPE into uint16 memmaps for train_nanogpt.py.
The first --val-tokens worth of documents go to val.bin, the rest to train.bin; documents never straddle splits.
Each document ends with the GPT-2 <|endoftext|> token. Streams, so memory stays flat.
"""

import argparse
import os

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--train-tokens", type=int, default=1_200_000_000)
    p.add_argument("--val-tokens", type=int, default=10_000_000)
    p.add_argument("--name", default="sample-10BT")
    p.add_argument("--batch-docs", type=int, default=1024)
    a = p.parse_args()

    import tiktoken
    from datasets import load_dataset
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    os.makedirs(a.out_dir, exist_ok=True)
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name=a.name, split="train", streaming=True)

    targets = [("val.bin", a.val_tokens), ("train.bin", a.train_tokens)]
    split, (fname, budget) = 0, targets[0]
    out = np.memmap(os.path.join(a.out_dir, fname), dtype=np.uint16, mode="w+", shape=(budget,))
    pos, buf = 0, []
    for doc in ds:
        buf.append(doc["text"])
        if len(buf) < a.batch_docs:
            continue
        for toks in enc.encode_ordinary_batch(buf):
            toks.append(eot)
            n = min(len(toks), budget - pos)
            out[pos:pos + n] = toks[:n]
            pos += n
            if pos == budget:
                out.flush()
                print(f"wrote {fname}: {pos:,} tokens")
                split += 1
                if split == len(targets):
                    return
                fname, budget = targets[split]
                out = np.memmap(os.path.join(a.out_dir, fname), dtype=np.uint16, mode="w+", shape=(budget,))
                pos = 0
                break  # drop the rest of this batch so no document straddles splits
        buf = []
        if pos and pos % 50_000_000 < 600_000:
            print(f"  {fname}: {pos / 1e6:.0f}M / {budget / 1e6:.0f}M tokens")


if __name__ == "__main__":
    main()
