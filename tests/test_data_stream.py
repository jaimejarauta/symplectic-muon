"""Synthetic stream: train and val must share one motif table (same distribution, different samples)."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from train_nanogpt import FastTokenStream  # noqa: E402


def test_val_uses_train_motif_table():
    kw = dict(data_path=None, block_size=32, batch_size=4, device=torch.device("cpu"), vocab_size=1000)
    tr = FastTokenStream(split="train", seed=42, **kw)
    va = FastTokenStream(split="val", seed=43, data_seed=42, **kw)
    assert torch.equal(tr.motifs, va.motifs)
    assert not torch.equal(tr.synth_tokens[:256], va.synth_tokens[:256])
