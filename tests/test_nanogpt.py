"""Unit tests for NanoGPT-124M architecture and Symplectic Muon training pipeline."""

import torch
import pytest

from symplectic_muon.nanogpt import GPTConfig, NanoGPT, partition_parameters
from symplectic_muon.optimizer import StandardMuon, SymplecticMuon


def test_nanogpt_parameter_shapes():
    """Verify standard GPT-2 124M structural parameters and weight tying."""
    config = GPTConfig(
        block_size=1024,
        vocab_size=50257,
        n_layer=12,
        n_head=12,
        n_embd=768,
        bias=True,
    )
    model = NanoGPT(config)

    # Weight tying invariant: language model head shares weights with token embeddings
    assert model.lm_head.weight is model.transformer.wte.weight

    # Total unique parameters should match GPT-2 124M (~124.4M params)
    unique_params = sum(p.numel() for p in set(model.parameters()))
    assert 120_000_000 <= unique_params <= 130_000_000, f"Unexpected parameter count: {unique_params}"

    # Verify layer count and key tensor dimensions
    assert len(model.transformer.h) == 12
    for block in model.transformer.h:
        assert block.attn.c_attn.weight.shape == (2304, 768)
        assert block.attn.c_proj.weight.shape == (768, 768)
        assert block.mlp.c_fc.weight.shape == (3072, 768)
        assert block.mlp.c_proj.weight.shape == (768, 3072)


def test_2d_matrix_identification_and_partitioning():
    """Verify parameter partitioning: 2D linear weights to Muon, 1D/embeddings to AdamW."""
    config = GPTConfig(
        block_size=128,
        vocab_size=1000,
        n_layer=12,
        n_head=4,
        n_embd=128,
        bias=True,
    )
    model = NanoGPT(config)
    muon_params, adamw_params = model.get_parameter_groups()

    # 4 2D linear weight matrices per layer (c_attn, c_proj, c_fc, mlp.c_proj) * 12 layers = 48
    assert len(muon_params) == 48, f"Expected 48 2D matrices, got {len(muon_params)}"

    # Every matrix in muon_params must be strictly 2D with min dimension > 1
    for p in muon_params:
        assert p.ndim == 2
        assert min(p.shape) > 1

    # Embeddings must not be in muon_params
    muon_param_ids = {id(p) for p in muon_params}
    assert id(model.transformer.wte.weight) not in muon_param_ids
    assert id(model.transformer.wpe.weight) not in muon_param_ids

    # Embedding weights must be present in adamw_params
    adamw_param_ids = {id(p) for p in adamw_params}
    assert id(model.transformer.wte.weight) in adamw_param_ids
    assert id(model.transformer.wpe.weight) in adamw_param_ids

    # Partitioning must be disjoint and complete over unique parameters
    all_unique_ids = {id(p) for p in model.parameters()}
    assert muon_param_ids.isdisjoint(adamw_param_ids)
    assert (muon_param_ids | adamw_param_ids) == all_unique_ids

    # Check standalone partition_parameters helper
    m_p, a_p = partition_parameters(model)
    assert len(m_p) == len(muon_params)
    assert len(a_p) == len(adamw_params)


def test_forward_and_hidden_states():
    """Verify forward pass, loss calculation, and hidden states extraction."""
    torch.manual_seed(42)
    config = GPTConfig(
        block_size=32,
        vocab_size=256,
        n_layer=4,
        n_head=2,
        n_embd=64,
        bias=True,
    )
    model = NanoGPT(config)
    idx = torch.randint(0, 256, (2, 16))
    targets = torch.randint(0, 256, (2, 16))

    # Standard forward without targets
    logits = model(idx, output_hidden_states=False)
    assert logits.shape == (2, 16, 256)

    # Standard forward with targets
    logits, loss = model(idx, targets=targets, output_hidden_states=False)
    assert logits.shape == (2, 16, 256)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    # Forward with hidden states
    logits, loss, hidden_states = model(idx, targets=targets, output_hidden_states=True)
    assert logits.shape == (2, 16, 256)
    assert torch.isfinite(loss)
    # 1 initial embedding + 4 transformer layers + 1 final layernorm = 6 states
    assert len(hidden_states) == 6
    for h in hidden_states:
        assert h.shape == (2, 16, 64)


def test_symplectic_muon_5step_optimization_loss_decrease():
    """Verify a 5-step forward-backward-optimizer step with SymplecticMuon runs without NaN and decreases loss."""
    torch.manual_seed(42)
    config = GPTConfig(
        block_size=32,
        vocab_size=128,
        n_layer=2,
        n_head=2,
        n_embd=64,
        bias=True,
    )
    model = NanoGPT(config)
    muon_params, adamw_params = model.get_parameter_groups()

    opt_muon = SymplecticMuon(muon_params, lr=0.02, momentum=0.95, weight_decay=0.01)
    opt_adamw = torch.optim.AdamW(adamw_params, lr=0.001, weight_decay=0.01)

    # Fixed training batch to test loss reduction
    idx = torch.randint(0, 128, (4, 16))
    targets = torch.randint(0, 128, (4, 16))

    losses = []
    for step in range(5):
        opt_muon.zero_grad(set_to_none=True)
        opt_adamw.zero_grad(set_to_none=True)

        _, loss = model(idx, targets=targets)
        assert torch.isfinite(loss), f"Loss is non-finite at step {step}: {loss.item()}"
        loss.backward()

        # Check gradients are finite
        for p in model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Gradient contains NaN/Inf at step {step}"

        opt_muon.step()
        opt_adamw.step()

        # Check parameters are finite
        for p in model.parameters():
            assert torch.isfinite(p).all(), f"Parameter contains NaN/Inf after step {step}"

        losses.append(loss.item())

    # Verify that loss monotonically or overall decreases over 5 steps
    assert losses[-1] < losses[0], f"Loss did not decrease: initial={losses[0]:.4f}, final={losses[-1]:.4f}"


def test_standard_muon_5step_optimization():
    """Verify standard Muon baseline runs 5 steps without NaN and decreases loss."""
    torch.manual_seed(42)
    config = GPTConfig(
        block_size=32,
        vocab_size=128,
        n_layer=2,
        n_head=2,
        n_embd=64,
        bias=True,
    )
    model = NanoGPT(config)
    muon_params, adamw_params = model.get_parameter_groups()

    opt_muon = StandardMuon(muon_params, lr=0.02, momentum=0.95, weight_decay=0.01)
    opt_adamw = torch.optim.AdamW(adamw_params, lr=0.001, weight_decay=0.01)

    idx = torch.randint(0, 128, (4, 16))
    targets = torch.randint(0, 128, (4, 16))

    losses = []
    for step in range(5):
        opt_muon.zero_grad(set_to_none=True)
        opt_adamw.zero_grad(set_to_none=True)

        _, loss = model(idx, targets=targets)
        loss.backward()

        opt_muon.step()
        opt_adamw.step()

        losses.append(loss.item())

    assert torch.isfinite(torch.tensor(losses)).all()
    assert losses[-1] < losses[0]
