"""NanoGPT-124M model implementation with parameter partitioning for Muon."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    norm_type: str = "layernorm"


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with scaled dot-product attention."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    """Transformer block with pre-normalization."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        norm_cls = RMSNorm if config.norm_type == "rmsnorm" else nn.LayerNorm
        self.ln_1 = norm_cls(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = norm_cls(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class NanoGPT(nn.Module):
    """NanoGPT-124M language model with tied token embeddings."""

    def __init__(self, config: Optional[GPTConfig] = None):
        super().__init__()
        if config is None:
            config = GPTConfig()
        self.config = config

        norm_cls = RMSNorm if config.norm_type == "rmsnorm" else nn.LayerNorm
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=norm_cls(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Tie weights between token embeddings and final linear projection
        self.lm_head.weight = self.transformer.wte.weight

        self.apply(self._init_weights)
        # Scaled initialization for residual projections
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            if hasattr(module, "weight") and module.weight is not None:
                torch.nn.init.ones_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, ...]]]:
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Sequence length {t} exceeds block size {self.config.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        hidden_states: Optional[List[torch.Tensor]] = [x] if output_hidden_states else None
        for block in self.transformer.h:
            x = block(x)
            if output_hidden_states and hidden_states is not None:
                hidden_states.append(x)

        x = self.transformer.ln_f(x)
        if output_hidden_states and hidden_states is not None:
            hidden_states.append(x)

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        if output_hidden_states:
            assert hidden_states is not None
            return logits, loss, tuple(hidden_states)
        if targets is not None:
            return logits, loss
        return logits

    def get_parameter_groups(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        """Partition parameters into 2D linear weight matrices and 1D/embedding parameters.

        Returns:
            muon_params: 2D linear weights in transformer blocks for Muon orthogonalization.
            adamw_params: 1D normalization weights, biases, and embedding tables for AdamW.
        """
        muon_params: List[nn.Parameter] = []
        adamw_params: List[nn.Parameter] = []
        seen_ids = set()

        embedding_param_ids = {
            id(self.transformer.wte.weight),
            id(self.transformer.wpe.weight),
        }

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in seen_ids:
                continue
            seen_ids.add(id(param))

            # 2D weight matrices in transformer blocks (excluding embeddings)
            if param.ndim == 2 and id(param) not in embedding_param_ids and min(param.shape) > 1:
                muon_params.append(param)
            else:
                adamw_params.append(param)

        return muon_params, adamw_params


def partition_parameters(model: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """Partition model parameters into (muon_params, adamw_params)."""
    if hasattr(model, "get_parameter_groups"):
        return model.get_parameter_groups()

    muon_params: List[nn.Parameter] = []
    adamw_params: List[nn.Parameter] = []
    seen_ids = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen_ids:
            continue
        seen_ids.add(id(param))

        # Check if 2D linear weight (exclude embeddings and 1D vectors)
        if param.ndim == 2 and min(param.shape) > 1 and "embed" not in name and "wte" not in name and "wpe" not in name:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    return muon_params, adamw_params
