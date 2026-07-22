"""Two model classes, both built from the same DecoderBlock:

- AlgorithmicTransformer: standard dense baseline, N distinct decoder
  blocks stacked (Table 1 sizes: Small/Medium/Large).
- LoopedTransformer: the paper's dedicated iterative-task baseline
  (Sec 1.3) -- a *single* shared decoder block applied k_iters times,
  with a learnable per-iteration timestep embedding added to the
  residual stream before each application.

Pre-LN blocks throughout (Norm -> sublayer -> residual), matching the
architecture diagram in the pre-defense slides.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .attention import GroupedQueryAttention
from .config import LoopConfig, ModelConfig
from .ffn import FFN
from .positional_encoding import PositionalEncoding, build_positional_encoding


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, config: ModelConfig, pos_enc: PositionalEncoding):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, config.attention, pos_enc)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, config.ffn)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), positions))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class _BaseAlgorithmicModel(nn.Module):
    """Shared embedding/head plumbing between the dense and looped
    variants, so both stay in sync automatically."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.pos_enc = build_positional_encoding(
            config.positional_encoding, config.d_model,
            config.attention.n_heads, config.attention.head_dim, config.max_seq_len,
        )
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.norm_out = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def _positions(self, input_ids: Tensor) -> Tensor:
        b, t = input_ids.shape
        return torch.arange(t, device=input_ids.device).unsqueeze(0).expand(b, t)

    def _embed(self, input_ids: Tensor, positions: Tensor) -> Tensor:
        return self.pos_enc.embed(self.token_emb(input_ids), positions)

    def num_parameters(self, exclude_embeddings: bool = True) -> int:
        """Table 1 of the paper reports params *excluding* embeddings."""
        total = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            total -= sum(p.numel() for p in self.token_emb.parameters())
            total -= sum(p.numel() for p in self.lm_head.parameters())
            pe = getattr(self.pos_enc, "pos_emb", None)  # only LearnedAbsolutePE has this
            if pe is not None:
                total -= sum(p.numel() for p in pe.parameters())
        return total


class AlgorithmicTransformer(_BaseAlgorithmicModel):
    """Standard dense decoder-only transformer, N distinct blocks."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.blocks = nn.ModuleList(
            DecoderBlock(config.d_model, config, self.pos_enc) for _ in range(config.n_layers)
        )

    def forward(self, input_ids: Tensor) -> Tensor:
        positions = self._positions(input_ids)
        x = self._embed(input_ids, positions)
        for block in self.blocks:
            x = block(x, positions)
        return self.lm_head(self.norm_out(x))

    @torch.no_grad()
    def get_layer_activations(self, input_ids: Tensor, layer_idx: int) -> Tensor:
        """Residual stream after decoder block `layer_idx` (0-indexed).
        Used for representation-similarity analysis (CKA); never called
        during training."""
        if not (0 <= layer_idx < len(self.blocks)):
            raise ValueError(f"layer_idx must be in [0, {len(self.blocks) - 1}]")
        positions = self._positions(input_ids)
        x = self._embed(input_ids, positions)
        for i, block in enumerate(self.blocks):
            x = block(x, positions)
            if i == layer_idx:
                return x
        return x


class LoopedTransformer(_BaseAlgorithmicModel):
    """A single shared DecoderBlock applied `k_iters` times, with a
    learnable timestep embedding -- the paper's dedicated baseline for
    iterative tasks (Sec 1.3), matched in effective depth to the dense
    baselines it is compared against (k_iters is chosen externally to
    match n_layers of the dense model it's being compared to)."""

    def __init__(self, config: ModelConfig, loop_config: LoopConfig):
        super().__init__(config)
        self.loop_config = loop_config
        self.block = DecoderBlock(config.d_model, config, self.pos_enc)  # single shared block
        self.timestep_emb = nn.Embedding(loop_config.k_iters, config.d_model)

    def forward(self, input_ids: Tensor) -> Tensor:
        positions = self._positions(input_ids)
        x = self._embed(input_ids, positions)
        for step in range(self.loop_config.k_iters):
            step_emb = self.timestep_emb.weight[step].view(1, 1, -1)
            x = self.block(x + step_emb, positions)
        return self.lm_head(self.norm_out(x))

    @torch.no_grad()
    def get_iteration_activations(self, input_ids: Tensor, step: int) -> Tensor:
        """Residual stream after loop iteration `step` (0-indexed) --
        the looped-model analogue of get_layer_activations, since there
        is no fixed set of distinct layers to index into."""
        if not (0 <= step < self.loop_config.k_iters):
            raise ValueError(f"step must be in [0, {self.loop_config.k_iters - 1}]")
        positions = self._positions(input_ids)
        x = self._embed(input_ids, positions)
        for i in range(self.loop_config.k_iters):
            step_emb = self.timestep_emb.weight[i].view(1, 1, -1)
            x = self.block(x + step_emb, positions)
            if i == step:
                return x
        return x
