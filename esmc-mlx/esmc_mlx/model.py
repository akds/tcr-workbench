"""ESM-C encoder adapted from MLX-LM PR #1484 (MIT, Apple Inc.).

Source: c26b9af872158d822a8c95589708eedd3b9c0831/mlx_lm/models/esmc.py.
Modifications: strict source-specific biases, Decoder keys, output selection,
padding equality, exact GELU, and optional stage capture without hidden stacks.
See licenses/MLX-LM-MIT.txt and THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

from .precision import configure_precision
import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig
from .rotary import NativeRoPE


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5
        d, eps = config.hidden_size, config.layer_norm_eps
        self.ln_qkv = nn.LayerNorm(d, eps=eps, bias=config.pre_norm_bias)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.q_ln = nn.LayerNorm(d, eps=eps, bias=config.qk_norm_bias)
        self.k_ln = nn.LayerNorm(d, eps=eps, bias=config.qk_norm_bias)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.rope = NativeRoPE(self.head_dim, base=config.rope_base)

    def __call__(self, x, mask, stages=None, prefix=""):
        batch, length, _ = x.shape
        q, k, v = mx.split(self.qkv(self.ln_qkv(x)), 3, axis=-1)
        q, k = self.q_ln(q), self.k_ln(k)
        if stages is not None:
            stages[prefix + "q_norm"], stages[prefix + "k_norm"] = q, k

        def heads(a):
            return a.reshape(batch, length, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        q, k, v = heads(q), heads(k), heads(v)
        q, k = self.rope(q), self.rope(k)
        if stages is not None:
            stages[prefix + "q_rope"], stages[prefix + "k_rope"] = q, k
        x = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self.out_proj(x.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class FFN(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        d, h = config.hidden_size, config.ffn_hidden
        self.ln = nn.LayerNorm(d, eps=config.layer_norm_eps, bias=config.pre_norm_bias)
        self.fc1 = nn.Linear(d, 2 * h, bias=False)
        self.fc2 = nn.Linear(h, d, bias=False)

    def __call__(self, x):
        gate, value = mx.split(self.fc1(self.ln(x)), 2, axis=-1)
        return self.fc2(nn.silu(gate) * value)


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn, self.ffn = Attention(config), FFN(config)
        self.scaling_factor = config.residue_scaling_factor

    def __call__(self, x, mask, stages=None, prefix=""):
        attention = self.attn(x, mask, stages, prefix)
        x = x + attention / self.scaling_factor
        ffn = self.ffn(x)
        x = x + ffn / self.scaling_factor
        if stages is not None:
            stages[prefix + "attention"] = attention
            stages[prefix + "ffn"] = ffn
            stages[prefix + "residual"] = x
        return x


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.blocks = [Block(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps,
                                 bias=config.final_norm_bias)


class ESMC(nn.Module):
    """Bidirectional masked encoder; no autoregressive KV cache or chain splitting.

    Inputs are integer token IDs [B,L]. ``positions`` selects head positions [B,P]
    after the complete encoder, preserving attention to all context residues.
    ``capture`` retains selected stages only for numerical debugging.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        configure_precision()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.transformer = Transformer(config)
        self.sequence_head = [
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(approx="none"),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            nn.Linear(config.hidden_size, config.vocab_size),
        ]

    def __call__(self, tokens, positions=None, *, output_embeddings=True,
                 compute_logits=True, capture=False):
        if tokens.ndim != 2 or not all(tokens.shape):
            raise ValueError("tokens must be a nonempty [batch,length] array")
        if not mx.issubdtype(tokens.dtype, mx.integer):
            raise ValueError("tokens must have an integer dtype")
        if tokens.shape[1] > self.config.max_position_embeddings:
            raise ValueError("context exceeds configured maximum total token length")
        valid = tokens != self.config.pad_token_id
        invalid_tokens, empty_row, unpadded = mx.stack([
            mx.any((tokens < 0) | (tokens >= self.config.vocab_size)),
            mx.any(~mx.any(valid, axis=1)), mx.all(valid),
        ]).tolist()
        if invalid_tokens:
            raise ValueError("token ID is outside the model vocabulary")
        if empty_row:
            raise ValueError("every input row must contain at least one non-padding token")
        if positions is not None:
            if (positions.ndim != 2 or positions.shape[0] != tokens.shape[0]
                    or not positions.shape[1] or not mx.issubdtype(positions.dtype, mx.integer)):
                raise ValueError("positions must be a nonempty integer [batch,positions] array")
            if bool(mx.any((positions < 0) | (positions >= tokens.shape[1])).item()):
                raise ValueError("requested position is outside the input")
        # The two authoritative sources differ for padded query positions.
        # Decoder permits valid->valid and pad->pad; published Biohub permits
        # every query to attend to valid keys. Valid-token outputs agree.
        mask = None
        if not unpadded:
            mask = (valid[:, None, :, None] == valid[:, None, None, :]
                    if self.config.source_variant == "decodertcr-lightning-v03"
                    else valid[:, None, None, :])
        stages = {} if capture else None
        selected = {0, (len(self.transformer.blocks) - 1) // 2,
                    len(self.transformer.blocks) - 1} if capture is True else set(capture or ())
        if any(type(i) is not int or i < 0 or i >= len(self.transformer.blocks) for i in selected):
            raise ValueError("capture layer indices must be integers inside the model")
        x = self.embed(tokens)
        if stages is not None:
            stages["embedding"] = x
        for index, block in enumerate(self.transformer.blocks):
            x = block(x, mask, stages if index in selected else None, f"block{index}.")
        prenorm = x
        postnorm = self.transformer.norm(x)
        result = {}
        if output_embeddings:
            result.update(prenorm=prenorm, postnorm=postnorm)
        if compute_logits:
            x = postnorm if positions is None else postnorm[mx.arange(tokens.shape[0])[:, None], positions]
            for layer in self.sequence_head:
                x = layer(x)
            result["logits"] = x
        if stages is not None:
            result["stages"] = stages
        return result
