import os

import torch
import torch.nn as nn
import torch.distributed as dist

from diffulex.attention import Attention
from diffulex.layer.layernorm import RMSNorm
from diffulex.layer.activation import SiluAndMul
from diffulex.layer.rotary_embedding import get_rope
from diffulex.layer.linear import RowParallelLinear, ColumnParallelLinear
from diffulex.layer.embed_head import VocabParallelEmbedding, ParallelLMHead
from diffulex.model.auto_model import AutoModelForDiffusionLM
from diffulex.model.config.sdar.configuration_sdar import SDARConfig


if os.environ.get("TRITON_INTERPRET", None) == "1":
    torch._dynamo.reset()
    torch._dynamo.config.suppress_errors = True
    torch.backends.optimized_mode = False


class SDARAttention(nn.Module):
    """SDAR attention (Diffulex native KV cache path).

    Compatible with Diffulex runner KV cache injection:
    runner sets `self.attn.k_cache` / `self.attn.v_cache` by assigning to modules
    that expose these attributes (see `diffulex/attention/attn_impl.py`).
    """

    def __init__(self, config: SDARConfig) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size

        head_dim = getattr(config, "head_dim", None)
        self.head_dim = head_dim or (config.hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        bias = getattr(config, "attention_bias", False)
        self.q_proj = ColumnParallelLinear(
            config.hidden_size,
            self.total_num_heads * self.head_dim,
            bias=bias,
        )
        self.k_proj = ColumnParallelLinear(
            config.hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=bias,
        )
        self.v_proj = ColumnParallelLinear(
            config.hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=bias,
        )

        # SDAR uses q/k per-head RMSNorm.
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=getattr(config, "rope_theta", 10000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )

        # Diffulex Attention implements KV cache store/load via injected k_cache/v_cache.
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Per-head norm.
        q_by_head = q.view(-1, self.num_heads, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)

        k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)

        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v, mask)
        return self.o_proj(o)


class SDARMLP(nn.Module):
    """SDAR MLP: SiLU(gate) * up -> down."""

    def __init__(self, config: SDARConfig) -> None:
        super().__init__()
        self.gate_proj = ColumnParallelLinear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = ColumnParallelLinear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = RowParallelLinear(config.intermediate_size, config.hidden_size, bias=False)
        assert getattr(config, "hidden_act", "silu") == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x = self.act_fn(torch.cat([gate, up], dim=-1))
        return self.down_proj(x)


class SDARDecoderLayer(nn.Module):
    def __init__(self, config: SDARConfig) -> None:
        super().__init__()
        self.self_attn = SDARAttention(config)
        self.mlp = SDARMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states, mask)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class SDARModel(nn.Module):
    def __init__(self, config: SDARConfig) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([SDARDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual, mask)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


@AutoModelForDiffusionLM.register("sdar")
class SDARForDiffusionLM(nn.Module):
    packed_modules_mapping = {}

    def __init__(self, config: SDARConfig) -> None:
        super().__init__()
        self.model = SDARModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, mask)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)


__all__ = [
    "SDARConfig",
    "SDARAttention",
    "SDARMLP",
    "SDARDecoderLayer",
    "SDARModel",
    "SDARForDiffusionLM",
]
