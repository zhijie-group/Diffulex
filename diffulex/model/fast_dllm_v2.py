import os
import torch
import torch.nn as nn
import torch.distributed as dist

from diffulex.attention import Attention
from diffulex.layer.layernorm import RMSNorm
from diffulex.layer.activation import SiluAndMul
from diffulex.layer.rotary_embedding import get_rope
from diffulex.model.auto_model import AutoModelForDiffusionLM
from diffulex.layer.linear import RowParallelLinear, ColumnParallelLinear
from diffulex.layer.embed_head import VocabParallelEmbedding, ParallelLMHead
from diffulex.model.config.fast_dllm_v2.configuration_fast_dllm_v2 import FastdLLMV2Config


if os.environ.get("TRITON_INTERPRET", None) == "1":
    torch._dynamo.reset()
    torch._dynamo.config.suppress_errors = True
    torch.backends.optimized_mode = False


class FastdLLMV2RMSNorm(RMSNorm):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__(hidden_size, eps)


class FastdLLMV2Attention(nn.Module):
    """FastdLLM V2 attention mechanism."""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 32768,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-6,
        qkv_bias: bool = True,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.q_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_heads * self.head_dim,
            bias=qkv_bias,
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=qkv_bias,
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            "diffusion_lm",  # Dream uses full attention
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v, mask)
        output = self.o_proj(o)
        return output


class FastdLLMV2MLP(nn.Module):
    """FastdLLM V2 MLP with SiLU activation."""
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x = self.act_fn(torch.cat([gate, up], dim=-1))
        x = self.down_proj(x)
        return x


class FastdLLMV2DecoderLayer(nn.Module):
    """FastdLLM V2 transformer decoder layer."""
    def __init__(
        self,
        config: FastdLLMV2Config,
    ) -> None:
        super().__init__()
        self.self_attn = FastdLLMV2Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=True,  # Dream uses bias in attention
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 10000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = FastdLLMV2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = FastdLLMV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = FastdLLMV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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


class FastdLLMV2Model(nn.Module):
    """FastdLLM V2 model for diffusion language modeling."""
    def __init__(
        self,
        config: FastdLLMV2Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([FastdLLMV2DecoderLayer(config) 
                                     for _ in range(config.num_hidden_layers)])
        self.norm = FastdLLMV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for _, layer in enumerate(self.layers):
            hidden_states, residual = layer(positions, hidden_states, residual, mask)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


@AutoModelForDiffusionLM.register("fast_dllm_v2")
class FastdLLMV2ForDiffusionLM(nn.Module):
    """FastdLLM V2 model for diffusion language modeling with LM head."""
    packed_modules_mapping = {}
    
    def __init__(
        self,
        config: FastdLLMV2Config,
    ) -> None:
        super().__init__()
        self.model = FastdLLMV2Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if getattr(config, 'tie_word_embeddings', False):
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, mask)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits
