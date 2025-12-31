import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from diffulex.utils.quantization.context import get_linear_strategy


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LoRAMixin:
    """Mixin class to add LoRA support to existing linear layers."""
    def __init_lora__(self, r: int = 0, lora_alpha: float = 1.0, lora_dropout: float = 0.0):
        if r > 0:
            self.r = r
            self.lora_alpha = lora_alpha
            self.scaling = lora_alpha / r
            
            # Initialize LoRA parameters
            if hasattr(self, 'output_size_per_partition'):
                out_features = self.output_size_per_partition
            else:
                out_features = self.output_size
            
            if hasattr(self, 'input_size_per_partition'):
                in_features = self.input_size_per_partition
            else:
                in_features = self.input_size
            
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
            self.merged = False
            
            # Initialize weights
            nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
            nn.init.zeros_(self.lora_B)
        else:
            self.r = 0
            self.merged = True
    
    def merge_lora(self):
        """Merge LoRA weights into base weight."""
        if not (hasattr(self, 'r') and self.r > 0 and not self.merged):
            return
        # If base weight is missing (e.g., quantized linear removed bf16 weight Parameter),
        # we cannot merge in-place. Keep LoRA unmerged and apply via lora_forward.
        weight = getattr(self, "weight", None)
        if weight is None or not hasattr(weight, "data"):
            return
        self.weight.data += self.scaling * torch.mm(self.lora_B, self.lora_A)
        self.merged = True
    
    def lora_forward(self, x: torch.Tensor, base_output: torch.Tensor) -> torch.Tensor:
        """Apply LoRA forward pass."""
        if not hasattr(self, 'r') or self.r == 0 or self.merged:
            return base_output
        
        lora_out = F.linear(self.lora_dropout(x), self.lora_A.T)
        lora_out = F.linear(lora_out, self.lora_B.T)
        return base_output + lora_out * self.scaling


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_dim: int | None = None,
        quant_kind: str = "other",
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_dim = tp_dim
        self.quant_kind = (quant_kind or "other").strip().lower() or "other"
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        # Quantized weight storage (W8A16 etc.). Empty by default.
        # NOTE: We keep these as buffers so they move with the module and do not appear as Parameters.
        self.register_buffer("quant_weight_int8", torch.empty(0, dtype=torch.int8), persistent=False)
        self.register_buffer("quant_scales", torch.empty(0, dtype=torch.bfloat16), persistent=False)
        self.register_buffer("_weight_is_quantized", torch.tensor(False, dtype=torch.bool), persistent=False)

    def has_quantized_weight(self) -> bool:
        return bool(self._weight_is_quantized.item()) and self.quant_weight_int8.numel() > 0 and self.quant_scales.numel() > 0

    def set_quantized_weight(self, quant_weight_int8: torch.Tensor, quant_scales: torch.Tensor) -> None:
        if quant_weight_int8.dtype != torch.int8:
            raise TypeError(f"quant_weight_int8 must be int8, got {quant_weight_int8.dtype}")
        # Store scales in bf16 by default (good balance for memory/accuracy).
        if quant_scales.dtype != torch.bfloat16:
            quant_scales = quant_scales.to(dtype=torch.bfloat16)
        self.quant_weight_int8 = quant_weight_int8
        self.quant_scales = quant_scales
        self._weight_is_quantized.fill_(True)

    def _maybe_quantize_loaded_weight_param(
        self,
        param: nn.Parameter,
        *,
        loaded_shard_id: object = None,
        expected_shard_ids: set[object] | None = None,
    ) -> None:
        """If current Linear is configured for W8A16, quantize the loaded bf16 weight and drop the bf16 Parameter.

        This is called at the end of weight_loader(), after the shard copy is done.
        """
        # Only process the real weight Parameter (ignore bias).
        current_weight = self._parameters.get("weight", None)
        if current_weight is None or current_weight is not param:
            return

        # Some modules load the same weight parameter in multiple shards (e.g., QKV / merged linears).
        # In that case, we must wait until all shards are loaded before quantizing/removing the bf16 Parameter,
        # otherwise subsequent shard loads would fail (model.get_parameter can't find it).
        if expected_shard_ids is not None:
            if not hasattr(self, "_loaded_weight_shard_ids"):
                self._loaded_weight_shard_ids: set[object] = set()
            self._loaded_weight_shard_ids.add(loaded_shard_id)
            if self._loaded_weight_shard_ids != expected_shard_ids:
                return

        # Get strategy for this kind; default bf16 strategy should not trigger quantization.
        strategy = get_linear_strategy(self.quant_kind)
        if strategy is None:
            return
        if getattr(strategy, "linear_weight_format", None) != "int8":
            return
        if getattr(strategy, "linear_act_format", None) != "bf16":
            return

        # Quantize on the same device as the loaded param (typically CUDA).
        qweight, scales = strategy.quantize_weight_for_kernel(param.data, device=param.data.device)
        self.set_quantized_weight(qweight, scales)

        # Drop bf16 weight Parameter to free GPU memory.
        self._parameters.pop("weight", None)
        # Keep attribute for compatibility, but ensure forward uses quant buffers.
        setattr(self, "weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase, LoRAMixin):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        quant_kind: str = "other",
    ):
        LinearBase.__init__(self, input_size, output_size, None, quant_kind)
        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)
        
        self.__init_lora__(r, lora_alpha, lora_dropout)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)
        self._maybe_quantize_loaded_weight_param(param, loaded_shard_id=None, expected_shard_ids={None})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        strategy = get_linear_strategy(self.quant_kind)
        if self.has_quantized_weight():
            if strategy is None:
                raise RuntimeError("Quantized weight is present but no linear strategy is configured.")
            base_out = strategy.linear_forward(
                x,
                self.quant_weight_int8,
                self.bias,
                quant_kind=self.quant_kind,
                quant_scales=self.quant_scales,
            )
        elif strategy is None:
            base_out = F.linear(x, self.weight, self.bias)
        else:
            base_out = strategy.linear_forward(x, self.weight, self.bias, quant_kind=self.quant_kind)
        return self.lora_forward(x, base_out)


class ColumnParallelLinear(LinearBase, LoRAMixin):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        quant_kind: str = "other",
    ):
        LinearBase.__init__(self, input_size, output_size, 0, quant_kind)
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)

        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, self.input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)
        
        self.__init_lora__(r, lora_alpha, lora_dropout)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)
        self._maybe_quantize_loaded_weight_param(param, loaded_shard_id=None, expected_shard_ids={None})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        strategy = get_linear_strategy(self.quant_kind)
        if self.has_quantized_weight():
            if strategy is None:
                raise RuntimeError("Quantized weight is present but no linear strategy is configured.")
            base_out = strategy.linear_forward(
                x,
                self.quant_weight_int8,
                self.bias,
                quant_kind=self.quant_kind,
                quant_scales=self.quant_scales,
            )
        elif strategy is None:
            base_out = F.linear(x, self.weight, self.bias)
        else:
            base_out = strategy.linear_forward(x, self.weight, self.bias, quant_kind=self.quant_kind)
        return self.lora_forward(x, base_out)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        quant_kind: str = "other",
    ):
        self.output_sizes = output_sizes
        super().__init__(
            input_size,
            sum(output_sizes),
            bias=bias,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            quant_kind=quant_kind,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)
        expected = set(range(len(self.output_sizes)))
        self._maybe_quantize_loaded_weight_param(param, loaded_shard_id=loaded_shard_id, expected_shard_ids=expected)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        quant_kind: str = "attn",
    ):
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads or total_num_heads
        tp_size = dist.get_world_size()
        self.num_heads = divide(self.total_num_heads, tp_size)
        self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
        input_size = hidden_size
        output_size = (self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_size
        super().__init__(input_size, output_size, bias, r, lora_alpha, lora_dropout, quant_kind=quant_kind)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)
        self._maybe_quantize_loaded_weight_param(param, loaded_shard_id=loaded_shard_id, expected_shard_ids={"q", "k", "v"})


class RowParallelLinear(LinearBase, LoRAMixin):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        quant_kind: str = "other",
    ):
        LinearBase.__init__(self, input_size, output_size, 1, quant_kind)
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size

        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size_per_partition))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)
        
        self.__init_lora__(r, lora_alpha, lora_dropout)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)
        self._maybe_quantize_loaded_weight_param(param, loaded_shard_id=None, expected_shard_ids={None})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias if self.tp_rank == 0 else None
        strategy = get_linear_strategy(self.quant_kind)
        if self.has_quantized_weight():
            if strategy is None:
                raise RuntimeError("Quantized weight is present but no linear strategy is configured.")
            y = strategy.linear_forward(
                x,
                self.quant_weight_int8,
                bias,
                quant_kind=self.quant_kind,
                quant_scales=self.quant_scales,
            )
        elif strategy is None:
            y = F.linear(x, self.weight, bias)
        else:
            y = strategy.linear_forward(x, self.weight, bias, quant_kind=self.quant_kind)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return self.lora_forward(x, y)
