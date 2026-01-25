from typing import Optional

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
        # Cache forward output features (avoid per-call inference).
        # Subclasses with TP partitions should overwrite this after partition sizes are known.
        self._forward_out_features: int = int(output_size)
        self.tp_dim = tp_dim
        self.quant_kind = (quant_kind or "other").strip().lower() or "other"
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        # Quantized weight storage (W8A16 etc.). Empty by default.
        # NOTE: We keep these as buffers so they move with the module and do not appear as Parameters.
        self.register_buffer("quant_weight_int8", torch.empty(0, dtype=torch.int8), persistent=False)
        self.register_buffer("quant_scales", torch.empty(0, dtype=torch.bfloat16), persistent=False)
        # Cache a 1xN view of scales to avoid per-call view/shape handling on hot paths.
        self.register_buffer("quant_scales_1xn", torch.empty(0, dtype=torch.bfloat16), persistent=False)
        self.register_buffer("_weight_is_quantized", torch.tensor(False, dtype=torch.bool), persistent=False)
        
        # GPTQ/AWQ offline quantized weight storage (W4A16).
        # NOTE(vLLM-format):
        # - GPTQ: qweight int32 [K/pack, N], qzeros int32 [K/group, N/pack],
        #         scales fp16 [K/group, N], g_idx optional (usually empty when desc_act=False)
        # - AWQ : qweight int32 [K, N/pack], qzeros int32 [K/group, N/pack],
        #         scales fp16 [K/group, N]
        #
        # Where pack = 32 / bits (bits=4 => pack=8), K=in_features, N=out_features.
        self.register_buffer("gptq_qweight", torch.empty(0, dtype=torch.int32), persistent=False)
        self.register_buffer("gptq_qzeros", torch.empty(0, dtype=torch.int32), persistent=False)
        self.register_buffer("gptq_scales", torch.empty(0, dtype=torch.float16), persistent=False)
        self.register_buffer("gptq_g_idx", torch.empty(0, dtype=torch.int32), persistent=False)
        self.register_buffer("awq_qweight", torch.empty(0, dtype=torch.int32), persistent=False)
        self.register_buffer("awq_qzeros", torch.empty(0, dtype=torch.int32), persistent=False)
        self.register_buffer("awq_scales", torch.empty(0, dtype=torch.float16), persistent=False)
        # Metadata for offline quantized weights
        self.register_buffer("_offline_quant_format", torch.empty(0, dtype=torch.int8), persistent=False)  # 0=none, 1=gptq, 2=awq
        # Bits for offline GPTQ/AWQ weights (needed for marlin-exported layouts where
        # we cannot infer bits from packed tensor shapes).
        self.register_buffer("_offline_quant_bits", torch.tensor(0, dtype=torch.int32), persistent=False)
        self.register_buffer("_offline_quant_group_size", torch.tensor(128, dtype=torch.int32), persistent=False)
        self.register_buffer("_offline_quant_out_features", torch.tensor(0, dtype=torch.int32), persistent=False)
        self.register_buffer("_offline_quant_in_features", torch.tensor(0, dtype=torch.int32), persistent=False)
        # GPTQ runtime prep state (vLLM requires gptq_shuffle before first gemm).
        self.register_buffer("_gptq_is_shuffled", torch.tensor(False, dtype=torch.bool), persistent=False)

        # ---- vLLM Marlin variants (GPTQ/AWQ) one-time repack cache ----
        # These buffers are populated lazily when a *_marlin strategy is selected.
        self.register_buffer("_gptq_marlin_is_prepared", torch.tensor(False, dtype=torch.bool), persistent=False)
        self.register_buffer("gptq_marlin_qweight", torch.empty(0, dtype=torch.int32), persistent=False)
        self.register_buffer("gptq_marlin_scales", torch.empty(0, dtype=torch.float16), persistent=False)
        self.register_buffer("gptq_marlin_zp", torch.empty(0, dtype=torch.int32), persistent=False)
        self.register_buffer("gptq_marlin_g_idx", torch.empty(0, dtype=torch.int32), persistent=False)
        self.register_buffer("gptq_marlin_g_idx_sort_indices", torch.empty(0, dtype=torch.int32), persistent=False)
        self.register_buffer("gptq_marlin_workspace", torch.empty(0, dtype=torch.int32), persistent=False)
        self.register_buffer("_awq_marlin_is_prepared", torch.tensor(False, dtype=torch.bool), persistent=False)
        self.register_buffer("awq_marlin_qweight", torch.empty(0, dtype=torch.int32), persistent=False)
        self.register_buffer("awq_marlin_scales", torch.empty(0, dtype=torch.float16), persistent=False)
        self.register_buffer("awq_marlin_zp", torch.empty(0, dtype=torch.int32), persistent=False)
        self.register_buffer("awq_marlin_workspace", torch.empty(0, dtype=torch.int32), persistent=False)

        # ---- Python-side meta cache (CUDA Graph friendly) ----
        # Avoid `.item()` on CUDA tensors in hot paths (it introduces GPU->CPU sync and breaks graph capture).
        self._weight_is_quantized_py: bool = False
        # 0=none, 1=gptq, 2=awq
        self._offline_quant_format_py: int = 0
        self._offline_quant_bits_py: int = 0
        self._offline_quant_group_size_py: int = 128
        self._offline_quant_out_features_py: int = 0
        self._offline_quant_in_features_py: int = 0
        self._gptq_is_shuffled_py: bool = False
        self._gptq_marlin_is_prepared_py: bool = False
        self._awq_marlin_is_prepared_py: bool = False

    def has_quantized_weight(self) -> bool:
        return self._weight_is_quantized_py and self.quant_weight_int8.numel() > 0 and self.quant_scales.numel() > 0

    def has_offline_quantized_weight(self) -> bool:
        """Check if offline quantized weights (GPTQ/AWQ) are present."""
        if self._offline_quant_format_py == 1:  # GPTQ
            return (
                self.gptq_qweight.numel() > 0
                and self.gptq_qzeros.numel() > 0
                and self.gptq_scales.numel() > 0
            )
        elif self._offline_quant_format_py == 2:  # AWQ
            return (
                self.awq_qweight.numel() > 0
                and self.awq_qzeros.numel() > 0
                and self.awq_scales.numel() > 0
            )
        return False

    def set_offline_quantized_weight(
        self,
        format: str,
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        *,
        out_features: int,
        in_features: int,
        group_size: int = 128,
        g_idx: Optional[torch.Tensor] = None,
    ) -> None:
        """Set offline quantized weights (GPTQ or AWQ format).

        Args:
            format: "gptq" or "awq"
            qweight/qzeros/scales: vLLM standard tensors (see notes above).
            out_features: Output features (N)
            in_features: Input features (K)
            group_size: Group size for quantization (default: 128)
            g_idx: Optional int32 tensor [in_features] for act-order (GPTQ only; usually empty)
        """
        # NOTE: Offline quantized weights are typically loaded from safetensors on CPU.
        # In Diffulex, the engine may move modules to CUDA before calling this method,
        # so we must ensure tensors are moved to the module device here.
        def _infer_module_device() -> torch.device:
            w = getattr(self, "weight", None)
            if isinstance(w, torch.Tensor):
                return w.device
            for p in self.parameters(recurse=False):
                return p.device
            for b in self.buffers(recurse=False):
                return b.device
            return torch.device("cpu")

        module_device = _infer_module_device()

        format = format.strip().lower()
        if format not in ("gptq", "awq"):
            raise ValueError(f"Unsupported offline quant format: {format}. Supported: 'gptq', 'awq'")

        # Infer bits/pack_factor from packed tensor shapes to support GPTQ W2/W4/W8.
        # vLLM packing convention:
        # - GPTQ: qweight [K/pack, N], qzeros [K/group, N/pack]
        # -  AWQ: qweight [K,      N/pack], qzeros [K/group, N/pack]
        # where pack = 32 / bits and bits must divide 32.
        if format == "gptq":
            if int(qweight.shape[0]) <= 0 or in_features % int(qweight.shape[0]) != 0:
                raise ValueError(
                    "Cannot infer GPTQ pack_factor from qweight shape: "
                    f"in_features={in_features}, qweight.shape={tuple(qweight.shape)}"
                )
            pack_factor = in_features // int(qweight.shape[0])
        else:  # awq
            if int(qweight.shape[1]) <= 0 or out_features % int(qweight.shape[1]) != 0:
                raise ValueError(
                    "Cannot infer AWQ pack_factor from qweight shape: "
                    f"out_features={out_features}, qweight.shape={tuple(qweight.shape)}"
                )
            pack_factor = out_features // int(qweight.shape[1])
        if 32 % pack_factor != 0:
            raise ValueError(
                f"Unsupported pack_factor={pack_factor} (requires 32%pack_factor==0) "
                f"for offline format={format}. "
                f"in_features={in_features}, out_features={out_features}, "
                f"qweight.shape={tuple(qweight.shape)}, qzeros.shape={tuple(qzeros.shape)}, scales.shape={tuple(scales.shape)}"
            )
        bits = 32 // pack_factor
        if format == "awq" and bits != 4:
            raise ValueError(f"AWQ 目前仅支持 4-bit（pack_factor=8），当前推断 bits={bits} (pack_factor={pack_factor})")
        # Cache meta as Python primitives (graph-friendly).
        self._offline_quant_bits_py = int(bits)
        # Record bits for downstream kernels (esp. marlin path).
        self._offline_quant_bits = torch.tensor(bits, dtype=torch.int32, device=module_device)

        if qweight.dtype != torch.int32:
            raise TypeError(f"qweight must be int32 (vLLM format), got {qweight.dtype}")
        if qzeros.dtype != torch.int32:
            raise TypeError(f"qzeros must be int32 (vLLM format), got {qzeros.dtype}")
        if scales.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(
                f"scales must be float16/bfloat16/float32 (vLLM format), got {scales.dtype}"
            )
        if scales.dtype != torch.float16:
            scales = scales.to(dtype=torch.float16)

        # Move to module device before validation/assignment.
        if qweight.device != module_device:
            qweight = qweight.to(device=module_device)
        if qzeros.device != module_device:
            qzeros = qzeros.to(device=module_device)
        if scales.device != module_device:
            scales = scales.to(device=module_device)
        if g_idx is not None and g_idx.device != module_device:
            g_idx = g_idx.to(device=module_device)

        # Make packed tensors contiguous once at load-time (avoid per-call checks/copies).
        qweight = qweight.contiguous()
        qzeros = qzeros.contiguous()
        scales = scales.contiguous()
        if g_idx is not None:
            g_idx = g_idx.contiguous()

        # group_size == -1 means channelwise in some ecosystems; vLLM normalizes -1 to K.
        group_size_norm = in_features if group_size == -1 else group_size
        if group_size_norm <= 0 or (in_features % group_size_norm != 0):
            raise ValueError(
                f"Invalid group_size={group_size} for in_features={in_features}. "
                "Expected group_size == -1 or a positive divisor of in_features."
            )
        num_groups = in_features // group_size_norm

        if format == "gptq":
            expected_qweight_shape = (in_features // pack_factor, out_features)
            expected_qzeros_shape = (num_groups, out_features // pack_factor)
            expected_scales_shape = (num_groups, out_features)
        else:  # awq
            expected_qweight_shape = (in_features, out_features // pack_factor)
            expected_qzeros_shape = (num_groups, out_features // pack_factor)
            expected_scales_shape = (num_groups, out_features)

        if qweight.shape != expected_qweight_shape:
            raise ValueError(
                f"qweight shape mismatch: got {tuple(qweight.shape)}, expected {expected_qweight_shape}"
            )
        if qzeros.shape != expected_qzeros_shape:
            raise ValueError(
                f"qzeros shape mismatch: got {tuple(qzeros.shape)}, expected {expected_qzeros_shape}"
            )
        if scales.shape != expected_scales_shape:
            raise ValueError(
                f"scales shape mismatch: got {tuple(scales.shape)}, expected {expected_scales_shape}"
            )

        if format == "gptq":
            self.gptq_qweight = qweight
            self.gptq_qzeros = qzeros
            self.gptq_scales = scales
            if g_idx is not None and getattr(g_idx, "numel", lambda: 1)() == 0:
                g_idx = None
            if g_idx is not None:
                if g_idx.shape != (in_features,):
                    raise ValueError(
                        f"g_idx shape mismatch: got {g_idx.shape}, expected ({in_features},)"
                    )
                if g_idx.dtype != torch.int32:
                    g_idx = g_idx.to(dtype=torch.int32)
                self.gptq_g_idx = g_idx
            else:
                # Clear g_idx if not provided
                self.gptq_g_idx = torch.empty(0, dtype=torch.int32, device=module_device)
            self._offline_quant_format = torch.tensor(1, dtype=torch.int8, device=module_device)
            self._gptq_is_shuffled = torch.tensor(False, dtype=torch.bool, device=module_device)
            self._offline_quant_format_py = 1
            self._gptq_is_shuffled_py = False
        else:  # AWQ
            self.awq_qweight = qweight
            self.awq_qzeros = qzeros
            self.awq_scales = scales
            # AWQ doesn't use g_idx, clear it
            self.gptq_qweight = torch.empty(0, dtype=torch.int32, device=module_device)
            self.gptq_qzeros = torch.empty(0, dtype=torch.int32, device=module_device)
            self.gptq_scales = torch.empty(0, dtype=torch.float16, device=module_device)
            self.gptq_g_idx = torch.empty(0, dtype=torch.int32, device=module_device)
            self._offline_quant_format = torch.tensor(2, dtype=torch.int8, device=module_device)
            self._gptq_is_shuffled = torch.tensor(False, dtype=torch.bool, device=module_device)
            self._offline_quant_format_py = 2
            self._gptq_is_shuffled_py = False

        # Reset marlin-prep caches (weights may have changed / moved).
        self._gptq_marlin_is_prepared = torch.tensor(False, dtype=torch.bool, device=module_device)
        self.gptq_marlin_qweight = torch.empty(0, dtype=torch.int32, device=module_device)
        self.gptq_marlin_scales = torch.empty(0, dtype=torch.float16, device=module_device)
        self.gptq_marlin_zp = torch.empty(0, dtype=torch.int32, device=module_device)
        self.gptq_marlin_g_idx = torch.empty(0, dtype=torch.int32, device=module_device)
        self.gptq_marlin_g_idx_sort_indices = torch.empty(0, dtype=torch.int32, device=module_device)
        self.gptq_marlin_workspace = torch.empty(0, dtype=torch.int32, device=module_device)
        self._awq_marlin_is_prepared = torch.tensor(False, dtype=torch.bool, device=module_device)
        self.awq_marlin_qweight = torch.empty(0, dtype=torch.int32, device=module_device)
        self.awq_marlin_scales = torch.empty(0, dtype=torch.float16, device=module_device)
        self.awq_marlin_zp = torch.empty(0, dtype=torch.int32, device=module_device)
        self.awq_marlin_workspace = torch.empty(0, dtype=torch.int32, device=module_device)

        self._offline_quant_group_size = torch.tensor(group_size, dtype=torch.int32, device=module_device)
        self._offline_quant_out_features = torch.tensor(out_features, dtype=torch.int32, device=module_device)
        self._offline_quant_in_features = torch.tensor(in_features, dtype=torch.int32, device=module_device)
        # Python meta mirrors.
        self._offline_quant_group_size_py = int(group_size)
        self._offline_quant_out_features_py = int(out_features)
        self._offline_quant_in_features_py = int(in_features)
        self._gptq_marlin_is_prepared_py = False
        self._awq_marlin_is_prepared_py = False

        # Drop bf16 weight Parameter if present (to free memory)
        if "weight" in self._parameters:
            self._parameters.pop("weight", None)
            setattr(self, "weight", None)

    def _maybe_prepare_offline_gptq(self, x: torch.Tensor) -> None:
        """Prepare vLLM GPTQ weights on first use (required gptq_shuffle)."""
        if self._offline_quant_format_py != 1:
            return
        if self.gptq_qweight.numel() == 0:
            return
        if self._gptq_is_shuffled_py:
            return

        # Lazy import to avoid pulling vLLM unless GPTQ offline weights are used.
        try:
            from vllm import _custom_ops as ops  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "GPTQ offline 权重已加载，但无法导入 vLLM CUDA custom ops（vllm._custom_ops）。"
            ) from e

        # vLLM uses torch.int for g_idx (can be empty when desc_act=False).
        if self.gptq_g_idx.numel() == 0:
            g_idx = torch.empty((0,), device=x.device, dtype=torch.int)
        else:
            g_idx = self.gptq_g_idx.to(device=x.device, dtype=torch.int)

        if self.gptq_qweight.device != x.device:
            raise RuntimeError(
                f"GPTQ qweight device mismatch: qweight on {self.gptq_qweight.device}, x on {x.device}. "
                "请确保模型与输入在同一设备。"
            )

        # Infer weight_bits from packed qweight shape to support GPTQ W2/W4/W8.
        # qweight: [K/pack_factor, N], where pack_factor = 32 / weight_bits.
        in_features = int(self._offline_quant_in_features_py)
        if in_features is None or in_features <= 0:
            raise RuntimeError("GPTQ offline 权重已加载，但无法推断 in_features 以计算 weight_bits。")
        if self.gptq_qweight.shape[0] <= 0 or in_features % int(self.gptq_qweight.shape[0]) != 0:
            raise RuntimeError(
                f"GPTQ qweight shape 不合法，无法推断 weight_bits: "
                f"in_features={in_features}, qweight.shape={tuple(self.gptq_qweight.shape)}"
            )
        pack_factor = in_features // int(self.gptq_qweight.shape[0])
        if 32 % pack_factor != 0:
            raise RuntimeError(
                f"GPTQ pack_factor={pack_factor} 不支持（需要 32 % pack_factor == 0），"
                f"in_features={in_features}, qweight.shape={tuple(self.gptq_qweight.shape)}"
            )
        weight_bits = 32 // pack_factor
        ops.gptq_shuffle(self.gptq_qweight, g_idx, weight_bits)
        # Do NOT create new tensors on hot paths; update in-place + python mirror.
        self._gptq_is_shuffled.fill_(True)
        self._gptq_is_shuffled_py = True

    def _maybe_prepare_offline_gptq_marlin(self, x: torch.Tensor) -> None:
        """Prepare vLLM GPTQ Marlin weights on first use (repack + permute scales/zp).

        IMPORTANT: This path must NOT call `gptq_shuffle` (that is specific to gptq_gemm/exllama).
        """
        if self._offline_quant_format_py != 1:
            return
        if self.gptq_qweight.numel() == 0:
            return
        if self._gptq_marlin_is_prepared_py:
            return

        try:
            from vllm import _custom_ops as ops  # type: ignore
            from vllm.model_executor.layers.quantization.utils.marlin_utils import (  # type: ignore
                marlin_make_empty_g_idx,
                marlin_make_workspace_new,
                marlin_permute_scales,
                marlin_sort_g_idx,
            )
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "GPTQ Marlin 需要 vLLM CUDA custom ops + marlin_utils，但当前环境不可用。"
            ) from e

        device = x.device
        if self.gptq_qweight.device != device:
            raise RuntimeError(
                f"GPTQ qweight device mismatch: qweight on {self.gptq_qweight.device}, x on {device}. "
                "请确保模型与输入在同一设备。"
            )

        in_features = int(self._offline_quant_in_features_py)
        out_features = int(self._offline_quant_out_features_py)
        group_size = int(self._offline_quant_group_size_py)
        if in_features <= 0 or out_features <= 0:
            raise RuntimeError(
                f"GPTQ Marlin: invalid feature sizes: in_features={in_features}, out_features={out_features}"
            )

        # Determine weight_bits.
        # - Standard GPTQ layout: infer from qweight K packing.
        # - Marlin-exported layout: bits cannot be inferred from qweight shape; use recorded bits.
        weight_bits = int(self._offline_quant_bits_py)
        if weight_bits <= 0:
            if self.gptq_qweight.shape[0] <= 0 or in_features % int(self.gptq_qweight.shape[0]) != 0:
                raise RuntimeError(
                    "GPTQ Marlin: cannot infer pack_factor from qweight shape: "
                    f"in_features={in_features}, qweight.shape={tuple(self.gptq_qweight.shape)}"
                )
            pack_factor = in_features // int(self.gptq_qweight.shape[0])
            if 32 % pack_factor != 0:
                raise RuntimeError(
                    f"GPTQ Marlin: unsupported pack_factor={pack_factor} (requires 32%pack_factor==0)"
                )
            weight_bits = 32 // pack_factor
        if weight_bits not in (4, 8):
            raise RuntimeError(
                f"GPTQ Marlin: only 4/8-bit are supported in this integration, got bits={weight_bits}"
            )

        # If loader already provided marlin-ready weights/scales (exported offline),
        # skip repack/permute but still create workspace / g_idx metadata.
        already_marlin_ready = (
            self.gptq_marlin_qweight.numel() > 0
            and self.gptq_marlin_scales.numel() > 0
        )
        if already_marlin_ready:
            if self.gptq_marlin_qweight.device != device or self.gptq_marlin_scales.device != device:
                raise RuntimeError(
                    "GPTQ Marlin: prepacked marlin tensors device mismatch: "
                    f"qweight on {self.gptq_marlin_qweight.device}, scales on {self.gptq_marlin_scales.device}, x on {device}."
                )

        # g_idx (act-order) handling: marlin expects sorted g_idx + sort indices; otherwise empty.
        if self.gptq_g_idx.numel() > 0:
            g_idx_sorted, g_idx_sort_indices = marlin_sort_g_idx(self.gptq_g_idx.to(device=device, dtype=torch.int32))
            self.gptq_marlin_g_idx = g_idx_sorted.contiguous()
            self.gptq_marlin_g_idx_sort_indices = g_idx_sort_indices.contiguous()
        else:
            self.gptq_marlin_g_idx = marlin_make_empty_g_idx(device)
            self.gptq_marlin_g_idx_sort_indices = marlin_make_empty_g_idx(device)

        # Workspace (internal locking mechanism).
        self.gptq_marlin_workspace = marlin_make_workspace_new(device)

        if not already_marlin_ready:
            # Repack qweight to marlin format.
            self.gptq_marlin_qweight = ops.gptq_marlin_repack(
                self.gptq_qweight.contiguous(),
                perm=self.gptq_marlin_g_idx_sort_indices,
                size_k=in_features,
                size_n=out_features,
                num_bits=weight_bits,
                is_a_8bit=False,
            ).contiguous()

            # Permute scales to marlin format.
            self.gptq_marlin_scales = marlin_permute_scales(
                self.gptq_scales.contiguous(),
                size_k=in_features,
                size_n=out_features,
                group_size=group_size,
                is_a_8bit=False,
            ).contiguous()

        # GPTQ Marlin only supports symmetric weights (no runtime zero-points).
        # Use empty zp to keep has_zp=False in the kernel.
        self.gptq_marlin_zp = marlin_make_empty_g_idx(device)

        self._gptq_marlin_is_prepared.fill_(True)
        self._gptq_marlin_is_prepared_py = True

    def _maybe_prepare_offline_awq_marlin(self, x: torch.Tensor) -> None:
        """Prepare vLLM AWQ Marlin weights on first use (repack + permute scales/zp)."""
        if self._offline_quant_format_py != 2:
            return
        if self.awq_qweight.numel() == 0:
            return
        if self._awq_marlin_is_prepared_py:
            return

        try:
            from vllm import _custom_ops as ops  # type: ignore
            from vllm.model_executor.layers.quantization.utils.marlin_utils import (  # type: ignore
                awq_to_marlin_zero_points,
                marlin_make_workspace_new,
                marlin_permute_scales,
            )
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "AWQ Marlin 需要 vLLM CUDA custom ops + marlin_utils，但当前环境不可用。"
            ) from e

        device = x.device
        if self.awq_qweight.device != device:
            raise RuntimeError(
                f"AWQ qweight device mismatch: qweight on {self.awq_qweight.device}, x on {device}. "
                "请确保模型与输入在同一设备。"
            )

        in_features = int(self._offline_quant_in_features_py)
        out_features = int(self._offline_quant_out_features_py)
        group_size = int(self._offline_quant_group_size_py)
        if in_features <= 0 or out_features <= 0:
            raise RuntimeError(
                f"AWQ Marlin: invalid feature sizes: in_features={in_features}, out_features={out_features}"
            )

        # AWQ is 4-bit only.
        pack_factor = out_features // int(self.awq_qweight.shape[1])
        if pack_factor != 8:
            raise RuntimeError(f"AWQ Marlin: expected pack_factor=8 (W4), got pack_factor={pack_factor}")
        weight_bits = 4
        num_groups = (in_features // (in_features if group_size == -1 else group_size))

        self.awq_marlin_workspace = marlin_make_workspace_new(device)

        # Repack qweight to marlin format.
        self.awq_marlin_qweight = ops.awq_marlin_repack(
            self.awq_qweight.contiguous(),
            size_k=in_features,
            size_n=out_features,
            num_bits=weight_bits,
            is_a_8bit=False,
        ).contiguous()

        # Permute scales to marlin format.
        self.awq_marlin_scales = marlin_permute_scales(
            self.awq_scales.contiguous(),
            size_k=in_features,
            size_n=out_features,
            group_size=group_size,
            is_a_8bit=False,
        ).contiguous()

        # Convert zero-points to marlin format.
        self.awq_marlin_zp = awq_to_marlin_zero_points(
            self.awq_qzeros.contiguous(),
            size_k=num_groups,
            size_n=out_features,
            num_bits=weight_bits,
            is_a_8bit=False,
        ).contiguous()

        self._awq_marlin_is_prepared.fill_(True)
        self._awq_marlin_is_prepared_py = True

    def set_quantized_weight(self, quant_weight_int8: torch.Tensor, quant_scales: torch.Tensor) -> None:
        # Support:
        # - int8: int8/int4 weight-only quantization
        # - float8: FP8 weight-only quantization (vLLM-aligned)
        # - uint8: legacy FP8 storage (kept for backward compatibility)
        fp8_dtypes: tuple[torch.dtype, ...] = tuple(
            d
            for d in (
                getattr(torch, "float8_e4m3fn", None),
                getattr(torch, "float8_e4m3fnuz", None),
                getattr(torch, "float8_e5m2", None),
                getattr(torch, "float8_e5m2fnuz", None),
            )
            if d is not None
        )
        if quant_weight_int8.dtype not in (torch.int8, torch.uint8, *fp8_dtypes):
            raise TypeError(
                f"quant_weight_int8 must be int8/uint8/float8, got {quant_weight_int8.dtype}"
            )
        # Store scales dtype depends on strategy:
        # - W8A16/W4A16 kernels currently take bf16 scales.
        # - W8A8/W4A8 paths are more sensitive to scale precision; keep scales at fp16.
        # - FP8 W8A16 uses float32 scales.
        # - FP8 W8A8 uses float16 scales.
        try:
            strategy = get_linear_strategy(self.quant_kind)
        except Exception:
            strategy = None
        scale_dtype = torch.bfloat16
        force_weight_contig = True
        if strategy is not None:
            weight_format = getattr(strategy, "linear_weight_format", None)
            act_format = getattr(strategy, "linear_act_format", None)
            # FP8 W8A16 uses float32 scales
            if weight_format in ("fp8_e4m3", "fp8_e5m2") and act_format == "bf16":
                scale_dtype = torch.float32
                # Keep KxN transpose-view layout (do NOT force contiguous) for vLLM FP8 kernels.
                force_weight_contig = False
            # W8A8 int8 uses float32 [1, N] weight scales in vLLM cutlass_scaled_mm path.
            elif weight_format == "int8" and act_format == "int8":
                scale_dtype = torch.float32
                # vLLM CUTLASS scaled_mm expects int8 weight in KxN with stride(0)==1,
                # which is typically produced as a transpose-view (non-contiguous).
                # Do NOT force contiguous here; just avoid per-call conversions.
                force_weight_contig = False
            # FP8 W8A8 keeps float32 scales; also keep KxN transpose-view layout.
            elif act_format in ("fp8_e4m3", "fp8_e5m2"):
                scale_dtype = torch.float32
                force_weight_contig = False
            # Other int8/int4 mixed paths use float16 scales by default.
            elif act_format == "int8":
                scale_dtype = torch.float16
        if quant_scales.dtype != scale_dtype:
            quant_scales = quant_scales.to(dtype=scale_dtype)
        # Make sure scales are contiguous once at load-time.
        # NOTE: Some kernels require specific non-contiguous weight layouts (e.g., W8A8 KxN with stride(0)==1).
        # We avoid per-call `is_contiguous/contiguous` checks while preserving required layouts.
        if force_weight_contig:
            quant_weight_int8 = quant_weight_int8.contiguous()
        quant_scales = quant_scales.contiguous()
        self.quant_weight_int8 = quant_weight_int8
        self.quant_scales = quant_scales
        # 1xN view for fused kernels expecting 2D scales.
        self.quant_scales_1xn = quant_scales if quant_scales.dim() == 2 else quant_scales.view(1, -1)
        self._weight_is_quantized.fill_(True)
        self._weight_is_quantized_py = True

    def _maybe_promote_weight_to_quantized_at_runtime(
        self,
        x: torch.Tensor,
        strategy,
        *,
        expected_weight_formats: tuple[str, ...] = ("int8", "int4", "fp8_e4m3", "fp8_e5m2"),
    ) -> None:
        """Runtime safety net: if a Linear is configured for quantization but the bf16/fp16
        weight Parameter was not quantized+removed at load-time (e.g., due to sharded load
        ordering), quantize once on first forward and drop the bf16 weight Parameter.

        This avoids keeping both bf16 weights and quantized weights resident on GPU.
        """
        if strategy is None:
            return
        if self.has_offline_quantized_weight() or self.has_quantized_weight():
            return
        weight_param = self._parameters.get("weight", None)
        if weight_param is None:
            return
        weight_format = getattr(strategy, "linear_weight_format", None)
        if weight_format not in expected_weight_formats:
            return
        if getattr(strategy, "name", "").startswith("linear_stub"):
            return
        w = getattr(self, "weight", None)
        if w is None or getattr(w, "dtype", None) not in (torch.bfloat16, torch.float16):
            return
        try:
            qweight, scales = strategy.quantize_weight_for_kernel(w.data, device=w.data.device)
        except Exception:
            return
        self.set_quantized_weight(qweight, scales)
        # Drop bf16 weight Parameter to free GPU memory.
        self._parameters.pop("weight", None)
        setattr(self, "weight", None)

    def _maybe_quantize_loaded_weight_param(
        self,
        param: nn.Parameter,
        *,
        loaded_shard_id: object = None,
        expected_shard_ids: set[object] | None = None,
    ) -> None:
        """If current Linear is configured for quantization, quantize the loaded bf16 weight and drop the bf16 Parameter.

        This is called at the end of weight_loader(), after the shard copy is done.
        Supports int8 (W8A16/W8A8), int4 (W4A16/W4A8), and FP8 (FP8 W8A16/FP8 W8A8) quantization.
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
        weight_format = getattr(strategy, "linear_weight_format", None)
        # NOTE: We intentionally do NOT require act_format == "bf16" here.
        # For W8A8/W4A8/FP8 W8A8 we still want to quantize+drop the bf16 weight Parameter at load-time.
        # But we must avoid doing this for the generic stub strategy (unsupported combos),
        # otherwise we'd drop weights and then raise NotImplementedError at runtime.
        if getattr(strategy, "name", "").startswith("linear_stub"):
            return
        
        # Support int8/int4/FP8 weight formats (W8A16/W8A8, W4A16/W4A8, FP8 W8A16/FP8 W8A8).
        if weight_format not in ("int8", "int4", "fp8_e4m3", "fp8_e5m2"):
            return

        # Quantize on the same device as the loaded param (typically CUDA).
        qweight, scales = strategy.quantize_weight_for_kernel(param.data, device=param.data.device)
        self.set_quantized_weight(qweight, scales)

        # Drop bf16 weight Parameter to free GPU memory.
        self._parameters.pop("weight", None)
        # Keep attribute for compatibility, but ensure forward uses quant buffers.
        setattr(self, "weight", None)

    def _get_linear_strategy(self):
        """Return strategy for current `quant_kind` (or None).

        NOTE: do not swallow TypeError here; a wrong strategy type should fail fast.
        """
        return get_linear_strategy(self.quant_kind)

    def _offline_meta(self) -> tuple[int, int, int]:
        """Return (out_features, in_features, group_size) for offline GPTQ/AWQ."""
        return (
            int(self._offline_quant_out_features_py),
            int(self._offline_quant_in_features_py),
            int(self._offline_quant_group_size_py),
        )

    def _infer_gptq_weight_bits(self, *, in_features: int) -> int:
        """Infer/return GPTQ weight bits for downstream kernels.

        Priority:
        - use recorded bits (e.g., marlin-exported layouts),
        - otherwise infer from qweight packing.
        """
        bits = int(self._offline_quant_bits_py)
        if bits > 0:
            return bits
        if self.gptq_qweight.numel() == 0:
            raise RuntimeError("GPTQ bits 推断失败：gptq_qweight 为空。")
        if self.gptq_qweight.shape[0] <= 0 or in_features % int(self.gptq_qweight.shape[0]) != 0:
            raise RuntimeError(
                f"GPTQ bits 推断失败：in_features={in_features}, qweight.shape={tuple(self.gptq_qweight.shape)}"
            )
        pack_factor = in_features // int(self.gptq_qweight.shape[0])
        if 32 % pack_factor != 0:
            raise RuntimeError(f"GPTQ bits 推断失败：pack_factor={pack_factor} 不满足 32%pack_factor==0")
        return 32 // pack_factor

    def _maybe_int4_original_in_features_kwargs(self, strategy, x: torch.Tensor) -> Optional[dict]:
        """Some int4 kernels need original K (before packing)."""
        if strategy is None:
            return None
        if getattr(strategy, "linear_weight_format", None) == "int4":
            return {"original_in_features": x.shape[1]}
        return None

    def _build_offline_forward_kwargs(self, x: torch.Tensor, strategy) -> dict:
        """Build kwargs for offline GPTQ/AWQ (including Marlin variants)."""
        if strategy is None:
            raise RuntimeError("Offline quantized weight is present but no linear strategy is configured.")

        format_val = int(self._offline_quant_format_py)
        weight_format = getattr(strategy, "linear_weight_format", None)
        out_features, in_features, group_size = self._offline_meta()

        meta = {
            "out_features": out_features,
            "in_features": in_features,
            "group_size": group_size,
        }

        if format_val == 1:  # GPTQ
            # IMPORTANT: only gptq_gemm needs gptq_shuffle; marlin variants require the original format.
            if weight_format == "gptq":
                self._maybe_prepare_offline_gptq(x)
                return {
                    **meta,
                    "gptq_qweight": self.gptq_qweight,
                    "gptq_qzeros": self.gptq_qzeros,
                    "gptq_scales": self.gptq_scales,
                    "gptq_group_size": group_size,
                    # Always pass g_idx (can be empty). vLLM expects it for GPTQ kernels.
                    "gptq_g_idx": self.gptq_g_idx,
                }

            if weight_format == "gptq_marlin":
                self._maybe_prepare_offline_gptq_marlin(x)
                bits = self._infer_gptq_weight_bits(in_features=in_features)
                return {
                    **meta,
                    "gptq_weight_bits": bits,
                    "gptq_marlin_qweight": self.gptq_marlin_qweight,
                    "gptq_marlin_scales": self.gptq_marlin_scales,
                    "gptq_marlin_zp": self.gptq_marlin_zp,
                    "gptq_marlin_g_idx": self.gptq_marlin_g_idx,
                    "gptq_marlin_g_idx_sort_indices": self.gptq_marlin_g_idx_sort_indices,
                    "gptq_marlin_workspace": self.gptq_marlin_workspace,
                }

            raise RuntimeError(
                f"Offline GPTQ weights are present, but current strategy weight_format={weight_format!r} is not compatible."
            )

        if format_val == 2:  # AWQ
            if weight_format == "awq":
                return {
                    **meta,
                    "awq_qweight": self.awq_qweight,
                    "awq_qzeros": self.awq_qzeros,
                    "awq_scales": self.awq_scales,
                    "awq_group_size": group_size,
                }

            if weight_format == "awq_marlin":
                self._maybe_prepare_offline_awq_marlin(x)
                return {
                    **meta,
                    "awq_marlin_qweight": self.awq_marlin_qweight,
                    "awq_marlin_scales": self.awq_marlin_scales,
                    "awq_marlin_zp": self.awq_marlin_zp,
                    "awq_marlin_workspace": self.awq_marlin_workspace,
                    "awq_weight_bits": 4,
                }

            raise RuntimeError(
                f"Offline AWQ weights are present, but current strategy weight_format={weight_format!r} is not compatible."
            )

        raise RuntimeError(f"Unknown offline quant format: {format_val}")

    def _forward_base(self, x: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
        """Unified forward dispatcher for bf16 / online quant / offline GPTQ/AWQ."""
        strategy = self._get_linear_strategy()
        # Runtime safety net: ensure we don't keep bf16+quant weights both resident.
        self._maybe_promote_weight_to_quantized_at_runtime(x, strategy)

        # Offline quantized weights (GPTQ/AWQ) have higher priority.
        if self.has_offline_quantized_weight():
            if strategy is None:
                raise RuntimeError("Offline quantized weight is present but no linear strategy is configured.")
            weight_format = getattr(strategy, "linear_weight_format", None)
            out_features, in_features, group_size = self._offline_meta()

            # Avoid per-call kwargs dict construction on hot paths.
            if weight_format == "gptq":
                self._maybe_prepare_offline_gptq(x)
                bits = self._infer_gptq_weight_bits(in_features=in_features)
                return strategy.linear_forward(
                    x,
                    None,  # weight not used for offline quantized weights
                    bias,
                    quant_kind=self.quant_kind,
                    gptq_qweight=self.gptq_qweight,
                    gptq_qzeros=self.gptq_qzeros,
                    gptq_scales=self.gptq_scales,
                    gptq_g_idx=self.gptq_g_idx,
                    weight_bits=bits,
                    use_v2_format=False,
                    out_features=out_features,
                    in_features=in_features,
                    group_size=group_size,
                )

            if weight_format == "awq":
                # AWQ is 4-bit only in vLLM; bits stored in _offline_quant_bits.
                bits = int(self._offline_quant_bits_py) if int(self._offline_quant_bits_py) > 0 else 4
                pack_factor = 32 // max(1, bits)
                return strategy.linear_forward(
                    x,
                    None,
                    bias,
                    quant_kind=self.quant_kind,
                    awq_qweight=self.awq_qweight,
                    awq_qzeros=self.awq_qzeros,
                    awq_scales=self.awq_scales,
                    pack_factor=pack_factor,
                    out_features=out_features,
                    in_features=in_features,
                    group_size=group_size,
                )

            if weight_format == "gptq_marlin":
                self._maybe_prepare_offline_gptq_marlin(x)
                bits = self._infer_gptq_weight_bits(in_features=in_features)
                return strategy.linear_forward(
                    x,
                    None,
                    bias,
                    quant_kind=self.quant_kind,
                    qweight=self.gptq_marlin_qweight,
                    scales=self.gptq_marlin_scales,
                    zp=self.gptq_marlin_zp,
                    g_idx=self.gptq_marlin_g_idx,
                    g_idx_sort_indices=self.gptq_marlin_g_idx_sort_indices,
                    workspace=self.gptq_marlin_workspace,
                    in_features=in_features,
                    out_features=out_features,
                    group_size=group_size,
                    weight_bits=bits,
                    tp_dim=self.tp_dim,
                )

            if weight_format == "awq_marlin":
                self._maybe_prepare_offline_awq_marlin(x)
                return strategy.linear_forward(
                    x,
                    None,
                    bias,
                    quant_kind=self.quant_kind,
                    qweight=self.awq_marlin_qweight,
                    scales=self.awq_marlin_scales,
                    zp=self.awq_marlin_zp,
                    workspace=self.awq_marlin_workspace,
                    in_features=in_features,
                    out_features=out_features,
                    group_size=group_size,
                    tp_dim=self.tp_dim,
                )

            # Fallback: compatibility for any remaining strategies.
            kwargs = self._build_offline_forward_kwargs(x, strategy)
            return strategy.linear_forward(
                x,
                None,
                bias,
                quant_kind=self.quant_kind,
                **kwargs,
            )

        if self.has_quantized_weight():
            if strategy is None:
                raise RuntimeError("Quantized weight is present but no linear strategy is configured.")
            # Hot path: avoid per-call dict construction when possible.
            extra_kwargs = self._maybe_int4_original_in_features_kwargs(strategy, x)
            # W8A16(AllSpark) expects scales in 1xN layout and needs explicit N.
            if getattr(strategy, "name", "") == "linear_int8_w8a16":
                if extra_kwargs:
                    return strategy.linear_forward(
                        x,
                        self.quant_weight_int8,
                        bias,
                        quant_kind=self.quant_kind,
                        quant_scales=self.quant_scales_1xn,
                        out_features=self._forward_out_features,
                        **extra_kwargs,
                    )
                return strategy.linear_forward(
                    x,
                    self.quant_weight_int8,
                    bias,
                    quant_kind=self.quant_kind,
                    quant_scales=self.quant_scales_1xn,
                    out_features=self._forward_out_features,
                )

            # W8A8 expects scales in 1xN layout and is sensitive to weight layout (KxN stride0==1).
            if getattr(strategy, "name", "") == "linear_int8_w8a8":
                if extra_kwargs:
                    return strategy.linear_forward(
                        x,
                        self.quant_weight_int8,
                        bias,
                        quant_kind=self.quant_kind,
                        quant_scales=self.quant_scales_1xn,
                        out_features=self._forward_out_features,
                        **extra_kwargs,
                    )
                return strategy.linear_forward(
                    x,
                    self.quant_weight_int8,
                    bias,
                    quant_kind=self.quant_kind,
                    quant_scales=self.quant_scales_1xn,
                    out_features=self._forward_out_features,
                )

            if extra_kwargs:
                return strategy.linear_forward(
                    x,
                    self.quant_weight_int8,
                    bias,
                    quant_kind=self.quant_kind,
                    quant_scales=self.quant_scales,
                    **extra_kwargs,
                )
            return strategy.linear_forward(
                x,
                self.quant_weight_int8,
                bias,
                quant_kind=self.quant_kind,
                quant_scales=self.quant_scales,
            )

        if strategy is None:
            weight = getattr(self, "weight", None)
            if weight is None:
                raise RuntimeError("No strategy is configured and bf16 weight is missing.")
            return F.linear(x, weight, bias)

        weight = getattr(self, "weight", None)
        if weight is None:
            raise RuntimeError("Strategy is configured but weight is missing (expected bf16 weight).")
        kwargs = self._maybe_int4_original_in_features_kwargs(strategy, x)
        if kwargs:
            return strategy.linear_forward(x, weight, bias, quant_kind=self.quant_kind, **kwargs)
        return strategy.linear_forward(x, weight, bias, quant_kind=self.quant_kind)

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
        base_out = self._forward_base(x, self.bias)
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
        self._forward_out_features = int(self.output_size_per_partition)

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
        base_out = self._forward_base(x, self.bias)
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
        y = self._forward_base(x, bias)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return self.lora_forward(x, y)
