# Integrate FP8 KV Cache Support in Attention Layers

## Overview

在 `diffulex_legacy/layers/attention/attention_v4.py` 和 `attention_v5.py` 中集成 FP8 KV cache 支持，使得 store/load 函数能够正确处理 FP8 量化/反量化。采用 running max 策略维护 per-head scale。

## Current State Analysis

- `store_kvcache_unified_layout()` 和 `store_kvcache_distinct_layout()` 已支持 `kv_cache_dtype`, `k_scale`, `v_scale` 参数（默认值：`"bf16"`, `None`, `None`）
- `load_kvcache()` 已支持 `kv_cache_dtype`, `k_scale`, `v_scale` 参数
- Attention 层目前调用 store/load 时未传递这些参数
- 对于 diffusion_lm：可通过 `context.seqs[0].config.kv_cache_dtype` 获取配置
- 对于 causal_lm：ContextForCausalLM 中缺少 config 信息

## Implementation Plan

### Phase 1: Add kv_cache_dtype Access Support

#### 1.1 Extend ContextForCausalLM to support kv_cache_dtype

- **File**: `diffulex_legacy/utils/context.py`
- **Changes**:
- 在 `ContextForCausalLM` dataclass 中添加 `kv_cache_dtype: str = "bf16"` 字段
- 在 `set_context_causal_lm()` 函数中添加 `kv_cache_dtype: str = "bf16"` 参数（带默认值，保持向后兼容）
- 在 `ModelRunnerForCausalLM` 中调用 `set_context_causal_lm()` 时传递 `kv_cache_dtype=self.config.kv_cache_dtype`
    - 位置1: `prepare_prefill()` 方法（约第274行）
    - 位置2: `prepare_decode()` 方法（约第295行）
    - 位置3: `capture_cudagraph()` 方法（约第360行）

#### 1.2 Add helper function to get kv_cache_dtype from context

- **Files**: `attention_v4.py`, `attention_v5.py`
- **Changes**:
- 在文件顶部添加辅助函数：
    ```python
            def _get_kv_cache_dtype(context: ContextForDiffusionLM, model_type: str) -> str:
                if model_type == 'diffusion_lm':
                    return context.seqs[0].config.kv_cache_dtype
                else:  # causal_lm
                    return getattr(context, 'kv_cache_dtype', 'bf16')  # fallback for backward compatibility
    ```




### Phase 2: Implement Running Max Scale Management

#### 2.1 Add running max state to Attention class

- **Files**: `attention_v4.py`, `attention_v5.py`
- **Changes**:
- 在 `Attention.__init__()` 中添加：
    ```python
            # FP8 scale management: maintain running max per head
            self.k_max_abs: torch.Tensor | None = None  # [num_kv_heads]
            self.v_max_abs: torch.Tensor | None = None  # [num_kv_heads]
            self.kv_cache_dtype_cache: str | None = None
    ```




#### 2.2 Create scale computation utility function

- **Files**: `attention_v4.py`, `attention_v5.py`
- **Changes**:
- 添加 `_update_and_compute_fp8_scales()` 方法：
    ```python
            def _update_and_compute_fp8_scales(
                self,
                k: torch.Tensor, 
                v: torch.Tensor, 
                kv_cache_dtype: str, 
                device: torch.device
            ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
                """
                Update running max and compute FP8 scales.
                Returns (k_scale, v_scale) or (None, None) if not FP8.
                """
                from diffulex.utils.kv_cache_dtype import parse_kv_cache_dtype
                spec = parse_kv_cache_dtype(kv_cache_dtype)
                if not spec.is_fp8:
                    return None, None
                
                # Reset running max if dtype changed
                if self.kv_cache_dtype_cache != kv_cache_dtype:
                    self.k_max_abs = None
                    self.v_max_abs = None
                    self.kv_cache_dtype_cache = kv_cache_dtype
                
                # Compute current batch absmax: [num_kv_heads]
                k_absmax = k.to(torch.float32).abs().amax(dim=(0, 2))  # [num_kv_heads]
                v_absmax = v.to(torch.float32).abs().amax(dim=(0, 2))  # [num_kv_heads]
                
                # Update running max
                if self.k_max_abs is None:
                    self.k_max_abs = k_absmax.clone().detach()
                    self.v_max_abs = v_absmax.clone().detach()
                else:
                    self.k_max_abs = torch.maximum(self.k_max_abs, k_absmax)
                    self.v_max_abs = torch.maximum(self.v_max_abs, v_absmax)
                
                # Compute scale from running max
                eps = 1e-8
                fp8_max = spec.fp8_max
                k_scale = (self.k_max_abs / fp8_max).clamp_min(eps)
                v_scale = (self.v_max_abs / fp8_max).clamp_min(eps)
                
                return k_scale, v_scale
    ```




#### 2.3 Add helper method to get scales from running max

- **Files**: `attention_v4.py`, `attention_v5.py`
- **Changes**:
- 添加辅助方法：
    ```python
            def _get_fp8_scales_from_max(self, kv_cache_dtype: str) -> tuple[torch.Tensor | None, torch.Tensor | None]:
                """Convert running max to scales. Returns (None, None) if not FP8 or max not initialized."""
                from diffulex.utils.kv_cache_dtype import parse_kv_cache_dtype
                spec = parse_kv_cache_dtype(kv_cache_dtype)
                if not spec.is_fp8 or self.k_max_abs is None or self.v_max_abs is None:
                    return None, None
                eps = 1e-8
                fp8_max = spec.fp8_max
                k_scale = (self.k_max_abs / fp8_max).clamp_min(eps)
                v_scale = (self.v_max_abs / fp8_max).clamp_min(eps)
                return k_scale, v_scale
    ```




### Phase 3: Integrate Scale Computation in Attention Layers

#### 3.1 Modify forward() to compute and pass scales for store

- **Files**: `attention_v4.py` (line 98-99), `attention_v5.py` (line 99-100)
- **Current code**:
  ```python
      store_kvcache = store_kvcache_unified_layout if is_unified_layout else store_kvcache_distinct_layout
      store_kvcache(k, v, k_cache, v_cache, context.slot_mapping, self.model_type, context)
  ```




- **New code**:
  ```python
      kv_cache_dtype = _get_kv_cache_dtype(context, self.model_type)
      k_scale, v_scale = self._update_and_compute_fp8_scales(k, v, kv_cache_dtype, k.device)
      store_kvcache = store_kvcache_unified_layout if is_unified_layout else store_kvcache_distinct_layout
      store_kvcache(
          k, v, k_cache, v_cache, context.slot_mapping, self.model_type, context,
          kv_cache_dtype=kv_cache_dtype,
          k_scale=k_scale,
          v_scale=v_scale
      )
  ```




#### 3.2 Modify forward() to pass scales for load

- **Files**: `attention_v4.py` (line 132), `attention_v5.py` (line 132)
- **Current code**:
  ```python
      k_comb, v_comb = load_kvcache(self.k_cache, self.v_cache, context, k, v)
  ```




- **New code**:
  ```python
      kv_cache_dtype = _get_kv_cache_dtype(context, self.model_type)
      # Try to get scales from running max, or compute if not available
      k_scale, v_scale = self._get_fp8_scales_from_max(kv_cache_dtype)
      if k_scale is None and v_scale is None:
          # Scale not initialized yet, compute from current k, v
          k_scale, v_scale = self._update_and_compute_fp8_scales(k, v, kv_cache_dtype, k.device)
      k_comb, v_comb = load_kvcache(
          self.k_cache, self.v_cache, context, k, v,
          kv_cache_dtype=kv_cache_dtype,
          k_scale=k_scale,
          v_scale=v_scale
      )
  ```




### Phase 4: Update ModelRunnerForCausalLM

#### 4.1 Pass kv_cache_dtype to context

- **File**: `diffulex_legacy/engine/model_runner.py`
- **Changes**:
- 在 `prepare_prefill()` 方法中，修改 `set_context_causal_lm()` 调用（约第274行）：
    ```python
            set_context_causal_lm(
                True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, 
                slot_mapping, None, block_tables, 
                kv_cache_dtype=self.config.kv_cache_dtype
            )
    ```




- 在 `prepare_decode()` 方法中，修改 `set_context_causal_lm()` 调用（约第295行）：
    ```python
            set_context_causal_lm(
                False, cu_seqlens_k=cu_seqlens_k, slot_mapping=slot_mapping, 
                context_lens=context_lens, block_tables=block_tables,
                kv_cache_dtype=self.config.kv_cache_dtype
            )
    ```




- 在 `capture_cudagraph()` 方法中，修改 `set_context_causal_lm()` 调用（约第360行）：
    ```python
            set_context_causal_lm(
                False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], 
                block_tables=block_tables[:bs],
                kv_cache_dtype=self.config.kv_cache_dtype
            )
    ```




## Risk Assessment

### Low Risk

- 添加 `kv_cache_dtype` 参数到 ContextForCausalLM：向后兼容（默认值 "bf16"）
- 在 store/load 调用中添加可选参数：函数已有默认值，不影响现有调用
- Running max 初始化：使用 None 作为初始值，首次使用时初始化

### Medium Risk

- Running max 的内存管理：需要在设备上维护 tensor，需要考虑设备一致性
- Scale 计算性能：每次 forward 时更新 running max 和计算 scale 有开销，但这是必要的
- 多线程/多进程安全：如果 Attention 层在多线程环境中共享，需要考虑同步

### High Risk

- **Scale 一致性**：如果 load 在 store 之前被调用，需要确保 scale 正确初始化
- **Cache 重置时机**：当 kv_cache_dtype 改变时，需要重置 running max，但如何检测改变需要仔细处理

### Mitigation Strategies

1. **向后兼容性**：所有新增参数都有默认值，不会破坏现有代码
2. **设备一致性**：确保 running max tensor 与 k/v tensor 在同一设备上
3. **Scale 初始化**：在 load 之前检查 scale 是否存在，如果不存在则先计算
4. **Dtype 变更检测**：通过比较 `self.kv_cache_dtype_cache` 与当前 `kv_cache_dtype` 来检测变更

## Testing Strategy

### Unit Tests

1. **Test running max update**:

- 验证首次调用时正确初始化
- 验证后续调用时正确更新（取最大值）
- 验证 dtype 变更时正确重置

2. **Test scale computation**:

- 验证 FP8 时正确计算 scale
- 验证非 FP8 时返回 None
- 验证 scale 形状正确（[num_kv_heads]）

3. **Test context kv_cache_dtype**:

- 验证 causal_lm context 正确设置和获取 kv_cache_dtype
- 验证 diffusion_lm context 从 config 获取 kv_cache_dtype

### Integration Tests

1. **Test attention layer with FP8**:

- 使用 FP8 KV cache 运行完整 forward pass
- 验证 store 和 load 正确传递参数
- 验证量化/反量化正确性（可复用现有 roundtrip 测试思路）
- 验证多次 forward 调用时 running max 正确累积

2. **Test backward compatibility**:

- 使用默认 bf16 运行，确保行为不变
- 验证未指定 kv_cache_dtype 时使用默认值

### Manual Testing

1. 使用实际模型运行 inference，验证 FP8 KV cache 功能
2. 对比 FP8 和 BF16 的内存使用和性能
3. 验证长时间运行（多次 forward）时 scale 正确维护

## Files to Modify

1. `diffulex_legacy/utils/context.py` - 添加 kv_cache_dtype 到 ContextForCausalLM
2. `diffulex_legacy/engine/model_runner.py` - 传递 kv_cache_dtype 到 context（3处）
3. `diffulex_legacy/layers/attention/attention_v4.py` - 集成 FP8 支持
4. `diffulex_legacy/layers/attention/attention_v5.py` - 集成 FP8 支持

## Implementation Order

1. Phase 1: Context extension (causal_lm support)
2. Phase 2: Running max scale management infrastructure
3. Phase 3: Attention layer integration (v4 and v5 in parallel)
4. Phase 4: ModelRunner update

## Notes

- Running max 策略确保 scale 能够适应逐渐增大的值，同时保持 per-head 的固定性（每个 head 一个固定的 scale）