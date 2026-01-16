# Diffulex 量化模块架构总结

## 一、架构概述

Diffulex的量化模块采用**策略模式（Strategy Pattern）**和**上下文管理（Context Management）**设计，支持灵活的量化策略扩展。模块主要包含以下组件：

### 1. 核心组件

#### 1.1 配置层 (Config)
- **QuantizationConfig**: 顶级量化配置，包含KV cache、权重、激活的量化配置
- **KVCacheQuantConfig**: KV cache量化配置（dtype: bf16/fp8_e4m3/fp8_e5m2）
- **WeightQuantConfig**: 权重量化配置（支持按类型区分：attn/mlp）
- **ActivationQuantConfig**: 激活量化配置（支持按类型区分：attn/mlp）

#### 1.2 上下文管理 (Context)
- **QuantizationContext**: 线程本地存储（Thread-Local Storage），管理量化策略实例
  - 存储策略实例：`kv_cache`, `linear_attn`, `linear_mlp`, `linear_other`
  - 提供激活量化缓存（step-local cache）
  - 通过全局函数访问：`get_quantization_context()`, `get_kv_cache_strategy()`, `get_linear_strategy()`

#### 1.3 工厂模式 (Factory)
- **QuantizationStrategyFactory**: 从配置创建量化策略
  - `create_from_config()`: 从Diffulex配置对象创建并配置量化上下文
  - `create_kv_cache_strategy()`: 创建KV cache量化策略

#### 1.4 注册表 (Registry)
- **KV Cache策略注册表**: 通过`@register_kv_cache_strategy`装饰器注册
- **Linear策略注册表**: 通过`@register_linear_strategy`装饰器注册（按weight_dtype + act_dtype配对）
- 支持dtype别名和规范化（如"fp8" -> "fp8_e4m3"）

#### 1.5 策略接口 (Strategy Interfaces)
- **QuantizationStrategy**: 基础抽象类
  - `quantize()`: 量化张量
  - `dequantize()`: 反量化张量
  - `get_storage_dtype()`: 获取存储数据类型
  - `get_scale_shape()`: 获取scale张量形状
  
- **KVCacheQuantizationStrategy**: KV cache量化策略接口
  - `compute_scales()`: 计算量化scale
  - `update_scales()`: 更新量化scale（如running max策略）
  - `init_scales()`: 初始化scale
  - `quantize_kv_for_store()`: 量化KV用于存储
  - `view_kv_cache_for_kernels()`: 为kernel提供视图

- **LinearQuantizationStrategy**: Linear层量化策略接口
  - `linear_forward()`: 执行量化Linear前向传播
  - `quantize_weight_for_kernel()`: 为kernel量化权重
  - `quantize_act_for_kernel()`: 为kernel量化激活

#### 1.6 具体策略实现 (Strategy Implementations)

**KV Cache策略**:
- `KVCacheBF16Strategy`: BF16存储（无量化）
- `KVCacheFP8RunningMaxStrategy`: FP8量化（E4M3/E5M2），使用running max管理scale

**Linear策略**:
- `LinearBF16Strategy`: BF16权重+BF16激活（无量化）
- `LinearGPTQW4A16Strategy`: GPTQ W4权重+BF16激活
- `LinearAWQW4A16Strategy`: AWQ W4权重+BF16激活
- `LinearInt8W8A16Strategy`: INT8权重+BF16激活
- `LinearInt8W8A8Strategy`: INT8权重+INT8激活
- `LinearInt4W4A16Strategy`: INT4权重+BF16激活
- `LinearInt4W4A8Strategy`: INT4权重+INT8激活
- `LinearFP8W8A16Strategy`: FP8权重+BF16激活
- `LinearFP8W8A8Strategy`: FP8权重+FP8激活
- `LinearStubStrategy`: 占位策略（未实现的组合）

#### 1.7 工具函数 (Utilities)
- **kv_cache_dtype.py**: KV cache数据类型处理
  - `parse_kv_cache_dtype()`: 解析dtype字符串
  - `view_fp8_cache()`: FP8 cache视图转换
  - `ensure_scale_tensor()`: 确保scale张量格式正确

## 二、与其他模块的耦合关系

### 2.1 模型运行器 (Model Runner)
**文件**: `diffulex/engine/model_runner.py`
- **初始化**: 在`ModelRunnerBase.__init__()`中调用`QuantizationStrategyFactory.create_from_config(config)`
- **KV Cache分配**: 使用`get_kv_cache_strategy()`获取策略，根据策略分配KV cache存储

### 2.2 Linear层
**文件**: `diffulex/layer/linear.py`
- **前向传播**: 在`forward()`中调用`get_linear_strategy(quant_kind)`获取策略
- **权重量化**: 在`_maybe_quantize_loaded_weight_param()`中，加载权重后自动量化并删除BF16权重参数
- **离线量化支持**: 支持GPTQ/AWQ离线量化权重的加载和使用

### 2.3 KV Cache Kernels
**文件**: `diffulex_kernel/python/kv_cache_kernels.py`, `diffulex_kernel/python/dllm_flash_attn_kernels.py`
- **策略获取**: 在kernel函数中调用`get_kv_cache_strategy()`获取策略
- **Scale管理**: 使用策略的`update_scales()`更新scale
- **Cache视图**: 使用策略的`view_kv_cache_for_kernels()`获取适合kernel的视图

### 2.4 注意力实现
**文件**: `diffulex/attention/attn_impl.py`
- **策略获取**: 在注意力计算中获取KV cache策略
- **Scale传递**: 将scale传递给attention metadata

### 2.5 TP Worker
**文件**: `diffulex/engine/tp_worker.py`
- **缓存清理**: 在每个step开始时调用`clear_act_quant_cache()`清理激活量化缓存

## 三、量化流程

### 3.1 初始化流程
1. `ModelRunnerBase.__init__()` 调用 `QuantizationStrategyFactory.create_from_config(config)`
2. Factory从config解析`QuantizationConfig`
3. Factory创建KV cache策略和Linear策略（按attn/mlp/other分类）
4. 策略注册到`QuantizationContext`（线程本地存储）

### 3.2 KV Cache量化流程
1. **初始化**: 调用`strategy.init_scales()`初始化scale张量
2. **存储**: 在KV cache存储时，调用`strategy.quantize_kv_for_store()`量化K和V
3. **更新**: 每次前向传播后，调用`strategy.update_scales()`更新running max scale
4. **使用**: Kernel使用`strategy.view_kv_cache_for_kernels()`获取适合的视图

### 3.3 Linear量化流程
1. **权重量化**: 
   - 在线量化：加载权重时自动调用`strategy.quantize_weight_for_kernel()`
   - 离线量化：通过`set_offline_quantized_weight()`加载GPTQ/AWQ权重
2. **前向传播**: 
   - 调用`strategy.linear_forward()`执行量化计算
   - 支持TileLang kernel加速（如GPTQ W4A16）
   - 支持Python fallback实现

### 3.4 激活量化流程（W8A8/W4A8）
1. **缓存**: 使用`QuantizationContext`的step-local cache缓存激活量化结果
2. **量化**: 在Linear层前向传播时，调用`strategy.quantize_act_for_kernel()`
3. **清理**: 每个step开始时清理缓存

## 四、扩展性设计

### 4.1 添加新的KV Cache策略
1. 实现`KVCacheQuantizationStrategy`接口
2. 使用`@register_kv_cache_strategy("dtype_alias")`注册
3. 在`strategies/__init__.py`中导入（触发注册）

### 4.2 添加新的Linear策略
1. 实现`LinearQuantizationStrategy`接口
2. 使用`@register_linear_strategy(weight_dtype="...", act_dtype="...")`注册
3. 在`strategies/__init__.py`中导入（触发注册）

### 4.3 支持新的量化方法
- 权重量化：GPTQ, AWQ, INT8, INT4, FP8
- 激活量化：INT8, INT4, FP8
- KV Cache量化：FP8 (E4M3/E5M2)

## 五、架构图

详见下面的Mermaid图表。
