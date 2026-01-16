# Diffulex 量化模块架构图

## 完整架构图

```mermaid
graph TB
    subgraph "用户配置层"
        Config[Diffulex Config<br/>kv_cache_dtype<br/>linear_attn_weight_dtype<br/>linear_mlp_weight_dtype<br/>...]
    end

    subgraph "量化模块核心"
        subgraph "配置解析"
            QC[QuantizationConfig]
            KVC[KVCacheQuantConfig]
            WC[WeightQuantConfig]
            AC[ActivationQuantConfig]
            Config --> QC
            QC --> KVC
            QC --> WC
            QC --> AC
        end

        subgraph "工厂与注册表"
            Factory[QuantizationStrategyFactory<br/>create_from_config<br/>create_kv_cache_strategy]
            RegKV[KV Cache Registry<br/>@register_kv_cache_strategy]
            RegLinear[Linear Registry<br/>@register_linear_strategy]
            Factory --> RegKV
            Factory --> RegLinear
        end

        subgraph "上下文管理"
            Context[QuantizationContext<br/>Thread-Local Storage]
            Context --> |存储| KVStrategy[KV Cache Strategy]
            Context --> |存储| LinearAttn[Linear Attn Strategy]
            Context --> |存储| LinearMLP[Linear MLP Strategy]
            Context --> |存储| LinearOther[Linear Other Strategy]
            Context --> |缓存| ActCache[Activation Quant Cache<br/>Step-Local]
        end

        subgraph "策略接口层"
            BaseStrategy[QuantizationStrategy<br/>quantize/dequantize<br/>get_storage_dtype]
            KVInterface[KVCacheQuantizationStrategy<br/>compute_scales<br/>update_scales<br/>quantize_kv_for_store]
            LinearInterface[LinearQuantizationStrategy<br/>linear_forward<br/>quantize_weight_for_kernel<br/>quantize_act_for_kernel]
            BaseStrategy --> KVInterface
            BaseStrategy --> LinearInterface
        end

        subgraph "KV Cache策略实现"
            KVBF16[KVCacheBF16Strategy<br/>BF16存储]
            KVFP8[KVCacheFP8RunningMaxStrategy<br/>FP8 E4M3/E5M2<br/>Running Max Scale]
            KVInterface --> KVBF16
            KVInterface --> KVFP8
        end

        subgraph "Linear策略实现"
            LBF16[LinearBF16Strategy<br/>BF16/BF16]
            LGPTQ[LinearGPTQW4A16Strategy<br/>GPTQ W4/BF16]
            LAWQ[LinearAWQW4A16Strategy<br/>AWQ W4/BF16]
            LInt8W8A16[LinearInt8W8A16Strategy<br/>INT8/BF16]
            LInt8W8A8[LinearInt8W8A8Strategy<br/>INT8/INT8]
            LInt4W4A16[LinearInt4W4A16Strategy<br/>INT4/BF16]
            LInt4W4A8[LinearInt4W4A8Strategy<br/>INT4/INT8]
            LFP8W8A16[LinearFP8W8A16Strategy<br/>FP8/BF16]
            LFP8W8A8[LinearFP8W8A8Strategy<br/>FP8/FP8]
            LinearInterface --> LBF16
            LinearInterface --> LGPTQ
            LinearInterface --> LAWQ
            LinearInterface --> LInt8W8A16
            LinearInterface --> LInt8W8A8
            LinearInterface --> LInt4W4A16
            LinearInterface --> LInt4W4A8
            LinearInterface --> LFP8W8A16
            LinearInterface --> LFP8W8A8
        end

        subgraph "工具函数"
            KVDType[kv_cache_dtype.py<br/>parse_kv_cache_dtype<br/>view_fp8_cache<br/>ensure_scale_tensor]
        end
    end

    subgraph "运行时模块"
        subgraph "模型运行器"
            MR[ModelRunnerBase<br/>__init__]
            MR --> |初始化| Factory
            MR --> |获取| Context
        end

        subgraph "Linear层"
            Linear[LinearBase<br/>ReplicatedLinear<br/>ColumnParallelLinear<br/>RowParallelLinear]
            Linear --> |forward| Context
            Linear --> |quantize_weight| Context
        end

        subgraph "KV Cache Kernels"
            KVKernel[kv_cache_kernels.py<br/>dllm_flash_attn_kernels.py]
            KVKernel --> |获取策略| Context
            KVKernel --> |更新scale| KVStrategy
        end

        subgraph "注意力实现"
            Attn[attn_impl.py]
            Attn --> |获取策略| Context
        end

        subgraph "TP Worker"
            TP[tp_worker.py]
            TP --> |清理缓存| Context
        end
    end

    subgraph "离线量化工具"
        Offline[quantize_model.py<br/>GPTQ/AWQ离线量化]
    end

    %% 连接关系
    QC --> Factory
    Factory --> Context
    RegKV --> KVBF16
    RegKV --> KVFP8
    RegLinear --> LBF16
    RegLinear --> LGPTQ
    RegLinear --> LAWQ
    RegLinear --> LInt8W8A16
    RegLinear --> LInt8W8A8
    RegLinear --> LInt4W4A16
    RegLinear --> LInt4W4A8
    RegLinear --> LFP8W8A16
    RegLinear --> LFP8W8A8
    KVStrategy --> KVInterface
    LinearAttn --> LinearInterface
    LinearMLP --> LinearInterface
    LinearOther --> LinearInterface
    KVDType --> KVFP8

    style Config fill:#e1f5ff
    style QC fill:#fff4e1
    style Factory fill:#fff4e1
    style Context fill:#e8f5e9
    style KVInterface fill:#f3e5f5
    style LinearInterface fill:#f3e5f5
    style KVBF16 fill:#fff9c4
    style KVFP8 fill:#fff9c4
    style LGPTQ fill:#fff9c4
    style LAWQ fill:#fff9c4
    style MR fill:#ffebee
    style Linear fill:#ffebee
    style KVKernel fill:#ffebee
```

## 数据流图

```mermaid
sequenceDiagram
    participant Config as Diffulex Config
    participant Factory as QuantizationStrategyFactory
    participant Context as QuantizationContext
    participant KVStrategy as KV Cache Strategy
    participant LinearStrategy as Linear Strategy
    participant ModelRunner as ModelRunner
    participant LinearLayer as Linear Layer
    participant KVKernel as KV Cache Kernel

    Note over Config,KVKernel: 初始化阶段
    Config->>Factory: create_from_config(config)
    Factory->>Context: 创建并配置上下文
    Factory->>KVStrategy: 创建KV cache策略
    Factory->>LinearStrategy: 创建Linear策略(attn/mlp/other)
    Context->>Context: 存储策略实例

    Note over ModelRunner,KVKernel: 运行时阶段
    ModelRunner->>Context: get_kv_cache_strategy()
    Context->>KVStrategy: 返回策略实例
    ModelRunner->>KVStrategy: init_scales()
    KVStrategy->>KVStrategy: 初始化scale张量

    LinearLayer->>Context: get_linear_strategy(quant_kind)
    Context->>LinearStrategy: 返回策略实例
    LinearLayer->>LinearStrategy: linear_forward(x, weight, bias)
    LinearStrategy->>LinearStrategy: 执行量化计算

    KVKernel->>Context: get_kv_cache_strategy()
    Context->>KVStrategy: 返回策略实例
    KVKernel->>KVStrategy: update_scales(k, v, k_scale, v_scale)
    KVStrategy->>KVStrategy: 更新running max scale
    KVKernel->>KVStrategy: quantize_kv_for_store(k, v, scales)
    KVStrategy->>KVKernel: 返回量化后的K和V
```

## 策略选择流程图

```mermaid
flowchart TD
    Start[开始] --> LoadConfig[加载Diffulex Config]
    LoadConfig --> ParseConfig[解析QuantizationConfig]
    ParseConfig --> CheckKVCache{检查kv_cache_dtype}
    
    CheckKVCache -->|bf16/fp16/fp32| CreateKVBF16[创建KVCacheBF16Strategy]
    CheckKVCache -->|fp8/fp8_e4m3| CreateKVFP8E4M3[创建KVCacheFP8RunningMaxStrategy<br/>E4M3]
    CheckKVCache -->|fp8_e5m2| CreateKVFP8E5M2[创建KVCacheFP8RunningMaxStrategy<br/>E5M2]
    
    ParseConfig --> CheckLinearAttn{检查linear_attn配置}
    CheckLinearAttn -->|weight_dtype + act_dtype| CreateLinearAttn[创建Linear策略<br/>注册到linear_attn]
    
    ParseConfig --> CheckLinearMLP{检查linear_mlp配置}
    CheckLinearMLP -->|weight_dtype + act_dtype| CreateLinearMLP[创建Linear策略<br/>注册到linear_mlp]
    
    CreateKVBF16 --> RegisterContext[注册到QuantizationContext]
    CreateKVFP8E4M3 --> RegisterContext
    CreateKVFP8E5M2 --> RegisterContext
    CreateLinearAttn --> RegisterContext
    CreateLinearMLP --> RegisterContext
    
    RegisterContext --> End[完成初始化]
    
    style CheckKVCache fill:#e1f5ff
    style CheckLinearAttn fill:#e1f5ff
    style CheckLinearMLP fill:#e1f5ff
    style RegisterContext fill:#e8f5e9
```

## Linear量化决策流程图

```mermaid
flowchart TD
    Start[Linear.forward调用] --> GetStrategy[get_linear_strategy<br/>quant_kind]
    GetStrategy --> CheckOffline{检查离线量化权重<br/>GPTQ/AWQ}
    
    CheckOffline -->|有GPTQ权重| UseGPTQ[使用GPTQ策略<br/>linear_forward<br/>传递qweight/qzeros/scales]
    CheckOffline -->|有AWQ权重| UseAWQ[使用AWQ策略<br/>linear_forward<br/>传递qweight/qzeros/scales]
    CheckOffline -->|无离线量化| CheckOnline{检查在线量化权重<br/>int8/int4/fp8}
    
    CheckOnline -->|有量化权重| UseOnline[使用量化策略<br/>linear_forward<br/>传递quant_weight_int8/scales]
    CheckOnline -->|无量化权重| CheckStrategy{检查策略}
    
    CheckStrategy -->|有策略| UseStrategy[使用策略<br/>linear_forward<br/>传递bf16 weight]
    CheckStrategy -->|无策略| UseDefault[使用默认F.linear<br/>bf16 weight]
    
    UseGPTQ --> TryKernel{尝试TileLang Kernel}
    TryKernel -->|成功| KernelResult[Kernel计算结果]
    TryKernel -->|失败| PythonFallback[Python Fallback<br/>dequantize + F.linear]
    
    UseAWQ --> TryKernel
    UseOnline --> KernelOrPython[Kernel或Python实现]
    UseStrategy --> KernelOrPython
    UseDefault --> Result[返回结果]
    
    KernelResult --> Result
    PythonFallback --> Result
    KernelOrPython --> Result
    
    style CheckOffline fill:#e1f5ff
    style CheckOnline fill:#e1f5ff
    style CheckStrategy fill:#e1f5ff
    style TryKernel fill:#fff9c4
```

## KV Cache量化流程图

### 完整KV Cache量化流程（包含Store和Load）

```mermaid
flowchart TB
    subgraph "Store阶段"
        Start[KV Cache Store] --> GetStrategy1[get_kv_cache_strategy]
        GetStrategy1 --> CheckFormat1{检查kv_cache_format}
        
        CheckFormat1 -->|bf16| BF16Store[BF16 Store路径]
        CheckFormat1 -->|fp8| FP8Store[FP8 Store路径]
        
        BF16Store --> StoreBF16[直接存储为BF16<br/>dtype: bfloat16<br/>无需量化]
        
        FP8Store --> UpdateScales["update_scales<br/>更新running max scale<br/>k_scale/v_scale: float32<br/>shape: (num_kv_heads)"]
        UpdateScales --> QuantizeKV["quantize_kv_for_store<br/>K/V: bfloat16 -> uint8<br/>使用k_scale/v_scale量化"]
        QuantizeKV --> StoreFP8["存储为uint8<br/>dtype: uint8<br/>FP8格式"]
        
        StoreBF16 --> CheckLayout1{检查Layout}
        StoreFP8 --> CheckLayout1
        
        CheckLayout1 -->|unified| StoreUnified["store_kvcache_unified_layout<br/>shape: (num_blocks, page_size, num_kv_heads, head_dim)"]
        CheckLayout1 -->|distinct| StoreDistinct["store_kvcache_distinct_layout<br/>k_cache: (num_blks, h, hdim//x, blk_sz, x)<br/>v_cache: (num_blks, h, hdim, blk_sz)"]
    end
    
    subgraph "Load阶段"
        LoadStart[KV Cache Load] --> GetStrategy2[get_kv_cache_strategy]
        GetStrategy2 --> CheckFormat2{检查kv_cache_format}
        
        CheckFormat2 -->|bf16| BF16Load[BF16 Load路径]
        CheckFormat2 -->|fp8| FP8Load[FP8 Load路径]
        
        BF16Load --> CheckLayout2{检查Layout}
        FP8Load --> CheckLayout2
        
        CheckLayout2 -->|unified| UnifiedLoad[Unified Layout Load]
        CheckLayout2 -->|distinct| DistinctLoad[Distinct Layout Load<br/>总是使用varlen路径]
        
        UnifiedLoad --> CheckDecodeMode{检查decode_mode}
        CheckDecodeMode -->|static| StaticPath[Static模式<br/>TileLang Kernel]
        CheckDecodeMode -->|varlen| VarlenPath[Varlen模式<br/>load_kvcache + flash_attn_varlen_func]
        
        DistinctLoad --> VarlenPath
        
        StaticPath --> StaticBF16{BF16?}
        StaticPath --> StaticFP8{FP8?}
        
        StaticBF16 --> TileLangBF16[dllm_flash_attn_decode_kernel<br/>TileLang Kernel<br/>输入: q/k/v/cache bfloat16<br/>输出: bfloat16]
        
        StaticFP8 --> ViewFP8Cache[strategy.view_kv_cache_for_kernels<br/>uint8 -> float8 view<br/>dtype转换]
        ViewFP8Cache --> TileLangFP8[dllm_flash_attn_decode_kernel_bf16_q_fp8_kv<br/>TileLang Kernel<br/>输入: q bfloat16, cache float8<br/>k_scale/v_scale float32<br/>kernel内反量化+scale<br/>输出: bfloat16]
        
        VarlenPath --> LoadKVCache[load_kvcache函数]
        LoadKVCache --> LoadBF16{BF16?}
        LoadKVCache --> LoadFP8{FP8?}
        
        LoadBF16 --> LoadBF16Kernel[_load_kvcache_bf16<br/>Triton Kernel<br/>gather cache blocks<br/>输出: bfloat16]
        
        LoadFP8 --> LoadFP8Kernel[_load_kvcache_fp8<br/>Triton Fused Kernel<br/>gather + dequant + scale<br/>输入: cache uint8/float8 view<br/>k_scale/v_scale float32<br/>输出: bfloat16]
        
        LoadBF16Kernel --> FlashAttnBF16[flash_attn_varlen_func<br/>输入: q/k_comb/v_comb bfloat16<br/>输出: bfloat16]
        LoadFP8Kernel --> FlashAttnFP8[flash_attn_varlen_func<br/>输入: q/k_comb/v_comb bfloat16<br/>输出: bfloat16]
    end
    
    StoreUnified --> LoadStart
    StoreDistinct --> LoadStart
    TileLangBF16 --> End[完成]
    TileLangFP8 --> End
    FlashAttnBF16 --> End
    FlashAttnFP8 --> End
    
    style CheckFormat1 fill:#e1f5ff
    style CheckFormat2 fill:#e1f5ff
    style CheckLayout1 fill:#fff9c4
    style CheckLayout2 fill:#fff9c4
    style CheckDecodeMode fill:#fff9c4
    style QuantizeKV fill:#ffebee
    style ViewFP8Cache fill:#ffebee
    style StaticPath fill:#e8f5e9
    style VarlenPath fill:#e8f5e9
```

### 数据类型传递详细图

```mermaid
sequenceDiagram
    participant AttnImpl as Attention Implementation
    participant Strategy as KV Cache Strategy
    participant StoreKernel as Store Kernel
    participant Cache as KV Cache Storage
    participant LoadKernel as Load Kernel
    participant DecodeKernel as Decode Kernel
    participant FlashAttn as flash_attn_varlen_func
    
    Note over AttnImpl,FlashAttn: BF16路径 (Unified Layout, Static Mode)
    AttnImpl->>Strategy: get_kv_cache_strategy()
    Strategy-->>AttnImpl: KVCacheBF16Strategy
    AttnImpl->>AttnImpl: k: (N, H, D) bfloat16<br/>v: (N, H, D) bfloat16
    AttnImpl->>StoreKernel: store_kvcache_unified_layout<br/>k, v, cache, slot_mapping
    StoreKernel->>Cache: 直接存储<br/>dtype: bfloat16<br/>shape: (num_blocks, page_size, H, D)
    AttnImpl->>DecodeKernel: dllm_flash_attn_decode<br/>q: bfloat16<br/>k_cache: bfloat16<br/>v_cache: bfloat16
    DecodeKernel->>DecodeKernel: TileLang Kernel<br/>内部gather + attention计算
    DecodeKernel-->>AttnImpl: output: bfloat16
    
    Note over AttnImpl,FlashAttn: FP8路径 (Unified Layout, Static Mode)
    AttnImpl->>Strategy: get_kv_cache_strategy()
    Strategy-->>AttnImpl: KVCacheFP8RunningMaxStrategy
    AttnImpl->>AttnImpl: k: (N, H, D) bfloat16<br/>v: (N, H, D) bfloat16
    AttnImpl->>Strategy: update_scales(k, v, k_scale, v_scale)
    Strategy-->>AttnImpl: k_scale: (H) float32<br/>v_scale: (H) float32
    AttnImpl->>Strategy: quantize_kv_for_store(k, v, k_scale, v_scale)
    Strategy->>Strategy: 量化: k/v bfloat16 -> uint8<br/>使用scale进行量化
    Strategy-->>AttnImpl: k_q: (N, H, D) uint8<br/>v_q: (N, H, D) uint8
    AttnImpl->>StoreKernel: store_kvcache_unified_layout<br/>k_q, v_q (uint8)
    StoreKernel->>Cache: 存储为uint8<br/>dtype: uint8<br/>shape: (num_blocks, page_size, H, D)
    AttnImpl->>Strategy: view_kv_cache_for_kernels(cache)
    Strategy->>Strategy: uint8 -> float8 view<br/>dtype转换（不改变存储）
    Strategy-->>AttnImpl: cache_fp8: float8 view
    AttnImpl->>DecodeKernel: dllm_flash_attn_decode_bf16_q_fp8_kv<br/>q: bfloat16<br/>k_cache: float8 view<br/>v_cache: float8 view<br/>k_scale: (H) float32<br/>v_scale: (H) float32
    DecodeKernel->>DecodeKernel: TileLang Kernel<br/>内部: gather + dequant + scale + attention<br/>float8 -> bfloat16 (反量化)
    DecodeKernel-->>AttnImpl: output: bfloat16
    
    Note over AttnImpl,FlashAttn: FP8路径 (Unified/Distinct Layout, Varlen Mode)
    AttnImpl->>Strategy: get_kv_cache_strategy()
    Strategy-->>AttnImpl: KVCacheFP8RunningMaxStrategy
    AttnImpl->>Strategy: update_scales(k, v, k_scale, v_scale)
    Strategy-->>AttnImpl: k_scale: (H) float32<br/>v_scale: (H) float32
    AttnImpl->>Strategy: quantize_kv_for_store(k, v, k_scale, v_scale)
    Strategy-->>AttnImpl: k_q: (N, H, D) uint8<br/>v_q: (N, H, D) uint8
    AttnImpl->>StoreKernel: store_kvcache_*_layout<br/>k_q, v_q (uint8)
    StoreKernel->>Cache: 存储为uint8<br/>dtype: uint8
    AttnImpl->>LoadKernel: load_kvcache(cache, metadata, k_new, v_new)
    LoadKernel->>Strategy: view_kv_cache_for_kernels(cache)
    Strategy-->>LoadKernel: cache_fp8: float8 view
    LoadKernel->>LoadKernel: Triton Fused Kernel<br/>load_kvcache_kernel_fp8_*<br/>输入: cache float8 view<br/>k_scale/v_scale float32<br/>操作: gather + dequant + scale<br/>输出: k_comb/v_comb bfloat16
    LoadKernel-->>AttnImpl: k_comb: (total_len, H, D) bfloat16<br/>v_comb: (total_len, H, D) bfloat16
    AttnImpl->>FlashAttn: flash_attn_varlen_func<br/>q: bfloat16<br/>k_comb: bfloat16<br/>v_comb: bfloat16
    FlashAttn-->>AttnImpl: output: bfloat16
```

### Layout和Decode模式决策树

```mermaid
flowchart TD
    Start[KV Cache操作] --> CheckLayout{检查kv_cache_layout}
    
        CheckLayout -->|unified| UnifiedPath["Unified Layout<br/>shape: (num_blocks, page_size, H, D)"]
        CheckLayout -->|distinct| DistinctPath["Distinct Layout<br/>k: (num_blks, h, hdim//x, blk_sz, x)<br/>v: (num_blks, h, hdim, blk_sz)"]
    
    UnifiedPath --> CheckDecodeMode{检查decode_mode}
    CheckDecodeMode -->|static| UnifiedStatic[Static模式<br/>TileLang Kernel]
    CheckDecodeMode -->|varlen| UnifiedVarlen[Varlen模式<br/>load_kvcache + flash_attn_varlen_func]
    
    DistinctPath --> DistinctVarlen[总是Varlen模式<br/>load_kvcache + flash_attn_varlen_func]
    
    UnifiedStatic --> CheckQuant1{量化格式?}
    CheckQuant1 -->|bf16| StaticBF16[TileLang BF16 Kernel<br/>dllm_flash_attn_decode_kernel<br/>输入/输出: bfloat16]
    CheckQuant1 -->|fp8| StaticFP8[TileLang FP8 Kernel<br/>dllm_flash_attn_decode_kernel_bf16_q_fp8_kv<br/>输入: q bfloat16, cache float8<br/>scale: float32<br/>输出: bfloat16]
    
    UnifiedVarlen --> CheckQuant2{量化格式?}
    DistinctVarlen --> CheckQuant2
    
    CheckQuant2 -->|bf16| VarlenBF16[load_kvcache_bf16<br/>Triton gather kernel<br/>输出: bfloat16<br/>+ flash_attn_varlen_func]
    CheckQuant2 -->|fp8| VarlenFP8[load_kvcache_fp8<br/>Triton fused kernel<br/>gather + dequant + scale<br/>输入: cache float8, scale float32<br/>输出: bfloat16<br/>+ flash_attn_varlen_func]
    
    StaticBF16 --> End[完成]
    StaticFP8 --> End
    VarlenBF16 --> End
    VarlenFP8 --> End
    
    style CheckLayout fill:#e1f5ff
    style CheckDecodeMode fill:#e1f5ff
    style CheckQuant1 fill:#fff9c4
    style CheckQuant2 fill:#fff9c4
    style UnifiedStatic fill:#e8f5e9
    style UnifiedVarlen fill:#e8f5e9
    style DistinctVarlen fill:#e8f5e9
    style StaticFP8 fill:#ffebee
    style VarlenFP8 fill:#ffebee
```

### 详细数据流图：Unified Layout Static模式（FP8）

```mermaid
flowchart LR
    subgraph "Store阶段"
        K1["K: bfloat16<br/>(N, H, D)"] --> UpdateScale["update_scales<br/>计算/更新scale"]
        V1["V: bfloat16<br/>(N, H, D)"] --> UpdateScale
        UpdateScale --> KScale["k_scale: float32<br/>(H)"]
        UpdateScale --> VScale["v_scale: float32<br/>(H)"]
        K1 --> Quantize["quantize_kv_for_store<br/>使用scale量化"]
        V1 --> Quantize
        KScale --> Quantize
        VScale --> Quantize
        Quantize --> KQ["K_q: uint8<br/>(N, H, D)"]
        Quantize --> VQ["V_q: uint8<br/>(N, H, D)"]
        KQ --> Store["store_kvcache_unified_layout<br/>Triton Kernel"]
        VQ --> Store
        Store --> Cache["Cache: uint8<br/>(num_blocks, page_size, H, D)"]
    end
    
    subgraph "Load阶段 - Static模式"
        Cache --> View["view_kv_cache_for_kernels<br/>uint8 -> float8 view"]
        View --> CacheFP8["Cache: float8 view<br/>(num_blocks, page_size, H, D)"]
        Q["Q: bfloat16<br/>(num_seqs, num_heads, D)"] --> DecodeKernel
        CacheFP8 --> DecodeKernel["dllm_flash_attn_decode_kernel_bf16_q_fp8_kv<br/>TileLang Kernel"]
        KScale --> DecodeKernel
        VScale --> DecodeKernel
        DecodeKernel --> Output["Output: bfloat16<br/>(num_seqs, num_heads, D)"]
    end
    
    style UpdateScale fill:#fff9c4
    style Quantize fill:#ffebee
    style View fill:#ffebee
    style DecodeKernel fill:#e8f5e9
```

### 详细数据流图：Varlen模式（FP8，Unified/Distinct Layout）

```mermaid
flowchart LR
    subgraph "Store阶段"
        K1["K: bfloat16<br/>(N, H, D)"] --> UpdateScale["update_scales<br/>计算/更新scale"]
        V1["V: bfloat16<br/>(N, H, D)"] --> UpdateScale
        UpdateScale --> KScale["k_scale: float32<br/>(H)"]
        UpdateScale --> VScale["v_scale: float32<br/>(H)"]
        K1 --> Quantize["quantize_kv_for_store<br/>使用scale量化"]
        V1 --> Quantize
        KScale --> Quantize
        VScale --> Quantize
        Quantize --> KQ["K_q: uint8<br/>(N, H, D)"]
        Quantize --> VQ["V_q: uint8<br/>(N, H, D)"]
        KQ --> Store{Layout?}
        VQ --> Store
        Store -->|unified| StoreUnified["store_kvcache_unified_layout"]
        Store -->|distinct| StoreDistinct["store_kvcache_distinct_layout"]
        StoreUnified --> CacheU["Cache: uint8<br/>Unified: (num_blocks, page_size, H, D)"]
        StoreDistinct --> CacheD["Cache: uint8<br/>Distinct: k (num_blks, h, hdim//x, blk_sz, x)<br/>v (num_blks, h, hdim, blk_sz)"]
    end
    
    subgraph "Load阶段 - Varlen模式"
        CacheU --> LoadKernel
        CacheD --> LoadKernel["load_kvcache<br/>Triton Fused Kernel"]
        KNew["K_new: bfloat16<br/>(N_new, H, D)"] --> LoadKernel
        VNew["V_new: bfloat16<br/>(N_new, H, D)"] --> LoadKernel
        KScale --> LoadKernel
        VScale --> LoadKernel
        Metadata["attn_metadata<br/>block_tables, cu_seqlens, etc."] --> LoadKernel
        LoadKernel --> View["view_kv_cache_for_kernels<br/>uint8 -> float8 view"]
        View --> GatherDequant["load_kvcache_kernel_fp8_*<br/>gather + dequant + scale<br/>float8 -> bfloat16"]
        GatherDequant --> KComb["K_comb: bfloat16<br/>(total_len, H, D)"]
        GatherDequant --> VComb["V_comb: bfloat16<br/>(total_len, H, D)"]
        Q["Q: bfloat16<br/>(total_len, num_heads, D)"] --> FlashAttn
        KComb --> FlashAttn["flash_attn_varlen_func<br/>Flash Attention"]
        VComb --> FlashAttn
        FlashAttn --> Output["Output: bfloat16<br/>(total_len, num_heads, D)"]
    end
    
    style UpdateScale fill:#fff9c4
    style Quantize fill:#ffebee
    style View fill:#ffebee
    style GatherDequant fill:#ffebee
    style FlashAttn fill:#e8f5e9
```

### 关键数据类型转换总结表

| 阶段 | 操作 | 输入类型 | 输出类型 | 说明 |
|------|------|---------|---------|------|
| **Store (BF16)** | 直接存储 | `bfloat16 [N, H, D]` | `bfloat16 [num_blocks, page_size, H, D]` | 无需量化，直接存储 |
| **Store (FP8)** | quantize_kv_for_store | `bfloat16 [N, H, D]` + `float32 [H]` scale | `uint8 [N, H, D]` | 量化并存储为uint8 |
| **Store (FP8)** | 存储到cache | `uint8 [N, H, D]` | `uint8 [num_blocks, page_size, H, D]` | 存储为uint8格式 |
| **Load (Static FP8)** | view_kv_cache_for_kernels | `uint8 [num_blocks, page_size, H, D]` | `float8 view [num_blocks, page_size, H, D]` | 视图转换，不改变存储 |
| **Load (Static FP8)** | TileLang Kernel | `float8 view` + `float32 [H]` scale | `bfloat16 [num_seqs, num_heads, D]` | Kernel内反量化+scale |
| **Load (Varlen FP8)** | view_kv_cache_for_kernels | `uint8 [num_blocks, page_size, H, D]` | `float8 view [num_blocks, page_size, H, D]` | 视图转换 |
| **Load (Varlen FP8)** | Triton Fused Kernel | `float8 view` + `float32 [H]` scale | `bfloat16 [total_len, H, D]` | gather + dequant + scale |
| **Attention** | flash_attn_varlen_func | `bfloat16 [total_len, num_heads, D]` | `bfloat16 [total_len, num_heads, D]` | Flash Attention计算 |

### 路径选择决策表

| Layout | Decode Mode | 量化格式 | Store Kernel | Load Kernel | Attention Kernel |
|--------|-------------|---------|--------------|-------------|------------------|
| Unified | static | bf16 | `store_kvcache_unified_layout` → BF16 kernel | 无（直接使用cache） | `dllm_flash_attn_decode_kernel` (TileLang) |
| Unified | static | fp8 | `store_kvcache_unified_layout` → FP8 kernel | `view_kv_cache_for_kernels` | `dllm_flash_attn_decode_kernel_bf16_q_fp8_kv` (TileLang) |
| Unified | varlen | bf16 | `store_kvcache_unified_layout` → BF16 kernel | `load_kvcache_bf16` (Triton) | `flash_attn_varlen_func` |
| Unified | varlen | fp8 | `store_kvcache_unified_layout` → FP8 kernel | `load_kvcache_fp8` (Triton fused) | `flash_attn_varlen_func` |
| Distinct | varlen | bf16 | `store_kvcache_distinct_layout` → BF16 kernel | `load_kvcache_bf16` (Triton) | `flash_attn_varlen_func` |
| Distinct | varlen | fp8 | `store_kvcache_distinct_layout` → FP8 kernel | `load_kvcache_fp8` (Triton fused) | `flash_attn_varlen_func` |

**注意**：
- Distinct layout **总是**使用varlen模式（因为K的split layout不适合static模式）
- Static模式**仅支持**Unified layout
- FP8量化在static模式下，反量化在TileLang kernel内部完成
- FP8量化在varlen模式下，反量化在`load_kvcache`的Triton fused kernel中完成
