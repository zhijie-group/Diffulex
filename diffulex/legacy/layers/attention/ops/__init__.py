from diffulex.legacy.layers.attention.ops.triton_decode_attn_clm import causal_lm_decode_attention_fwd as causal_lm_flash_decoding
from diffulex.legacy.layers.attention.ops.triton_decode_attn_dlm import diffusion_lm_flash_decoding, CHECK_ATTENTION
from diffulex.legacy.layers.attention.ops.chunked_prefill_decoding_unified_kernel import chunked_prefill_paged_decode as diffusion_lm_parallel_flash_decoding
from diffulex.legacy.layers.attention.ops.kv_cache_kernels import (
    store_kvcache_distinct_layout, store_kvcache_unified_layout, load_kvcache,
    CHECK_STORING, CHECK_LOADING
)