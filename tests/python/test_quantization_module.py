import pytest
import torch


def test_kv_cache_strategy_registry_bf16_aliases():
    from diffulex.utils.quantization import create_kv_cache_strategy

    s1 = create_kv_cache_strategy("bf16")
    assert getattr(s1, "kv_cache_format", None) == "bf16"
    assert s1.requires_kv_cache_scales is False

    # Currently routed to BF16 kernels for compatibility.
    s2 = create_kv_cache_strategy("fp16")
    assert getattr(s2, "kv_cache_format", None) == "bf16"

    s3 = create_kv_cache_strategy("fp32")
    assert getattr(s3, "kv_cache_format", None) == "bf16"


def test_attn_q_strategy_registry_and_factory():
    from types import SimpleNamespace
    from diffulex.utils.quantization import (
        QuantizationStrategyFactory,
        get_attn_q_strategy,
        create_attn_q_strategy,
    )

    # Registry creation works
    s_bf16 = create_attn_q_strategy("bf16")
    assert s_bf16.attn_q_format == "bf16"

    s_fp8 = create_attn_q_strategy("fp8")
    assert s_fp8.attn_q_format == "fp8"

    # Factory wiring: enable fp8 Q and ensure it lands in context
    cfg = SimpleNamespace(kv_cache_dtype="bf16", attn_q_dtype="fp8")
    QuantizationStrategyFactory.create_from_config(cfg)
    active = get_attn_q_strategy()
    assert active is not None
    assert active.attn_q_format == "fp8"


@pytest.mark.skipif(
    not (hasattr(torch, "float8_e4m3fn") or hasattr(torch, "float8_e4m3fnuz")),
    reason="This torch build does not expose float8 dtypes required by FP8 strategy.",
)
def test_kv_cache_fp8_strategy_metadata_and_views():
    from diffulex.utils.quantization import create_kv_cache_strategy
    from diffulex.attention.metadata import AttnMetaDataBase

    s = create_kv_cache_strategy("fp8")
    assert s.kv_cache_format == "fp8"
    assert s.requires_kv_cache_scales is True

    md = AttnMetaDataBase()
    k_scale = torch.ones((8,), dtype=torch.float32)
    v_scale = torch.ones((8,), dtype=torch.float32) * 2
    s.maybe_set_attn_metadata_scales(md, k_scale=k_scale, v_scale=v_scale)
    assert md.k_scale is k_scale
    assert md.v_scale is v_scale

    with pytest.raises(ValueError):
        s.maybe_set_attn_metadata_scales(AttnMetaDataBase(), k_scale=None, v_scale=None)

    # uint8 storage -> float8 view for kernels
    cache_u8 = torch.empty((16,), dtype=torch.uint8)
    cache_view = s.view_kv_cache_for_kernels(cache_u8)
    assert cache_view.dtype != torch.uint8


