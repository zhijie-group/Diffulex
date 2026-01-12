from . import metadata
from .metadata import set_fetch_fn_for_attn_metadata, AttnMetaDataBase

# Create a proxy that dynamically accesses fetch_attn_metadata from the metadata module
# This ensures we always get the current value, not a stale copy from __init__.py
class _FetchAttnMetadataProxy:
    """Proxy object that dynamically accesses fetch_attn_metadata from metadata module."""
    def __call__(self, *args, **kwargs):
        return metadata.fetch_attn_metadata(*args, **kwargs)
    
    def __repr__(self):
        return repr(metadata.fetch_attn_metadata)

fetch_attn_metadata = _FetchAttnMetadataProxy()


def __getattr__(name):
    """Lazy import to avoid circular deps during module init."""
    if name == "Attention":
        try:
            from .attn_impl import Attention
            return Attention
        except Exception as e:
            raise ImportError(f"Failed to import diffulex.attention.attn_impl.Attention: {e}")
    if name == "fetch_attn_metadata":
        return metadata.fetch_attn_metadata
    raise AttributeError(f"module {__name__} has no attribute {name}")