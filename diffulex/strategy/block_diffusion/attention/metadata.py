import torch

from dataclasses import dataclass

from diffulex.attention.metadata import AttnMetaDataBase


@dataclass
class BlockDiffusionAttnMetaData(AttnMetaDataBase):
    seq_lens: list[int] = None
    seq_lens_ts: torch.Tensor | None = None
    block_diffusion_pp: bool = False
    block_mask: list[torch.Tensor] | None = None
    

BLOCK_DIFFUSION_ATTN_METADATA = BlockDiffusionAttnMetaData()

def fetch_block_diffusion_attn_metadata() -> BlockDiffusionAttnMetaData:
    return BLOCK_DIFFUSION_ATTN_METADATA

def set_block_diffusion_attn_metadata() -> None:
    # TODO
    global BLOCK_DIFFUSION_ATTN_METADATA
    BLOCK_DIFFUSION_ATTN_METADATA = BlockDiffusionAttnMetaData()

def reset_block_diffusion_attn_metadata() -> None:
    global BLOCK_DIFFUSION_ATTN_METADATA
    BLOCK_DIFFUSION_ATTN_METADATA = BlockDiffusionAttnMetaData()