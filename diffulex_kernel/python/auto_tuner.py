import itertools

def build_configs():
    BLOCK_M_LIST = [64, 128, 256]
    BLOCK_N_LIST = [64, 128, 256]
    NUM_STAGES_LIST = [0, 1, 2]
    NUM_THREADS_LIST = [128, 256]
    CONFIGS = list(
        itertools.product(
            BLOCK_M_LIST,
            BLOCK_N_LIST,
            NUM_STAGES_LIST,
            NUM_THREADS_LIST,
        )
    )

    return [
        {
            "BLOCK_M": c[0],
            "BLOCK_N": c[1],
            "NUM_STAGES": c[2],
            "NUM_THREADS": c[3],
        } for c in CONFIGS
    ]


def build_linear_configs():
    """Autotune configs for TileLang linear/GEMM-style kernels.

    Notes:
    - Keys intentionally match the linear kernel function kwargs in `linear_kernels.py`
      (lowercase: block_M/block_N/block_K/num_stages/threads).
    - Keep the search space modest; these kernels are instantiated for many (M,N,K) shapes.
    """
    BLOCK_M_LIST = [32, 64, 128]
    BLOCK_N_LIST = [64, 128]
    BLOCK_K_LIST = [64, 128]
    NUM_STAGES_LIST = [2, 3]
    THREADS_LIST = [128, 256]

    CONFIGS = list(
        itertools.product(
            BLOCK_M_LIST,
            BLOCK_N_LIST,
            BLOCK_K_LIST,
            NUM_STAGES_LIST,
            THREADS_LIST,
        )
    )

    return [
        {
            "block_M": c[0],
            "block_N": c[1],
            "block_K": c[2],
            "num_stages": c[3],
            "threads": c[4],
        }
        for c in CONFIGS
    ]