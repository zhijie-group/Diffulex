import os
import pickle
from pathlib import Path

import torch
import tilelang
import tilelang.testing

from diffulex_kernel.python.dllm_flash_attn_kernels import dllm_flash_attn_decode_kernel
from test.python.utils.checker import CHECK_FLASH_ATTN_DECODE


def get_failed_test_cases_dir():
    """Get the directory containing failed test cases."""
    default_dir = Path(__file__).parent.parent.parent.parent / "failed_test_cases"
    return Path(os.getenv("TEST_CASE_SAVE_DIR", str(default_dir)))


def find_failed_test_cases():
    """Find all failed test case directories."""
    test_cases_dir = get_failed_test_cases_dir()
    if not test_cases_dir.exists():
        return []
    
    test_cases = []
    for case_dir in test_cases_dir.iterdir():
        if case_dir.is_dir() and case_dir.name.startswith("decode_kernel_failure_"):
            test_data_path = case_dir / "test_data.pkl"
            if test_data_path.exists():
                test_cases.append(case_dir)
    
    return sorted(test_cases)


def load_test_case(case_dir: Path):
    """Load a test case from directory."""
    test_data_path = case_dir / "test_data.pkl"
    if not test_data_path.exists():
        raise FileNotFoundError(f"test_data.pkl not found in {case_dir}")
    
    with open(test_data_path, "rb") as f:
        test_data = pickle.load(f)
    
    return test_data


def run_test_case_from_saved_data(case_dir: Path):
    """Run a test case from saved data."""
    # Load test data
    test_data = load_test_case(case_dir)
    
    # Extract inputs and move to device
    device = "cuda"
    q = test_data['inputs']['q'].to(device)
    k = test_data['inputs']['k'].to(device)
    v = test_data['inputs']['v'].to(device)
    k_cache = test_data['inputs']['k_cache'].to(device)
    v_cache = test_data['inputs']['v_cache'].to(device)
    block_tables = test_data['inputs']['block_tables'].to(device)
    context_lens = test_data['inputs']['context_lens'].to(device)
    cu_seqlens_q = test_data['inputs']['cu_seqlens_q'].to(device)
    cu_seqlens_k = test_data['inputs']['cu_seqlens_k'].to(device)
    
    # Extract parameters
    params = test_data['parameters']
    max_seqlen_q = params['max_seqlen_q']
    scale = params['scale']
    num_groups = params['num_groups']
    page_block_size = params['page_block_size']
    diffusion_block_size = params['diffusion_block_size']
    is_block_attn = params['is_block_attn']
    
    # Extract shapes to infer kernel parameters
    q_shape = test_data['shapes']['q_shape']
    k_shape = test_data['shapes']['k_shape']
    k_cache_shape = test_data['shapes']['k_cache_shape']
    block_tables_shape = test_data['shapes']['block_tables_shape']
    
    # Infer kernel parameters from shapes
    total_q_len = q_shape[0]
    total_kv_len = k_shape[0]
    num_heads = q_shape[1]
    num_kv_heads = k_shape[1]
    head_dim = q_shape[2]
    num_seqs = len(cu_seqlens_q) - 1
    num_page_blocks = k_cache_shape[0]
    max_seq_num_blocks = block_tables_shape[1]
    
    # Default kernel tuning parameters (can be overridden if saved in test_data)
    block_m = 64
    block_n = 64
    num_stages = 1
    num_threads = 128
    
    # Build kernel
    decode_kernel = dllm_flash_attn_decode_kernel(
        num_seqs,
        num_groups,
        num_page_blocks,
        total_q_len,
        total_kv_len,
        num_heads,
        head_dim,
        is_block_attn,
        diffusion_block_size,
        max_seq_num_blocks,
        page_block_size,
        block_m,
        block_n,
        num_stages,
        num_threads,
    )
    
    # Verify using CHECK_FLASH_ATTN_DECODE (it will run the kernel and verify)
    CHECK_FLASH_ATTN_DECODE(
        q, k, v,
        k_cache, v_cache,
        block_tables,
        context_lens,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        decode_kernel,
        scale,
        num_groups,
        page_block_size,
        diffusion_block_size,
        is_block_attn,
    )
    
    print(f"Test case {case_dir.name} passed! Shape: {q.shape}")


def test_all_failed_cases():
    """Test all failed test cases found in the failed_test_cases directory."""
    test_cases = find_failed_test_cases()
    
    if not test_cases:
        print("No failed test cases found. Skipping test.")
        return
    
    print(f"Found {len(test_cases)} failed test case(s) to verify:")
    for case_dir in test_cases:
        print(f"  - {case_dir.name}")
    
    # Run each test case
    for case_dir in test_cases:
        print(f"\n{'='*80}")
        print(f"Testing case: {case_dir.name}")
        print(f"{'='*80}")
        
        try:
            run_test_case_from_saved_data(case_dir)
        except Exception as e:
            print(f"Test case {case_dir.name} FAILED with error:")
            print(f"  {type(e).__name__}: {str(e)}")
            raise


# Generate individual test functions for each failed test case
def generate_test_functions():
    """Dynamically generate test functions for each failed test case."""
    test_cases = find_failed_test_cases()
    
    for idx, case_dir in enumerate(test_cases):
        case_name = case_dir.name.replace("decode_kernel_failure_", "").replace("-", "_").replace(".", "_")
        test_func_name = f"test_case_{case_name}"
        
        # Create a closure with the case_dir captured
        def make_test_func(case_path):
            def test_func():
                run_test_case_from_saved_data(case_path)
            return test_func
        
        # Create and register the test function
        test_func = make_test_func(case_dir)
        test_func.__name__ = test_func_name
        test_func.__doc__ = f"Test case from {case_dir.name}"
        globals()[test_func_name] = test_func


# Generate test functions at module load time
generate_test_functions()


if __name__ == "__main__":
    tilelang.testing.main()

