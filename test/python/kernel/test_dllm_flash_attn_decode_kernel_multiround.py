import os
import time
from pathlib import Path

import torch

from diffulex_kernel.python.dllm_flash_attn_kernels import dllm_flash_attn_decode_kernel
from test.python.kernel.test_dllm_flash_attn_decode_kernel import naive_sdpa_with_kvcache


def test_decode_multiround_context_len():
    """
    Test inference time and compilation behavior across different context_len values and num_seqs.
    This test verifies:
    1. Inference time for different context lengths and sequence counts
    2. Whether kernels are recompiled for different context_len values
    3. Block table configurations with trailing -1 entries
    """
    # Common parameters (same as test_decode_bf16_multi_seq)
    base_params = {
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "max_q_len": 64,
        "max_kv_len": 64,
        "page_block_size": 32,
        "diffusion_block_size": 32,
        "is_block_attn": False,
        "dtype": "bfloat16",
    }
    
    # Different sequence counts to test
    num_seqs_list = [1, 4, 8, 13, 14, 15, 16]
    
    # Different context lengths to test
    max_context_len = 2048
    context_lens = list(range(128, max_context_len + 1, 32))
    
    # Track compilation times and inference times
    # Key format: (num_seqs, context_len)
    compilation_times = {}
    inference_times = {}
    kernel_paths = {}
    kernel_instances = {}
    correctness_results = {}  # Track correctness verification results
    
    cuda_cache_dir = os.getenv("CUDA_CACHE_DIR", "./cuda_cache")
    cache_root = Path(cuda_cache_dir) / "test_dllm_flash_attn_decode_kernel_multiround"
    
    print("\n" + "=" * 80)
    print("Testing multiple num_seqs and context_len values")
    print(f"Testing num_seqs: {num_seqs_list}")
    print(f"Testing context_lens: {len(context_lens)} values from {context_lens[0]} to {context_lens[-1]}")
    print("=" * 80)
    
    # Test all combinations of num_seqs and context_len
    for num_seqs in num_seqs_list:
        # Calculate KV cache size based on max_context_len to ensure consistent allocation
        # across all tests for this num_seqs
        max_blocks_per_seq = (max_context_len + base_params["page_block_size"] - 1) // base_params["page_block_size"]
        max_seq_num_blocks = max_blocks_per_seq
        num_page_blocks = num_seqs * max_blocks_per_seq
        
        print(f"\n{'=' * 80}")
        print(f"Testing with num_seqs={num_seqs}")
        print(f"KV cache: max_seq_num_blocks={max_seq_num_blocks}, num_page_blocks={num_page_blocks}")
        print(f"{'=' * 80}")
        
        for context_len in context_lens:
            print(f"\n--- Testing num_seqs={num_seqs}, context_len={context_len} ---")
            
            # Check if kernel file already exists (indicates potential cache hit)
            case_dir = cache_root / (
                f"seq{num_seqs}_heads{base_params['num_heads']}_"
                f"kv{base_params['num_kv_heads']}_hd{base_params['head_dim']}_"
                f"ctx{context_len}_pbs{base_params['page_block_size']}_"
                f"dbs{base_params['diffusion_block_size']}_"
                f"block{int(base_params['is_block_attn'])}_dtype{base_params['dtype']}_"
                f"bm64_bn64_stg1_thr128_mq{base_params['max_q_len']}_mk{base_params['max_kv_len']}"
            )
            kernel_path = case_dir / "kernel.cu"
            
            kernel_existed_before = kernel_path.exists()
            kernel_mtime_before = kernel_path.stat().st_mtime if kernel_existed_before else None
            
            # Measure compilation + first inference time
            start_time = time.time()
            
            # Run the test (this includes kernel compilation if needed)
            # We'll create the kernel and run it to measure compilation time
            torch_dtype = getattr(torch, base_params["dtype"])
            device = "cuda"
            num_groups = base_params["num_heads"] // base_params["num_kv_heads"]
            total_q_len = num_seqs * base_params["diffusion_block_size"]
            total_kv_len = num_seqs * base_params["diffusion_block_size"]
            
            # Create kernel (this may trigger compilation)
            decode_kernel = dllm_flash_attn_decode_kernel(
                num_seqs,
                num_groups,
                num_page_blocks,
                total_q_len,
                total_kv_len,
                base_params["num_heads"],
                base_params["head_dim"],
                base_params["is_block_attn"],
                base_params["diffusion_block_size"],
                max_seq_num_blocks,
                base_params["page_block_size"],
                64,  # block_m
                64,  # block_n
                1,   # num_stages
                128, # num_threads
            )
        
            # Save kernel source
            kernel_source = decode_kernel.get_kernel_source()
            case_dir.mkdir(parents=True, exist_ok=True)
            kernel_path.write_text(kernel_source)
            
            # Prepare input tensors for first run
            q = torch.randn(total_q_len, base_params["num_heads"], base_params["head_dim"], 
                           dtype=torch_dtype, device=device)
            k = torch.randn(total_kv_len, base_params["num_kv_heads"], base_params["head_dim"], 
                           dtype=torch_dtype, device=device)
            v = torch.randn(total_kv_len, base_params["num_kv_heads"], base_params["head_dim"], 
                           dtype=torch_dtype, device=device)
            k_cache = torch.randn(num_page_blocks, base_params["page_block_size"], 
                                 base_params["num_kv_heads"], base_params["head_dim"], 
                                 dtype=torch_dtype, device=device)
            v_cache = torch.randn(num_page_blocks, base_params["page_block_size"], 
                                 base_params["num_kv_heads"], base_params["head_dim"], 
                                 dtype=torch_dtype, device=device)
            
            # Create block_tables with varying configurations
            # Some sequences will have trailing -1 entries even when context_len is sufficient
            block_tables = torch.zeros(num_seqs, max_seq_num_blocks, 
                                      dtype=torch.int32, device=device)
            # Calculate actual blocks needed for current context_len
            num_blocks_per_seq = (context_len + base_params["page_block_size"] - 1) // base_params["page_block_size"]
            
            for seq_idx in range(num_seqs):
                # Determine how many blocks to actually use for this sequence
                # For some sequences, use fewer blocks to create trailing -1 entries
                # Pattern: alternate between full blocks and partial blocks
                if seq_idx % 2 == 0:
                    # Even-indexed sequences: use all blocks needed
                    blocks_to_use = num_blocks_per_seq
                else:
                    # Odd-indexed sequences: use fewer blocks (leave some trailing -1)
                    # Use at least 1 block, but leave at least 1 trailing -1 if possible
                    blocks_to_use = max(1, num_blocks_per_seq - 1)
                
                # Fill in the blocks
                for block_idx in range(blocks_to_use):
                    block_tables[seq_idx, block_idx] = seq_idx * max_blocks_per_seq + block_idx
                
                # Set remaining blocks to -1 (invalid)
                for block_idx in range(blocks_to_use, max_seq_num_blocks):
                    block_tables[seq_idx, block_idx] = -1
            
            context_lens_tensor = torch.full((num_seqs,), context_len, 
                                            dtype=torch.int32, device=device)
            cu_seqlens_q = torch.arange(0, (num_seqs + 1) * base_params["diffusion_block_size"], 
                                       base_params["diffusion_block_size"], dtype=torch.int32, device=device)
            cu_seqlens_k = torch.arange(0, (num_seqs + 1) * base_params["diffusion_block_size"], 
                                       base_params["diffusion_block_size"], dtype=torch.int32, device=device)
        
            # First run (includes compilation if needed)
            _ = decode_kernel(
                q, k, v, k_cache, v_cache,
                block_tables,
                context_lens_tensor,
                cu_seqlens_q,
                cu_seqlens_k,
                base_params["max_q_len"],
            )
            torch.cuda.synchronize()
            
            compilation_time = time.time() - start_time
            key = (num_seqs, context_len)
            compilation_times[key] = compilation_time
            
            # Check if kernel was compiled (file was created, not just loaded from cache)
            # Note: This is a heuristic - the actual compilation happens when the kernel
            # is first called, and tilelang may have its own caching mechanism
            was_compiled = not kernel_existed_before
            
            kernel_paths[key] = str(kernel_path)
            
            # Count trailing -1 entries in block_tables
            trailing_neg_ones = 0
            for seq_idx in range(num_seqs):
                for block_idx in range(max_seq_num_blocks - 1, -1, -1):
                    if block_tables[seq_idx, block_idx].item() == -1:
                        trailing_neg_ones += 1
                    else:
                        break
            
            print(f"  Kernel path: {kernel_path}")
            print(f"  Kernel existed before: {kernel_existed_before}")
            print(f"  Was compiled: {was_compiled}")
            print(f"  Compilation + first inference time: {compilation_time:.4f}s")
            print(f"  Block table trailing -1 entries: {trailing_neg_ones}")
            
            # Measure pure inference time (warmup + actual measurement)
            # Warmup
            _ = decode_kernel(
                q, k, v, k_cache, v_cache,
                block_tables,
                context_lens_tensor,
                cu_seqlens_q,
                cu_seqlens_k,
                base_params["max_q_len"],
            )
            torch.cuda.synchronize()
            
            # Measure inference time
            num_iterations = 10
            start_time = time.time()
            for _ in range(num_iterations):
                _ = decode_kernel(
                    q, k, v, k_cache, v_cache,
                    block_tables,
                    context_lens_tensor,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    base_params["max_q_len"],
                )
            torch.cuda.synchronize()
            inference_time = (time.time() - start_time) / num_iterations
            inference_times[key] = inference_time
            
            print(f"  Average inference time ({num_iterations} iterations): {inference_time*1000:.4f}ms")
            
            # Verify correctness by comparing with reference implementation
            print(f"  Verifying correctness...")
            # Run kernel once more to get output for correctness verification
            output = decode_kernel(
                q, k, v, k_cache, v_cache,
                block_tables,
                context_lens_tensor,
                cu_seqlens_q,
                cu_seqlens_k,
                base_params["max_q_len"],
            )
            torch.cuda.synchronize()
            
            scale = 1.0 / (base_params["head_dim"] ** 0.5)
            ref_output = naive_sdpa_with_kvcache(
                q, k, v, k_cache, v_cache,
                block_tables, context_lens_tensor,
                cu_seqlens_q, cu_seqlens_k,
                scale, num_groups, base_params["page_block_size"],
            )
            
            try:
                torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-2)
                correctness_results[key] = True
                print(f"  ✓ Correctness check passed")
            except AssertionError as e:
                correctness_results[key] = False
                print(f"  ✗ Correctness check FAILED: {e}")
            
            # Store kernel instance for later use
            kernel_instances[key] = decode_kernel
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"{'Num Seqs':<12} {'Context Len':<15} {'Compiled':<10} {'Correct':<10} {'Compilation Time (s)':<20} {'Inference Time (ms)':<20}")
    print("-" * 100)
    for num_seqs in num_seqs_list:
        for context_len in context_lens:
            key = (num_seqs, context_len)
            if key in kernel_paths:
                was_compiled = kernel_paths[key] and Path(kernel_paths[key]).exists()
                is_correct = correctness_results.get(key, False)
                correct_str = "✓" if is_correct else "✗"
                print(f"{num_seqs:<12} {context_len:<15} {str(was_compiled):<10} {correct_str:<10} {compilation_times[key]:<20.4f} {inference_times[key]*1000:<20.4f}")
    
    print("\n" + "=" * 80)
    print("Analysis")
    print("=" * 80)
    
    # Check if kernels were recompiled for different (num_seqs, context_len) combinations
    unique_kernel_paths = set(kernel_paths.values())
    total_combinations = len(num_seqs_list) * len(context_lens)
    print(f"Number of unique kernel paths: {len(unique_kernel_paths)}")
    print(f"Number of (num_seqs, context_len) combinations tested: {total_combinations}")
    
    if len(unique_kernel_paths) == total_combinations:
        print("✓ Each (num_seqs, context_len) combination resulted in a unique kernel (expected behavior)")
    else:
        print(f"⚠ Some combinations shared the same kernel ({len(unique_kernel_paths)} unique kernels for {total_combinations} combinations)")
    
    # Check inference time scaling by num_seqs
    print(f"\nInference time scaling by num_seqs:")
    for num_seqs in num_seqs_list:
        seq_times = [inference_times[(num_seqs, ctx)] for ctx in context_lens if (num_seqs, ctx) in inference_times]
        if seq_times:
            base_time = seq_times[0]
            print(f"  num_seqs={num_seqs}:")
            for i, context_len in enumerate(context_lens):
                key = (num_seqs, context_len)
                if key in inference_times:
                    ratio = inference_times[key] / base_time
                    print(f"    context_len={context_len}: {ratio:.2f}x (vs context_len={context_lens[0]})")
    
    # Check inference time scaling by context_len
    print(f"\nInference time scaling by context_len:")
    for context_len in context_lens[::4]:  # Sample every 4th context_len to avoid too much output
        ctx_times = [inference_times[(ns, context_len)] for ns in num_seqs_list if (ns, context_len) in inference_times]
        if ctx_times:
            base_time = ctx_times[0]
            print(f"  context_len={context_len}:")
            for num_seqs in num_seqs_list:
                key = (num_seqs, context_len)
                if key in inference_times:
                    ratio = inference_times[key] / base_time
                    print(f"    num_seqs={num_seqs}: {ratio:.2f}x (vs num_seqs={num_seqs_list[0]})")
    
    # Check correctness summary
    print(f"\nCorrectness verification summary:")
    passed = sum(1 for v in correctness_results.values() if v)
    total = len(correctness_results)
    print(f"  Passed: {passed}/{total}")
    if passed < total:
        print(f"  Failed (num_seqs, context_len) combinations:")
        for key, is_correct in correctness_results.items():
            if not is_correct:
                num_seqs, context_len = key
                print(f"    - num_seqs={num_seqs}, context_len={context_len}")
    else:
        print("  ✓ All correctness checks passed!")


def test_decode_engine_like_scenarios():
    """
    Test decode kernel with scenarios that more closely match engine usage.
    This test simulates:
    1. Non-contiguous block_tables (like engine's prepare_block_tables)
    2. Variable cu_seqlens_k based on actual sequence lengths
    3. Memory reuse scenarios
    4. Different block_table patterns (some sequences with fewer blocks)
    """
    base_params = {
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "max_q_len": 64,
        "max_kv_len": 64,
        "page_block_size": 32,
        "diffusion_block_size": 32,
        "is_block_attn": False,
        "dtype": "bfloat16",
    }
    
    num_seqs_list = [1, 4, 8, 13, 14, 15, 16]
    context_lens_list = [128, 256, 512, 1024, 2048]
    
    torch_dtype = getattr(torch, base_params["dtype"])
    device = "cuda"
    num_groups = base_params["num_heads"] // base_params["num_kv_heads"]
    
    # Calculate maximum KV cache size to avoid recompilation
    max_num_seqs = max(num_seqs_list)
    max_context_len = max(context_lens_list)
    max_blocks_per_seq = (max_context_len + base_params["page_block_size"] - 1) // base_params["page_block_size"]
    max_num_page_blocks = max_num_seqs * max_blocks_per_seq
    
    # Setup cache directory for saving kernel sources
    cuda_cache_dir = os.getenv("CUDA_CACHE_DIR", "./cuda_cache")
    cache_root = Path(cuda_cache_dir) / "test_dllm_flash_attn_decode_kernel_multiround"
    
    # Create fixed-size KV cache (static allocation)
    print("\n" + "=" * 80)
    print("Testing engine-like scenarios")
    print(f"Using fixed large KV cache: num_page_blocks={max_num_page_blocks}")
    print("=" * 80)
    
    k_cache = torch.randn(max_num_page_blocks, base_params["page_block_size"], 
                         base_params["num_kv_heads"], base_params["head_dim"], 
                         dtype=torch_dtype, device=device)
    v_cache = torch.randn(max_num_page_blocks, base_params["page_block_size"], 
                         base_params["num_kv_heads"], base_params["head_dim"], 
                         dtype=torch_dtype, device=device)
    
    correctness_results = {}
    
    for num_seqs in num_seqs_list:
        print(f"\n{'=' * 80}")
        print(f"Testing with num_seqs={num_seqs}")
        print(f"{'=' * 80}")
        
        for context_len in context_lens_list:
            print(f"\n--- Testing num_seqs={num_seqs}, context_len={context_len} ---")
            
            # Simulate engine's prepare_block_tables behavior
            # Each sequence may have different number of blocks
            max_blocks_per_seq = (context_len + base_params["page_block_size"] - 1) // base_params["page_block_size"]
            max_seq_num_blocks = max_blocks_per_seq
            num_page_blocks = num_seqs * max_blocks_per_seq
            
            # Create block_tables like engine does: each seq may have different lengths
            block_tables_list = []
            for seq_idx in range(num_seqs):
                # Simulate variable block counts per sequence
                # Some sequences use fewer blocks (like engine scenarios)
                if seq_idx % 3 == 0:
                    # Every 3rd sequence uses all blocks
                    num_blocks = max_blocks_per_seq
                elif seq_idx % 3 == 1:
                    # Use 1 less block
                    num_blocks = max(1, max_blocks_per_seq - 1)
                else:
                    # Use 2 less blocks
                    num_blocks = max(1, max_blocks_per_seq - 2)
                
                seq_block_table = []
                for block_idx in range(num_blocks):
                    seq_block_table.append(seq_idx * max_blocks_per_seq + block_idx)
                # Engine pads with -1 to max_len
                seq_block_table.extend([-1] * (max_seq_num_blocks - num_blocks))
                block_tables_list.append(seq_block_table)
            
            block_tables = torch.tensor(block_tables_list, dtype=torch.int32, device=device)
            
            # Simulate engine's cu_seqlens calculation
            # In engine, cu_seqlens_k is based on actual sequence lengths (total_seqlen)
            # cu_seqlens_q is based on query lengths (total_seqlen - cached_num_tokens)
            total_q_len = num_seqs * base_params["diffusion_block_size"]
            total_kv_len = num_seqs * base_params["diffusion_block_size"]
            
            cu_seqlens_q = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
            cu_seqlens_k = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
            
            # Simulate variable sequence lengths (like in engine)
            for seq_idx in range(num_seqs):
                seqlen_q = base_params["diffusion_block_size"]  # Query length
                # KV length = context_len + seqlen_q (simulating cached + new tokens)
                seqlen_k = seqlen_q
                cu_seqlens_q[seq_idx + 1] = cu_seqlens_q[seq_idx] + seqlen_q
                cu_seqlens_k[seq_idx + 1] = cu_seqlens_k[seq_idx] + seqlen_k
            
            # Adjust total lengths based on actual cu_seqlens
            total_q_len = cu_seqlens_q[-1].item()
            total_kv_len = cu_seqlens_k[-1].item()
            
            # Prepare tensors
            q = torch.randn(total_q_len, base_params["num_heads"], base_params["head_dim"], 
                           dtype=torch_dtype, device=device)
            k = torch.randn(total_kv_len, base_params["num_kv_heads"], base_params["head_dim"], 
                           dtype=torch_dtype, device=device)
            v = torch.randn(total_kv_len, base_params["num_kv_heads"], base_params["head_dim"], 
                           dtype=torch_dtype, device=device)
            # Use the fixed-size KV cache (already allocated above)
            
            context_lens_tensor = torch.full((num_seqs,), context_len, 
                                            dtype=torch.int32, device=device)
            
            # Create kernel (use max_num_page_blocks for KV cache size)
            decode_kernel = dllm_flash_attn_decode_kernel(
                num_seqs,
                num_groups,
                max_num_page_blocks,  # Use fixed max size
                total_q_len,
                total_kv_len,
                base_params["num_heads"],
                base_params["head_dim"],
                base_params["is_block_attn"],
                base_params["diffusion_block_size"],
                max_seq_num_blocks,
                base_params["page_block_size"],
                64,  # block_m
                64,  # block_n
                1,   # num_stages
                128, # num_threads
            )
            
            # Save kernel source
            case_dir = cache_root / (
                f"seq{num_seqs}_heads{base_params['num_heads']}_"
                f"kv{base_params['num_kv_heads']}_hd{base_params['head_dim']}_"
                f"ctx{context_len}_pbs{base_params['page_block_size']}_"
                f"dbs{base_params['diffusion_block_size']}_"
                f"block{int(base_params['is_block_attn'])}_dtype{base_params['dtype']}_"
                f"bm64_bn64_stg1_thr128_mq{base_params['max_q_len']}_mk{base_params['max_kv_len']}"
            )
            kernel_path = case_dir / "kernel.cu"
            kernel_source = decode_kernel.get_kernel_source()
            case_dir.mkdir(parents=True, exist_ok=True)
            kernel_path.write_text(kernel_source)
            print(f"  Kernel saved to: {kernel_path}")
            
            # Test with memory reuse (simulate engine's behavior)
            # Run multiple times to check for memory corruption
            outputs = []
            for run_idx in range(3):
                output = decode_kernel(
                    q, k, v, k_cache, v_cache,
                    block_tables,
                    context_lens_tensor,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    base_params["max_q_len"],
                )
                torch.cuda.synchronize()
                outputs.append(output.clone())
            
            # Verify consistency across runs
            consistent = True
            for i in range(1, len(outputs)):
                if not torch.allclose(outputs[0], outputs[i], atol=1e-5, rtol=1e-5):
                    consistent = False
                    max_diff = (outputs[0] - outputs[i]).abs().max().item()
                    print(f"  ✗ Output inconsistency detected in run {i}: max_diff={max_diff:.6f}")
                    break
            
            if not consistent:
                correctness_results[(num_seqs, context_len)] = False
                continue
            
            # Verify correctness against reference
            scale = 1.0 / (base_params["head_dim"] ** 0.5)
            ref_output = naive_sdpa_with_kvcache(
                q, k, v, k_cache, v_cache,
                block_tables, context_lens_tensor,
                cu_seqlens_q, cu_seqlens_k,
                scale, num_groups, base_params["page_block_size"],
            )
            
            try:
                torch.testing.assert_close(outputs[0], ref_output, atol=1e-2, rtol=1e-2)
                correctness_results[(num_seqs, context_len)] = True
                print(f"  ✓ Correctness check passed")
            except AssertionError as e:
                correctness_results[(num_seqs, context_len)] = False
                abs_diff = (outputs[0] - ref_output).abs()
                max_diff = abs_diff.max().item()
                mean_diff = abs_diff.mean().item()
                print(f"  ✗ Correctness check FAILED: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
                print(f"    Error: {str(e)[:200]}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Engine-like Test Summary")
    print("=" * 80)
    passed = sum(1 for v in correctness_results.values() if v)
    total = len(correctness_results)
    print(f"  Passed: {passed}/{total}")
    if passed < total:
        print(f"  Failed (num_seqs, context_len) combinations:")
        for key, is_correct in correctness_results.items():
            if not is_correct:
                num_seqs, context_len = key
                print(f"    - num_seqs={num_seqs}, context_len={context_len}")
    else:
        print("  ✓ All correctness checks passed!")


if __name__ == "__main__":
    # tilelang.testing.main()
    # test_decode_multiround_context_len()
    # print("\n\n")
    test_decode_engine_like_scenarios()