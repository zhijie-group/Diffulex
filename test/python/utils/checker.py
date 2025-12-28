def CHECK_D2F_SLOT_MAPPING(seqs, slot_mapping):
    # check slot mapping layout
    start_idx = 0
    for seq in seqs:
        cur_ref_slot_mapping = []
        for idx in range(seq.num_diffusion_blocks):
            if seq.active_blocks[idx]:
                padding_num_tokens = (seq.num_diffusion_blocks - idx) * seq.diffusion_block_size
                cur_ref_slot_mapping.extend([-1] * padding_num_tokens)
                break
            elif seq.to_cache_blocks[idx]:
                cur_ref_slot_mapping.extend([0] * seq.diffusion_block_size)
        cur_slot_mapping = slot_mapping[start_idx:start_idx + len(cur_ref_slot_mapping)]
        for slot, ref_slot in zip(cur_slot_mapping, cur_ref_slot_mapping):
            try:
                if ref_slot == -1:
                    assert slot == -1
                elif ref_slot == 0:
                    assert slot != -1
                elif ref_slot is not None:
                    assert slot is not None
            except AssertionError:
                raise ValueError(f"Slot mapping mismatch: {slot} != {ref_slot}. "
                                    f"Check the implementation of prepare_decode.\n"
                                    f"slot_mapping: {cur_slot_mapping}\n"
                                    f"ref_slot_mapping: {cur_ref_slot_mapping}\n"
                                    f"diff: {[s - r for s, r in zip(cur_slot_mapping, cur_ref_slot_mapping)]}")
        start_idx += len(cur_ref_slot_mapping)


def CHECK_FLASH_ATTN_PREFILL(
    q, k, v, 
    cu_seqlens_q, 
    cu_seqlens_k, 
    max_seqlen_q, 
    prefill_kernel,
    diffusion_block_size: int = 32,
    is_block_attn: bool = False,
):
    """
    Verify prefill kernel correctness by comparing with PyTorch's scaled_dot_product_attention.
    
    Args:
        q: Query tensor [total_q_len, num_heads, head_dim]
        k: Key tensor [total_kv_len, num_kv_heads, head_dim]
        v: Value tensor [total_kv_len, num_kv_heads, head_dim]
        cu_seqlens_q: Cumulative sequence lengths for queries
        cu_seqlens_k: Cumulative sequence lengths for keys/values
        max_seqlen_q: Maximum sequence length for queries
        prefill_kernel: The kernel function to test
        diffusion_block_size: Size of diffusion blocks for block attention
        is_block_attn: Whether this is block attention mode
    """
    import torch
    import torch.nn.functional as F
    from einops import rearrange
    
    # Run kernel
    kernel_output = prefill_kernel(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q)
    
    # Compute reference output using PyTorch's SDPA
    head_dim = q.shape[2]
    scale = 1.0 / (head_dim ** 0.5)
    num_seqs = len(cu_seqlens_q) - 1
    
    gt_output = torch.zeros_like(q)
    for seq_idx in range(num_seqs):
        q_start = cu_seqlens_q[seq_idx].item()
        q_end = cu_seqlens_q[seq_idx + 1].item()
        kv_start = cu_seqlens_k[seq_idx].item()
        kv_end = cu_seqlens_k[seq_idx + 1].item()
        
        q_seq = q[q_start:q_end]
        k_seq = k[kv_start:kv_end]
        v_seq = v[kv_start:kv_end]
        
        q_len = q_seq.shape[0]
        kv_len = k_seq.shape[0]
        
        # Reshape for SDPA: [1, num_heads, seq_len, head_dim]
        q_sdpa = rearrange(q_seq, 's h d -> 1 h s d')
        k_sdpa = rearrange(k_seq, 's h d -> 1 h s d')
        v_sdpa = rearrange(v_seq, 's h d -> 1 h s d')
        
        if not is_block_attn:
            # Standard attention
            attn_out = F.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                dropout_p=0.0,
                is_causal=False,
                scale=scale,
                enable_gqa=True,
            )
        else:
            # Block attention with mask
            block_mask = torch.zeros((1, 1, q_len, kv_len), dtype=q.dtype, device=q.device).bool()
            num_diffusion_blocks = (kv_len + diffusion_block_size - 1) // diffusion_block_size
            for block_idx in range(num_diffusion_blocks):
                block_start = block_idx * diffusion_block_size
                block_end = min(block_start + diffusion_block_size, kv_len)
                block_mask[..., block_start:block_end, :block_end] = True
            
            attn_out = F.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                attn_mask=block_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=scale,
                enable_gqa=True,
            )
        
        gt_output[q_start:q_end] = rearrange(attn_out, '1 h s d -> s h d').to(gt_output.dtype)
    
    # Compare results
    atol = 1e-2
    rtol = 1e-2
    try:
        torch.testing.assert_close(
            kernel_output, 
            gt_output, 
            atol=atol, 
            rtol=rtol,
            msg="Kernel output does not match reference implementation"
        )
    except AssertionError as e:
        # Compute error statistics for debugging
        abs_diff = torch.abs(kernel_output - gt_output)
        max_diff = torch.max(abs_diff).item()
        mean_diff = torch.mean(abs_diff).item()
        rel_diff = torch.abs((kernel_output - gt_output) / (gt_output + 1e-8))
        max_rel_diff = torch.max(rel_diff).item()
        mean_rel_diff = torch.mean(rel_diff).item()
        
        # Count elements that exceed tolerance
        total_elements = kernel_output.numel()
        # Elements that exceed absolute tolerance
        exceeds_atol = (abs_diff > atol)
        num_exceeds_atol = exceeds_atol.sum().item()
        # Elements that exceed relative tolerance
        exceeds_rtol = (rel_diff > rtol)
        num_exceeds_rtol = exceeds_rtol.sum().item()
        # Elements that exceed either tolerance
        exceeds_tolerance = exceeds_atol | exceeds_rtol
        num_exceeds_tolerance = exceeds_tolerance.sum().item()
        pct_exceeds_tolerance = (num_exceeds_tolerance / total_elements * 100) if total_elements > 0 else 0
        
        raise AssertionError(
            f"Prefill kernel verification failed!\n"
            f"Max absolute difference: {max_diff:.6f}\n"
            f"Mean absolute difference: {mean_diff:.6f}\n"
            f"Max relative difference: {max_rel_diff:.6f}\n"
            f"Mean relative difference: {mean_rel_diff:.6f}\n"
            f"Total elements: {total_elements}\n"
            f"Elements exceeding absolute tolerance (atol={atol}): {num_exceeds_atol} ({num_exceeds_atol/total_elements*100:.2f}%)\n"
            f"Elements exceeding relative tolerance (rtol={rtol}): {num_exceeds_rtol} ({num_exceeds_rtol/total_elements*100:.2f}%)\n"
            f"Elements exceeding either tolerance: {num_exceeds_tolerance} ({pct_exceeds_tolerance:.2f}%)\n"
            f"Kernel output shape: {kernel_output.shape}\n"
            f"Reference output shape: {gt_output.shape}\n"
            f"Original error: {str(e)}"
        )


def CHECK_FLASH_ATTN_DECODE(
    q, k, v,
    k_cache, v_cache,
    block_tables,
    context_lens,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    decode_kernel,
    scale: float,
    num_groups: int,
    page_block_size: int,
    diffusion_block_size: int = 32,
    is_block_attn: bool = False,
):
    """
    Verify decode kernel correctness by comparing with reference implementation.
    This function mimics engine-like scenarios with memory reuse testing.
    
    Args:
        q: Query tensor [total_q_len, num_heads, head_dim]
        k: Key tensor [total_kv_len, num_kv_heads, head_dim]
        v: Value tensor [total_kv_len, num_kv_heads, head_dim]
        k_cache: KV cache for keys [num_page_blocks, page_block_size, num_kv_heads, head_dim]
        v_cache: KV cache for values [num_page_blocks, page_block_size, num_kv_heads, head_dim]
        block_tables: Block tables [num_seqs, max_seq_num_blocks]
        context_lens: Context lengths for each sequence [num_seqs]
        cu_seqlens_q: Cumulative sequence lengths for queries
        cu_seqlens_k: Cumulative sequence lengths for keys/values
        max_seqlen_q: Maximum sequence length for queries
        decode_kernel: The kernel function to test
        scale: Attention scale factor
        num_groups: Number of GQA groups (num_heads // num_kv_heads)
        page_block_size: Size of page blocks in KV cache
        diffusion_block_size: Size of diffusion blocks for block attention
        is_block_attn: Whether this is block attention mode
    """
    import torch
    from test.python.kernel.test_dllm_flash_attn_decode_kernel import naive_sdpa_with_kvcache
    
    # Test with memory reuse (simulate engine's behavior)
    # Run multiple times to check for memory corruption
    outputs = []
    for run_idx in range(3):
        output = decode_kernel(
            q, k, v, k_cache, v_cache,
            block_tables,
            context_lens,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
        )
        torch.cuda.synchronize()
        outputs.append(output.clone())
    
    # Verify consistency across runs
    consistent = True
    for i in range(1, len(outputs)):
        if not torch.allclose(outputs[0], outputs[i], atol=1e-5, rtol=1e-5):
            consistent = False
            max_diff = (outputs[0] - outputs[i]).abs().max().item()
            raise AssertionError(
                f"Output inconsistency detected in run {i}: max_diff={max_diff:.6f}. "
                f"This indicates potential memory corruption or non-deterministic behavior."
            )
    
    # Use the first output for comparison
    kernel_output = outputs[0]
    
    # Compute reference output using naive_sdpa_with_kvcache (same as test file)
    gt_output = naive_sdpa_with_kvcache(
        q, k, v, k_cache, v_cache,
        block_tables, context_lens,
        cu_seqlens_q, cu_seqlens_k,
        scale, num_groups, page_block_size,
    )
    
    # Compare results (using same tolerance as test file)
    atol = 1e-2
    rtol = 1e-2
    try:
        torch.testing.assert_close(
            kernel_output,
            gt_output,
            atol=atol,
            rtol=rtol,
            msg="Decode kernel output does not match reference implementation"
        )
    except AssertionError as e:
        # Compute error statistics for debugging
        abs_diff = torch.abs(kernel_output - gt_output)
        max_diff = torch.max(abs_diff).item()
        mean_diff = torch.mean(abs_diff).item()
        rel_diff = torch.abs((kernel_output - gt_output) / (gt_output + 1e-8))
        max_rel_diff = torch.max(rel_diff).item()
        mean_rel_diff = torch.mean(rel_diff).item()
        
        # Count elements that exceed tolerance
        total_elements = kernel_output.numel()
        # Elements that exceed absolute tolerance
        exceeds_atol = (abs_diff > atol)
        num_exceeds_atol = exceeds_atol.sum().item()
        # Elements that exceed relative tolerance
        exceeds_rtol = (rel_diff > rtol)
        num_exceeds_rtol = exceeds_rtol.sum().item()
        # Elements that exceed either tolerance
        exceeds_tolerance = exceeds_atol | exceeds_rtol
        num_exceeds_tolerance = exceeds_tolerance.sum().item()
        pct_exceeds_tolerance = (num_exceeds_tolerance / total_elements * 100) if total_elements > 0 else 0
        
        # Save test case data for debugging
        import os
        from pathlib import Path
        import pickle
        from datetime import datetime
        
        save_dir = Path(os.getenv("TEST_CASE_SAVE_DIR", "./failed_test_cases"))
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        case_name = f"decode_kernel_failure_{timestamp}"
        case_dir = save_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        
        # Save all input and output tensors
        test_data = {
            'inputs': {
                'q': q.cpu(),
                'k': k.cpu(),
                'v': v.cpu(),
                'k_cache': k_cache.cpu(),
                'v_cache': v_cache.cpu(),
                'block_tables': block_tables.cpu(),
                'context_lens': context_lens.cpu(),
                'cu_seqlens_q': cu_seqlens_q.cpu(),
                'cu_seqlens_k': cu_seqlens_k.cpu(),
            },
            'outputs': {
                'kernel_output': kernel_output.cpu(),
                'gt_output': gt_output.cpu(),
                'abs_diff': abs_diff.cpu(),
                'rel_diff': rel_diff.cpu(),
            },
            'parameters': {
                'max_seqlen_q': max_seqlen_q,
                'scale': scale,
                'num_groups': num_groups,
                'page_block_size': page_block_size,
                'diffusion_block_size': diffusion_block_size,
                'is_block_attn': is_block_attn,
                'atol': atol,
                'rtol': rtol,
            },
            'statistics': {
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'max_rel_diff': max_rel_diff,
                'mean_rel_diff': mean_rel_diff,
                'total_elements': total_elements,
                'num_exceeds_atol': num_exceeds_atol,
                'num_exceeds_rtol': num_exceeds_rtol,
                'num_exceeds_tolerance': num_exceeds_tolerance,
                'pct_exceeds_tolerance': pct_exceeds_tolerance,
            },
            'shapes': {
                'q_shape': list(q.shape),
                'k_shape': list(k.shape),
                'v_shape': list(v.shape),
                'k_cache_shape': list(k_cache.shape),
                'v_cache_shape': list(v_cache.shape),
                'block_tables_shape': list(block_tables.shape),
                'kernel_output_shape': list(kernel_output.shape),
                'gt_output_shape': list(gt_output.shape),
            },
        }
        
        # Save as pickle
        with open(case_dir / "test_data.pkl", "wb") as f:
            pickle.dump(test_data, f)
        
        # Save kernel source (same as test file)
        kernel_path = None
        try:
            kernel_source = decode_kernel.get_kernel_source()
            kernel_path = case_dir / "kernel.cu"
            kernel_path.write_text(kernel_source)
        except Exception as kernel_err:
            # If kernel source is not available, log but don't fail
            pass
        
        # Generate a Python script to reproduce the test case
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        repro_script = f'''"""
Auto-generated test case from failed CHECK_FLASH_ATTN_DECODE.
Generated at: {timestamp_str}

To use this test case:
1. Load the data: test_data = pickle.load(open("test_data.pkl", "rb"))
2. Move tensors to device: q = test_data['inputs']['q'].to(device), etc.
3. Call your kernel with the loaded inputs
"""
import torch
import pickle
from pathlib import Path

# Load test data
case_dir = Path(__file__).parent
with open(case_dir / "test_data.pkl", "rb") as f:
    test_data = pickle.load(f)

# Extract inputs
q = test_data['inputs']['q']
k = test_data['inputs']['k']
v = test_data['inputs']['v']
k_cache = test_data['inputs']['k_cache']
v_cache = test_data['inputs']['v_cache']
block_tables = test_data['inputs']['block_tables']
context_lens = test_data['inputs']['context_lens']
cu_seqlens_q = test_data['inputs']['cu_seqlens_q']
cu_seqlens_k = test_data['inputs']['cu_seqlens_k']

# Extract parameters
params = test_data['parameters']
max_seqlen_q = params['max_seqlen_q']
scale = params['scale']
num_groups = params['num_groups']
page_block_size = params['page_block_size']
diffusion_block_size = params['diffusion_block_size']
is_block_attn = params['is_block_attn']

# Extract expected outputs for comparison
gt_output = test_data['outputs']['gt_output']

# Print test case info
print("Test Case Information:")
q_shape = test_data['shapes']['q_shape']
k_shape = test_data['shapes']['k_shape']
v_shape = test_data['shapes']['v_shape']
print(f"  Shapes: q={{q_shape}}, k={{k_shape}}, v={{v_shape}}")
print(f"  Parameters: scale={{scale}}, num_groups={{num_groups}}, page_block_size={{page_block_size}}")
max_diff_val = test_data['statistics']['max_diff']
num_mismatches = test_data['statistics']['num_exceeds_tolerance']
print(f"  Statistics: max_diff={{max_diff_val:.6f}}, num_mismatches={{num_mismatches}}")

# TODO: Add your kernel call here
# kernel_output = your_kernel(q, k, v, k_cache, v_cache, block_tables, context_lens, 
#                             cu_seqlens_q, cu_seqlens_k, max_seqlen_q)
# torch.testing.assert_close(kernel_output, gt_output, atol=params['atol'], rtol=params['rtol'])
'''
        
        with open(case_dir / "reproduce_test.py", "w") as f:
            f.write(repro_script)
        
        # Save error summary
        error_summary = f"""Test Case Failure Summary
Generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Shapes:
  q: {test_data['shapes']['q_shape']}
  k: {test_data['shapes']['k_shape']}
  v: {test_data['shapes']['v_shape']}
  k_cache: {test_data['shapes']['k_cache_shape']}
  v_cache: {test_data['shapes']['v_cache_shape']}
  block_tables: {test_data['shapes']['block_tables_shape']}
  kernel_output: {test_data['shapes']['kernel_output_shape']}
  gt_output: {test_data['shapes']['gt_output_shape']}

Parameters:
  max_seqlen_q: {max_seqlen_q}
  scale: {scale}
  num_groups: {num_groups}
  page_block_size: {page_block_size}
  diffusion_block_size: {diffusion_block_size}
  is_block_attn: {is_block_attn}
  atol: {atol}
  rtol: {rtol}

Statistics:
  Max absolute difference: {max_diff:.6f}
  Mean absolute difference: {mean_diff:.6f}
  Max relative difference: {max_rel_diff:.6f}
  Mean relative difference: {mean_rel_diff:.6f}
  Total elements: {total_elements}
  Elements exceeding absolute tolerance: {num_exceeds_atol} ({num_exceeds_atol/total_elements*100:.2f}%)
  Elements exceeding relative tolerance: {num_exceeds_rtol} ({num_exceeds_rtol/total_elements*100:.2f}%)
  Elements exceeding either tolerance: {num_exceeds_tolerance} ({pct_exceeds_tolerance:.2f}%)
"""
        
        with open(case_dir / "error_summary.txt", "w") as f:
            f.write(error_summary)
        
        save_info = f"\n\nTest case data saved to: {case_dir}\n"
        save_info += f"  - test_data.pkl: All input/output tensors and metadata\n"
        save_info += f"  - reproduce_test.py: Script to reproduce the test case\n"
        save_info += f"  - error_summary.txt: Summary of the failure\n"
        if kernel_path is not None:
            save_info += f"  - kernel.cu: CUDA kernel source code\n"
        
        # Show mismatched elements layout
        mismatch_info = ""
        if num_exceeds_tolerance > 0:
            # Get indices of mismatched elements
            mismatch_indices = torch.nonzero(exceeds_tolerance, as_tuple=False)
            num_to_show = min(50, num_exceeds_tolerance)  # Show at most 50 mismatches
            
            mismatch_info = f"\n\nMismatched elements (showing first {num_to_show} of {num_exceeds_tolerance}):\n"
            mismatch_info += "-" * 100 + "\n"
            mismatch_info += f"{'Index':<30} {'Kernel Value':<20} {'Ref Value':<20} {'Abs Diff':<15} {'Rel Diff':<15}\n"
            mismatch_info += "-" * 100 + "\n"
            
            for i in range(num_to_show):
                idx = mismatch_indices[i]
                idx_tuple = tuple(idx.tolist())
                
                kernel_val = kernel_output[idx_tuple].item()
                gt_val = gt_output[idx_tuple].item()
                abs_err = abs_diff[idx_tuple].item()
                rel_err = rel_diff[idx_tuple].item()
                
                mismatch_info += (
                    f"{str(idx_tuple):<30} "
                    f"{kernel_val:>19.6f} "
                    f"{gt_val:>19.6f} "
                    f"{abs_err:>14.6f} "
                    f"{rel_err:>14.6f}\n"
                )
            
            if num_exceeds_tolerance > num_to_show:
                mismatch_info += f"\n... and {num_exceeds_tolerance - num_to_show} more mismatches\n"
            
            # Show distribution of mismatches by dimension
            if len(kernel_output.shape) >= 2:
                mismatch_info += f"\nMismatch distribution by dimensions:\n"
                for dim_idx in range(len(kernel_output.shape)):
                    dim_mismatches = exceeds_tolerance.sum(dim=tuple(j for j in range(len(kernel_output.shape)) if j != dim_idx))
                    mismatch_info += f"  Dim {dim_idx} (size {kernel_output.shape[dim_idx]}): {dim_mismatches.tolist()}\n"
        
        raise AssertionError(
            f"Decode kernel verification failed!\n"
            f"Max absolute difference: {max_diff:.6f}\n"
            f"Mean absolute difference: {mean_diff:.6f}\n"
            f"Max relative difference: {max_rel_diff:.6f}\n"
            f"Mean relative difference: {mean_rel_diff:.6f}\n"
            f"Total elements: {total_elements}\n"
            f"Elements exceeding absolute tolerance (atol={atol}): {num_exceeds_atol} ({num_exceeds_atol/total_elements*100:.2f}%)\n"
            f"Elements exceeding relative tolerance (rtol={rtol}): {num_exceeds_rtol} ({num_exceeds_rtol/total_elements*100:.2f}%)\n"
            f"Elements exceeding either tolerance: {num_exceeds_tolerance} ({pct_exceeds_tolerance:.2f}%)\n"
            f"Kernel output shape: {kernel_output.shape}\n"
            f"Reference output shape: {gt_output.shape}\n"
            f"{mismatch_info}"
            f"{save_info}"
            f"Original error: {str(e)}"
        ) 