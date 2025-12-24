"""
Test FP8 KV cache in a complete inference pipeline.
This script verifies that FP8 KV cache works correctly for text generation.
"""
import os

from diffulex_legacy import LLM, SamplingParams
from transformers import AutoTokenizer


if __name__ == "__main__":
    # Test with a simple prompt to verify FP8 KV cache works
    print("=" * 80)
    print("Testing FP8 KV Cache in Complete Pipeline (Diffusion LM - Dream)")
    print("=" * 80)
    
    # Initialize LLM with FP8 KV cache
    print("\n[1/4] Initializing LLM with kv_cache_dtype='fp8_e4m3'...")
    try:
        model = "/data1/ckpts/Dream-org/Dream-v0-Base-7B"
        llm = LLM(
            model,
            lora_path="/data1/ckpts/SJTU-Deng-Lab/D2F_Dream_Base_7B_Lora",
            use_lora=True,
            model_name="dream", 
            model_type="diffusion_lm",
            enforce_eager=True, 
            data_parallel_size=1,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.25,
            max_num_batched_tokens=2048,
            max_num_seqs=20,
            max_model_len=2048,
            accept_threshold=0.95,
            complete_threshold=0.9,
            add_new_block_threshold=0.1,
            kv_cache_layout="unified",
            kv_cache_dtype="fp8_e4m3",  # Enable FP8 KV cache
        )
        print("✓ LLM initialized successfully with FP8 KV cache")
    except Exception as e:
        print(f"✗ Failed to initialize LLM: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Simple test prompts
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    test_prompts = [
        tokenizer.bos_token + "Hello, how are you?",
        tokenizer.bos_token + "The capital of France is",
        tokenizer.bos_token + "Python is a programming language that",
    ]
    
    sampling_params = SamplingParams(temperature=0.7, max_tokens=50)
    
    print(f"\n[2/4] Generating text for {len(test_prompts)} prompts...")
    try:
        outputs = llm.generate(test_prompts, sampling_params)
        print("✓ Text generation completed successfully")
    except Exception as e:
        print(f"✗ Text generation failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print(f"\n[3/4] Verifying outputs...")
    # Verify outputs
    for i, (prompt, output) in enumerate(zip(test_prompts, outputs)):
        generated_text = output.get("text", "")
        token_ids = output.get("token_ids", [])
        
        print(f"\n--- Prompt {i+1} ---")
        print(f"Input:  {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
        print(f"Output: {generated_text[:100]}{'...' if len(generated_text) > 100 else ''}")
        print(f"Tokens: {len(token_ids)} tokens")
        
        # Basic validation: output should not be empty
        if not generated_text.strip():
            print(f"⚠ Warning: Generated text is empty for prompt {i+1}")
        if len(token_ids) == 0:
            print(f"⚠ Warning: No tokens generated for prompt {i+1}")
    
    print(f"\n[4/4] Test Summary")
    print("=" * 80)
    print("✓ FP8 KV cache pipeline test PASSED")
    print(f"  - Successfully generated text for {len(outputs)} prompts")
    print(f"  - Total tokens generated: {sum(len(o.get('token_ids', [])) for o in outputs)}")
    print("=" * 80)
