#!/usr/bin/zsh
TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES=7 python examples/test_dllm_decoding_kernel.py 2>&1 | tee log/test_dvllm_dllm_decoding_kernel.log