#!/usr/bin/zsh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python examples/test_dream_dvllm_gsm8k.py 2>&1 | tee log/test_dvllm_dream_gsm8k.log