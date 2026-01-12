#!/usr/bin/zsh
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/test_dream_dvllm_human_eval.py 2>&1 | tee log/test_dvllm_dream_human_eval.log
