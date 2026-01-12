#!/usr/bin/zsh

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true

export HF_HOME="$(pwd)/cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_METRICS_CACHE="$HF_HOME/metrics"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m diffulex_bench.main \
    --config custom_configs/d2f_dream_eval_gsm8k.yml \
    2>&1 | tee log/d2f_dream_eval_gsm8k.log