#!/usr/bin/zsh
export CUDA_VISIBLE_DEVICES=3,4

MODEL_ROOT_PATH=/data1/ckpts
MODEL_PATH=${MODEL_ROOT_PATH}/Dream-org/Dream-v0-Base-7B
LORA_PATH=${MODEL_ROOT_PATH}/SJTU-Deng-Lab/D2F_Dream_Base_7B_Lora
MODEL_TYPE=diffusion_lm
MODEL_NAME=dream

TP_SIZE=2
DP_SIZE=1

python -m d2f_engine.serve --model ${MODEL_PATH} \
    --use-lora --lora-path ${LORA_PATH} \
    --model-type ${MODEL_TYPE} \
    --tensor-parallel-size ${TP_SIZE} \
    --data-parallel-size ${DP_SIZE} \
    --model-name ${MODEL_NAME} \
    --port 8000