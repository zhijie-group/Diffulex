#!/usr/bin/zsh
SCRIPT=scripts/test_dvllm_dream.sh
OUTPUT=~/workspace-2/D2F/log/profiles/nsys_prof_dvllm_dream
nsys profile --trace=cuda,osrt,nvtx,cudnn,cublas --export sqlite --force-overwrite true -o ${OUTPUT} ${SCRIPT}
nsys export --type json --force-overwrite=true -o ${OUTPUT}.json ${OUTPUT}.sqlite