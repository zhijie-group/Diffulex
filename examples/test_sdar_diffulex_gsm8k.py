import os
import csv
import time

import pandas as pd

from tqdm import tqdm
from datasets import load_dataset
from viztracer import VizTracer
from transformers import AutoTokenizer

from diffulex import Diffulex, SamplingParams


FEW_SHOTS = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nQuestion: Jen and Tyler are gymnasts practicing flips. Jen is practicing the triple-flip while Tyler is practicing the double-flip. Jen did sixteen triple-flips during practice. Tyler flipped in the air half the number of times Jen did. How many double-flips did Tyler do?\nAnswer:<|im_end|>\n<|im_start|>assistant\nJen did 16 triple-flips, so she did 16 * 3 = <<16*3=48>>48 flips.\nTyler did half the number of flips, so he did 48 / 2 = <<48/2=24>>24 flips.\nA double flip has two flips, so Tyler did 24 / 2 = <<24/2=12>>12 double-flips.\n#### 12<|im_end|>\n<|im_start|>user\nQuestion: Four people in a law firm are planning a party. Mary will buy a platter of pasta for $20 and a loaf of bread for $2. Elle and Andrea will split the cost for buying 4 cans of soda which cost $1.50 each, and chicken wings for $10. Joe will buy a cake that costs $5. How much more will Mary spend than the rest of the firm put together?\nAnswer:<|im_end|>\n<|im_start|>assistant\nMary will spend $20 + $2 = $<<20+2=22>>22.\nElle and Andrea will spend $1.5 x 4 = $<<1.5*4=6>>6 for the soda.\nElle and Andrea will spend $6 + $10 = $<<6+10=16>>16 for the soda and chicken wings.\nElle, Andrea, and Joe together will spend $16 + $5 = $<<16+5=21>>21.\nSo, Mary will spend $22 - $21 = $<<22-21=1>>1 more than all of them combined.\n#### 1<|im_end|>\n<|im_start|>user\nQuestion: A charcoal grill burns fifteen coals to ash every twenty minutes of grilling. The grill ran for long enough to burn three bags of coals. Each bag of coal contains 60 coals. How long did the grill run?\nAnswer:<|im_end|>\n<|im_start|>assistant\nThe grill burned 3 * 60 = <<3*60=180>>180 coals.\nIt takes 20 minutes to burn 15 coals, so the grill ran for 180 / 15 * 20 = <<180/15*20=240>>240 minutes.\n#### 240<|im_end|>\n<|im_start|>user\nQuestion: A bear is preparing to hibernate for the winter and needs to gain 1000 pounds. At the end of summer, the bear feasts on berries and small woodland animals. During autumn, it devours acorns and salmon. It gained a fifth of the weight it needed from berries during summer, and during autumn, it gained twice that amount from acorns. Salmon made up half of the remaining weight it had needed to gain. How many pounds did it gain eating small animals?\nAnswer:<|im_end|>\n<|im_start|>assistant\nThe bear gained 1 / 5 * 1000 = <<1/5*1000=200>>200 pounds from berries.\nIt gained 2 * 200 = <<2*200=400>>400 pounds from acorns.\nIt still needed 1000 - 200 - 400 = <<1000-200-400=400>>400 pounds.\nThus, it gained 400 / 2 = <<400/2=200>>200 pounds from salmon.\nTherefore, the bear gained 400 - 200 = <<400-200=200>>200 pounds from small animals.\n#### 200<|im_end|>\n<|im_start|>user\nQuestion: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\nAnswer:<|im_end|>\n<|im_start|>assistant\n"
# FEW_SHOTS = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"

if __name__ == "__main__":
    PROFILE = True
    # model = "/root/data/ckpts/JetLM/SDAR-1.7B-Chat-b32"
    model = "/data1/ckpts/JetLM/SDAR-1.7B-Chat-b32"
    dataset = load_dataset("gsm8k", "main", split="test")["question"][:1]
    LLM = Diffulex(
        model,
        use_lora=False,
        model_name="sdar", 
        enforce_eager=True, 
        data_parallel_size=1,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.3,
        max_num_batched_tokens=2048,
        max_num_seqs=20,
        max_model_len=2048,
        kv_cache_layout="unified",
        decoding_strategy="block_diffusion",
        mask_token_id=151669,
    )
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
    
    prompts = [
        FEW_SHOTS + f"<|im_start|>user\nQuestion: {question}\nAnswer:<|im_end|>\n<|im_start|>assistant\n"
        for question in tqdm(dataset)
    ]
    s = time.time()
    if PROFILE:
        output_file = "log/profiles/perf_dvllm_dream_7B.json"
        if os.path.exists(output_file):
            os.remove(output_file)
        
        with VizTracer(output_file=output_file, file_info=True) as tracer:
            outputs = LLM.generate(prompts, sampling_params)
    else:
        outputs = LLM.generate(prompts, sampling_params)
    e = time.time()
    print("=*=" * 30, 
          "\nProfiling Results\n", 
          "=*=" * 30, "\n"
          f"Generated {len(outputs)} outputs.\n"
          f"Total tokens: {sum(len(o['token_ids']) for o in outputs)}\n"
          f"Total time: {e - s:.2f} seconds.\n"
          f"Avg TPS: {sum(len(o['token_ids']) for o in outputs) / (e - s):.2f} tok/s.\n"
          f"AVG Number of Diffusion Steps: {sum(o['n_diff_steps'] for o in outputs) / len(outputs):.2f}\n",
          "=*=" * 30)
    for idx, o in enumerate(outputs):
        print("\n", "=*=" * 30)
        print(f"[Prompt {idx} Result] \n{prompts[idx] + "\n-----<Start-of-Response>-----\n" + o['text']}\n")