import os
import csv
import time

import pandas as pd

from datasets import load_dataset
from viztracer import VizTracer
from transformers import AutoTokenizer

from diffulex.legacy import LLM, SamplingParams


def summarize_profiling(csv_path: str) -> dict:
    totals = {}
    total_nums = {}
    avgs = {}
    with open(csv_path, 'r', newline='') as f:
        reader = csv.dictReader(f)
        for row in reader:
            for k, v in row.items():
                try:
                    val = float(v)
                except ValueError:
                    continue
                if val != 0.0:
                    total_nums[k] = total_nums.get(k, 0) + 1
                totals[k] = totals.get(k, 0.0) + val
    print(pd.DataFrame([totals]).T)
    for k, v in totals.items():
        if k in total_nums and total_nums[k] > 0:
            avgs[k] = v / total_nums[k]
        else:
            avgs[k] = 0.0
    print(pd.DataFrame([avgs]).T)


if __name__ == "__main__":
    WEIGHT_DIR = "/data1/ckpts"
    DATA_DIR = "/data1/LargeData"
    model = f"{WEIGHT_DIR}/Dream-org/Dream-v0-Base-7B"
    LLM = LLM(
        model, 
        lora_path=f"{WEIGHT_DIR}/SJTU-Deng-Lab/D2F_Dream_Base_7B_Lora",
        use_lora=True,
        model_name="dream", 
        model_type="diffusion_lm", 
        enforce_eager=True, 
        data_parallel_size=1,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.30,
        max_num_batched_tokens=1024,
        max_num_seqs=1,
        max_model_len=1024,
        accept_threshold=0.95,
        complete_threshold=0.9,
        add_new_block_threshold=0.1,
        kvcache_block_size=32,
        kv_cache_layout="unified"
    )
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
    
    dataset = load_dataset(f"{DATA_DIR}/openai/openai_humaneval")["test"]['prompt'][:]
    prompts = [tokenizer.bos_token + p for p in dataset]
    
    output_file = "log/profiles/perf_dvllm_dream_7B.json"
    if os.path.exists(output_file):
        os.remove(output_file)
    # with VizTracer(output_file=output_file, file_info=True) as tracer:
    #     outputs = llm.generate(prompts[:5], sampling_params)
    time.sleep(20)
    s = time.time()
    outputs = LLM.generate(prompts[:], sampling_params)
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
        resp = prompts[idx] + "\n-----<Start-of-Response>-----\n" + o["text"]
        print(f"[Prompt {idx} Result]\n{resp}\n")