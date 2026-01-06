import os
import csv
import time

import pandas as pd

from tqdm import tqdm
from datasets import load_dataset
from viztracer import VizTracer
from transformers import AutoTokenizer

from diffulex import Diffulex, SamplingParams
    

FEW_SHOTS="""
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nA: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.\n\nQ: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nA: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.\n\nQ: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nA: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.\n\nQ: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nA: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.\n\nQ: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nA: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.\n\nQ: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nA: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.\n\nQ: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nA: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.\n\nQ: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nA: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.\n\n
"""


if __name__ == "__main__":
    model = "/data1/ckpts/Dream-org/Dream-v0-Base-7B"
    LLM = Diffulex(
        model,
        lora_path="/data1/ckpts/SJTU-Deng-Lab/D2F_Dream_Base_7B_Lora",
        use_lora=True,
        model_name="dream", 
        enforce_eager=True, 
        data_parallel_size=8,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.25,
        max_num_batched_tokens=2048,
        max_num_seqs=20,
        max_model_len=2048,
        accept_threshold=0.95,
        complete_threshold=0.9,
        add_new_block_threshold=0.1,
        kv_cache_layout="unified",
        decoding_strategy="d2f"
    )
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
    
    dataset = load_dataset(
        "gsm8k", "main", split="test")["question"][:10]
    prompts = [tokenizer.bos_token + FEW_SHOTS + p for p in tqdm(dataset)]
    
    output_file = "log/profiles/perf_dvllm_dream_7B.json"
    if os.path.exists(output_file):
        os.remove(output_file)
    # with VizTracer(output_file=output_file, file_info=True) as tracer:
    #     outputs = llm.generate(prompts[:5], sampling_params)
    # time.sleep(60)
    s = time.time()
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