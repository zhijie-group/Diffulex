<img src=./assets/imgs/diffulex_design.png />

<div align="center">

# Diffulex

[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white)](https://discord.gg/NSa9WH4EKu)

</div>

Diffulex is a Paged Attention-based dLLM accelerated decoding inference framework that is easy to develop and extensible. The design maximizes hiding the complexity of underlying KV Cache management, parallel strategy scheduling, and memory optimization. By providing a clean and unified API interface along with flexible inference strategy configurations (e.g., D2F, Block Diffusion, Fast-dLLM), Diffulex allows developers to focus on model inference logic and business requirements while maintaining production-level inference performance and resource utilization efficiency.

## Latest News
- 12/22/2025 ✨: We are excited to announce that Diffulex, a Paged Attention-based dLLM accelerated decoding inference framework, is now open source and available to the public!

## Tested Devices
Although Diffulex aims to be portable across a range of Devices, it has been specifically tested and validated on the following devices: for NVIDIA GPUs, this includes the H200, A100, RTX 4090, RTX 3090.

## Installation
### Method 1: Install with Pip

The only way to get started is to install from source:

```bash
uv pip install -e .
```

## Quick Start

Here's a simple example to get started with Diffulex:

```python
from diffulex import Diffulex, SamplingParams
from transformers import AutoTokenizer

# Initialize the Diffulex engine
model_path = "/path/to/your/model"
llm = Diffulex(
    model_path,
    model_name="fast_dllm_v2",  # or "dream", "llada", etc.
    tensor_parallel_size=1,
    data_parallel_size=1,
    gpu_memory_utilization=0.25,
    max_model_len=2048,
    decoding_strategy="block_diffusion",  # or "d2f", "fast_dllm"
    mask_token_id=151665,  # model-specific mask token ID
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Set sampling parameters
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=256,
)

# Prepare prompts
prompts = [
    "Question: What is the capital of France? Answer:",
    "Question: Explain quantum computing in simple terms. Answer:",
]

# Generate responses
outputs = llm.generate(prompts, sampling_params)

# Process results
for output in outputs:
    print(f"Generated text: {output['text']}")
    print(f"Number of diffusion steps: {output['n_diff_steps']}")
    print(f"Token IDs: {output['token_ids']}")
```

For more examples, check out the [examples](examples/) directory.

## Upcoming Features

Check our [Diffulex v0.0.1 release plan](https://github.com/zhijie-group/Diffulex/issues/14) for upcoming features.

## Join the Discussion

Welcome to join our Discord community for discussions, support, and collaboration!

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20Us-blue?logo=discord&style=for-the-badge)](https://discord.gg/NSa9WH4EKu)

## Acknowledgments

We would like to express our gratitude to [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm), which serves as the primary codebase foundation for this project, and [vLLM](https://github.com/vllm-project/vllm), from which we draw the core architectural concepts, particularly the Paged Attention mechanism. The initial version of this project was mainly developed by [Yijie Jin](https://github.com/drewjin0827) with supervision from Prof. [Zhijie Deng](https://thudzj.github.io) at Shanghai Jiao Tong University. 