"""
LM Eval Model - Diffulex integration with lm-evaluation-harness
"""

import logging
import time
import json
from typing import List, Optional, Tuple, Type, TypeVar, Union
from pathlib import Path

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from diffulex import Diffulex, SamplingParams
from diffulex_bench.runner import BenchmarkRunner
from diffulex.logger import setup_logger, get_logger

T = TypeVar("T", bound="LM")
eval_logger = logging.getLogger(__name__)


@register_model("diffulex")
class DiffulexLM(LM):
    """
    Diffulex model integration for lm-evaluation-harness
    """
    
    def __init__(
        self,
        pretrained: str,
        batch_size: Optional[Union[int, str]] = 1,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, type]] = "auto",
        max_new_tokens: Optional[int] = 256,
        max_length: Optional[int] = 2048,
        add_bos_token: Optional[bool] = False,
        trust_remote_code: Optional[bool] = True,
        temperature: Optional[float] = 0.0,
        model_name: Optional[str] = "dream",
        decoding_strategy: Optional[str] = "d2f",
        mask_token_id: Optional[int] = 151666,
        tensor_parallel_size: Optional[int] = 1,
        data_parallel_size: Optional[int] = 1,
        gpu_memory_utilization: Optional[float] = 0.9,
        max_model_len: Optional[int] = 2048,
        max_num_batched_tokens: Optional[int] = 4096,
        max_num_seqs: Optional[int] = 128,
        use_lora: Optional[bool] = False,
        lora_path: Optional[str] = "",
        enforce_eager: Optional[bool] = False,
        kv_cache_layout: Optional[str] = "unified",
        accept_threshold: Optional[float] = 0.9,
        complete_threshold: Optional[float] = 0.95,
        add_new_block_threshold: Optional[float] = 0.1,
        diffusion_block_size: Optional[int] = 32,
        save_dir: Optional[str] = None,
        wait_ready: Optional[bool] = True,
        **kwargs,
    ) -> None:
        super().__init__()
        
        # Setup logger
        self.logger = get_logger(__name__)
        
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, (int, str))
        
        self.pretrained = pretrained
        self.batch_size_per_gpu = batch_size
        if isinstance(batch_size, str):
            self.batch_size_per_gpu = int(batch_size)
        
        self.max_length = max_length
        self.add_bos_token = add_bos_token
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.save_dir = save_dir
        
        # Diffulex-specific parameters
        self.model_name = model_name
        self.decoding_strategy = decoding_strategy
        self.mask_token_id = mask_token_id
        
        # Statistics tracking
        self.total_generated_tokens = 0
        self.total_nfe = 0  # Number of Forward Evaluations (diffusion steps)
        self.total_generation_time = 0.0
        self.total_samples = 0
        self.all_generation_times = []
        self.all_nfe = []
        self.all_tokens = []
        
        # Initialize Diffulex runner
        self.runner = BenchmarkRunner(
            model_path=pretrained,
            tokenizer_path=pretrained,
            wait_ready=wait_ready,
            model_name=model_name,
            decoding_strategy=decoding_strategy,
            mask_token_id=mask_token_id,
            tensor_parallel_size=tensor_parallel_size,
            data_parallel_size=data_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            use_lora=use_lora,
            lora_path=lora_path if use_lora else "",
            enforce_eager=enforce_eager,
            kv_cache_layout=kv_cache_layout,
            accept_threshold=accept_threshold,
            complete_threshold=complete_threshold,
            add_new_block_threshold=add_new_block_threshold,
            diffusion_block_size=diffusion_block_size,
        )
        
        self.tokenizer = self.runner.tokenizer
        
        # Create sampling params
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        
        self.logger.success("Diffulex engine initialized successfully")
    
    @property
    def batch_size(self):
        return self.batch_size_per_gpu
    
    @property
    def device(self):
        return "cuda"  # Diffulex manages device internally
    
    @property
    def rank(self):
        return 0
    
    @property
    def world_size(self):
        return 1
    
    def tok_decode(self, tokens, skip_special_tokens=True):
        """Decode tokens to text"""
        if isinstance(tokens, list) and len(tokens) > 0 and isinstance(tokens[0], list):
            return [self.tokenizer.decode(t, skip_special_tokens=skip_special_tokens) for t in tokens]
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
    
    def tok_encode(self, text, add_special_tokens=True):
        """Encode text to tokens"""
        return self.tokenizer(
            text, return_tensors="pt", add_special_tokens=add_special_tokens
        ).input_ids
    
    @classmethod
    def create_from_arg_string(
        cls: Type[T], arg_string: str, additional_config: Optional[dict] = None
    ) -> T:
        """
        Creates an instance of the LM class using the given argument string and additional config.
        
        Args:
            arg_string: A string containing arguments in the format key1=value1,key2=value2
            additional_config: Optional dictionary containing additional configuration parameters
            
        Returns:
            Instance of the LM class
        """
        additional_config = {} if additional_config is None else additional_config
        args = utils.simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(**args, **args2)
    
    def apply_chat_template(
        self, chat_history, add_generation_prompt: bool = True
    ) -> str:
        """
        Apply a chat template to a list of chat history between user and model.
        """
        chat_templated = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )
        return chat_templated
    
    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")
    
    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False):
        """
        Generate text until stopping conditions are met.
        
        Args:
            requests: List of generation requests
            disable_tqdm: Whether to disable progress bar
            
        Returns:
            List of generated texts
        """
        self.logger.info(f"Processing {len(requests)} generation requests...")
        
        # Prepare prompts
        prompts = []
        gen_args = []
        
        for req in requests:
            prompt = req.arguments[0]
            if self.add_bos_token and self.tokenizer.bos_token:
                prompt = self.tokenizer.bos_token + prompt
            prompts.append(prompt)
            gen_args.append(req.arguments[1] if len(req.arguments) > 1 else {})
        
        # Run generation
        start_time = time.time()
        outputs = self.runner.generate(
            prompts,
            self.sampling_params,
            use_tqdm=not disable_tqdm,
        )
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # Extract results and accumulate statistics
        results = []
        num_tokens = 0
        num_nfe = 0
        
        for output in outputs:
            text = output.get('text', '')
            results.append(text)
            
            token_ids = output.get('token_ids', [])
            n_diff_steps = output.get('n_diff_steps', 0)
            
            num_tokens += len(token_ids)
            num_nfe += n_diff_steps
            
            self.all_generation_times.append(total_time / len(outputs) if outputs else 0)
            self.all_nfe.append(n_diff_steps)
            self.all_tokens.append(len(token_ids))
        
        # Update statistics
        self.total_samples += len(requests)
        self.total_generated_tokens += num_tokens
        self.total_nfe += num_nfe
        self.total_generation_time += total_time
        
        # Log statistics
        if self.total_samples > 0:
            avg_tokens = self.total_generated_tokens / self.total_samples
            avg_nfe = self.total_nfe / self.total_samples
            avg_time = self.total_generation_time / self.total_samples
            throughput = num_tokens / total_time if total_time > 0 else 0
            
            self.logger.info(
                f"Generated {len(results)} samples | "
                f"Tokens: {num_tokens} | "
                f"NFE: {num_nfe} | "
                f"Time: {total_time:.2f}s | "
                f"Throughput: {throughput:.2f} tok/s"
            )
        
        # Save statistics if save_dir is provided
        if self.save_dir is not None:
            self._save_statistics()
        
        return results
    
    def _save_statistics(self):
        """Save statistics to file"""
        import os
        os.makedirs(self.save_dir, exist_ok=True)
        
        stats = {
            'total_samples': self.total_samples,
            'total_tokens': self.total_generated_tokens,
            'total_nfe': self.total_nfe,
            'total_time': self.total_generation_time,
            'avg_tokens_per_sample': self.total_generated_tokens / self.total_samples if self.total_samples > 0 else 0,
            'avg_nfe_per_sample': self.total_nfe / self.total_samples if self.total_samples > 0 else 0,
            'avg_time_per_sample': self.total_generation_time / self.total_samples if self.total_samples > 0 else 0,
            'throughput_tok_s': self.total_generated_tokens / self.total_generation_time if self.total_generation_time > 0 else 0,
            'nfe_per_token': self.total_nfe / self.total_generated_tokens if self.total_generated_tokens > 0 else 0,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        stats_path = os.path.join(self.save_dir, 'diffulex_stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Statistics saved to {stats_path}")
    
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """
        Compute log-likelihood of continuations given contexts.
        
        Note: This is a placeholder implementation. Full loglikelihood computation
        for diffusion models requires special handling.
        """
        self.logger.warning(
            "loglikelihood computation for diffusion models is not fully implemented. "
            "Returning placeholder values."
        )
        return [(0.0, False) for _ in requests]
    
    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        """Compute log-likelihood of sequences."""
        raise NotImplementedError(
            "loglikelihood_rolling is not implemented for diffusion models"
        )

