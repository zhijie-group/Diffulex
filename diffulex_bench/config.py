"""
Benchmark Configuration - Configuration management with separated engine and eval configs
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import json
import yaml


@dataclass
class EngineConfig:
    """
    Engine configuration - Parameters for Diffulex engine initialization
    """
    # Model and weights
    model_path: str
    tokenizer_path: Optional[str] = None
    model_name: str = "dream"  # Options: dream, sdar, fast_dllm_v2
    decoding_strategy: str = "d2f"  # Options: d2f, block_diffusion, fast_dllm
    mask_token_id: int = 151666
    
    # LoRA configuration
    use_lora: bool = False
    lora_path: str = ""
    
    # Parallelism configuration
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    
    # Memory and capacity configuration
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 2048
    max_num_batched_tokens: int = 4096
    max_num_seqs: int = 128
    
    # Engine behavior configuration
    enforce_eager: bool = False
    kv_cache_layout: str = "unified"  # Options: unified, distinct
    
    # D2F-specific configuration
    accept_threshold: float = 0.9
    complete_threshold: float = 0.95
    add_new_block_threshold: float = 0.1
    diffusion_block_size: int = 32
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EngineConfig":
        """Create engine configuration from dictionary"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def get_diffulex_kwargs(self) -> Dict[str, Any]:
        """Get arguments to pass to Diffulex engine"""
        return {
            'model_name': self.model_name,
            'decoding_strategy': self.decoding_strategy,
            'mask_token_id': self.mask_token_id,
            'tensor_parallel_size': self.tensor_parallel_size,
            'data_parallel_size': self.data_parallel_size,
            'gpu_memory_utilization': self.gpu_memory_utilization,
            'max_model_len': self.max_model_len,
            'max_num_batched_tokens': self.max_num_batched_tokens,
            'max_num_seqs': self.max_num_seqs,
            'use_lora': self.use_lora,
            'lora_path': self.lora_path if self.use_lora else "",
            'enforce_eager': self.enforce_eager,
            'kv_cache_layout': self.kv_cache_layout,
            'accept_threshold': self.accept_threshold,
            'complete_threshold': self.complete_threshold,
            'add_new_block_threshold': self.add_new_block_threshold,
            'diffusion_block_size': self.diffusion_block_size,
        }


@dataclass
class EvalConfig:
    """
    Evaluation configuration - Parameters for benchmark evaluation
    """
    # Task/Dataset configuration
    dataset_name: str = "gsm8k"  # Task name (e.g., gsm8k, humaneval)
    dataset_split: str = "test"
    dataset_limit: Optional[int] = None
    
    # Sampling configuration
    temperature: float = 0.0
    max_tokens: int = 256
    ignore_eos: bool = False
    
    # Output configuration
    output_dir: str = "benchmark_results"
    save_results: bool = True
    use_tqdm: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EvalConfig":
        """Create evaluation configuration from dictionary"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def get_sampling_params(self):
        """Get sampling parameters"""
        from diffulex import SamplingParams
        return SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            ignore_eos=self.ignore_eos,
        )


@dataclass
class BenchmarkConfig:
    """
    Benchmark configuration - Combines engine and evaluation configurations
    """
    engine: EngineConfig
    eval: EvalConfig
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BenchmarkConfig":
        """
        Create benchmark configuration from dictionary
        
        Supports both flat and nested dictionary structures for backward compatibility
        """
        # Check if config_dict has nested structure
        if 'engine' in config_dict and 'eval' in config_dict:
            engine = EngineConfig.from_dict(config_dict['engine'])
            eval_config = EvalConfig.from_dict(config_dict['eval'])
        else:
            # Flat structure - backward compatibility
            # Split fields into engine and eval
            engine_fields = {
                'model_path', 'tokenizer_path', 'model_name', 'decoding_strategy',
                'mask_token_id', 'use_lora', 'lora_path', 'tensor_parallel_size',
                'data_parallel_size', 'gpu_memory_utilization', 'max_model_len',
                'max_num_batched_tokens', 'max_num_seqs', 'enforce_eager',
                'kv_cache_layout', 'accept_threshold', 'complete_threshold',
                'add_new_block_threshold', 'diffusion_block_size'
            }
            
            engine_dict = {k: v for k, v in config_dict.items() if k in engine_fields}
            eval_dict = {k: v for k, v in config_dict.items() if k not in engine_fields}
            
            engine = EngineConfig.from_dict(engine_dict)
            eval_config = EvalConfig.from_dict(eval_dict)
        
        return cls(engine=engine, eval=eval_config)
    
    @classmethod
    def from_json(cls, json_path: str) -> "BenchmarkConfig":
        """Load configuration from JSON file"""
        with open(json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "BenchmarkConfig":
        """Load configuration from YAML file"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with nested structure"""
        return {
            'engine': self.engine.to_dict(),
            'eval': self.eval.to_dict(),
        }
    
    def save_json(self, json_path: str):
        """Save to JSON file"""
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def save_yaml(self, yaml_path: str):
        """Save to YAML file"""
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, allow_unicode=True, default_flow_style=False)
    
    def get_diffulex_kwargs(self) -> Dict[str, Any]:
        """Get arguments to pass to Diffulex engine"""
        return self.engine.get_diffulex_kwargs()
    
    def get_sampling_params(self):
        """Get sampling parameters"""
        return self.eval.get_sampling_params()
    
    # Convenience properties for backward compatibility
    @property
    def model_path(self) -> str:
        return self.engine.model_path
    
    @property
    def tokenizer_path(self) -> Optional[str]:
        return self.engine.tokenizer_path
    
    @property
    def model_name(self) -> str:
        return self.engine.model_name
    
    @property
    def decoding_strategy(self) -> str:
        return self.engine.decoding_strategy
    
    @property
    def dataset_name(self) -> str:
        return self.eval.dataset_name
    
    @property
    def dataset_limit(self) -> Optional[int]:
        return self.eval.dataset_limit
    
    @property
    def output_dir(self) -> str:
        return self.eval.output_dir
    
    @dataset_name.setter
    def dataset_name(self, value: str):
        self.eval.dataset_name = value
    
    @dataset_limit.setter
    def dataset_limit(self, value: Optional[int]):
        self.eval.dataset_limit = value
    
    @output_dir.setter
    def output_dir(self, value: str):
        self.eval.output_dir = value
    
    @model_path.setter
    def model_path(self, value: str):
        self.engine.model_path = value
