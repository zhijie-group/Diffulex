import os
import json
import torch
import torch.nn as nn

from tqdm import tqdm
from glob import glob
from functools import partial
from safetensors import safe_open
from diffulex.config import Config
from diffulex.logger import get_logger

logger = get_logger(__name__)


def load_lora_config(lora_path: str) -> dict:
    """Load LoRA configuration from adapter_config.json."""
    config_path = os.path.join(lora_path, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}


def enable_lora_for_model(model: nn.Module, lora_config: dict):
    """Enable LoRA for existing linear layers in the model."""
    r = lora_config.get('r', 16)
    lora_alpha = lora_config.get('lora_alpha', 32.0)
    lora_dropout = lora_config.get('lora_dropout', 0.0)
    target_modules = lora_config.get('target_modules', [])
    
    for name, module in model.named_modules():
        if hasattr(module, '__init_lora__'):
            should_apply = True
            if target_modules:
                leaf = name.split('.')[-1] if name else name
                should_apply = any(target == leaf for target in target_modules)
            if should_apply:
                module.__init_lora__(r, lora_alpha, lora_dropout)
    return model


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, config: Config):
    """Load model weights and optionally LoRA weights."""
    # Enable LoRA for linear layers if LoRA is enabled
    if config.use_lora and config.lora_path:
        lora_config = load_lora_config(config.lora_path)
        if lora_config:
            logger.info(f"LoRA Config Loaded: {lora_config}")
            model = enable_lora_for_model(model, lora_config)
        else:
            logger.info("No adapter_config.json found, using default LoRA parameters")
            default_config = {'r': 16, 'lora_alpha': 32.0, 'lora_dropout': 0.0}
            model = enable_lora_for_model(model, default_config)
    
    # Load base model weights
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in tqdm(glob(os.path.join(config.model, "*.safetensors")), desc="Loading base model"):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        
                        if config.model_name == "llada" and k == "ff_out" and "transformer.ff_out" in weight_name:
                            continue
                        elif config.model_name == "llada" and k == "transformer.ff_out":
                            v, shard_id = packed_modules_mapping[k]
                            assert v == "lm_head"
                            param_name = "lm_head.weight"
                        else:
                            v, shard_id = packed_modules_mapping[k]
                            param_name = weight_name.replace(k, v)
                            
                        if "layernorm" in param_name:
                            param = model.get_parameter(param_name)
                            weight_loader = getattr(param, "weight_loader", default_weight_loader)
                            weight_loader(param, f.get_tensor(weight_name))
                        else:
                            param = model.get_parameter(param_name)
                            weight_loader = partial(getattr(param, "weight_loader"), param, f.get_tensor(weight_name)) 
                            if shard_id is None:
                                weight_loader()
                            else:
                                weight_loader(shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
    
    # Load LoRA weights if enabled
    if config.use_lora and config.lora_path:
        if os.path.exists(config.lora_path):
            logger.info(f"Loading LoRA weights from {config.lora_path}")
            load_lora_weights_fn = partial(load_lora_weights, model, config.lora_path)
            packed_modules_mapping = packed_modules_mapping if config.model_name == "llada" else None
            model = load_lora_weights_fn(packed_modules_mapping=packed_modules_mapping)
        else:
            logger.warning(f"LoRA path {config.lora_path} does not exist, skipping LoRA loading")
    
    return model


def load_lora_weights(model: nn.Module, lora_path: str, packed_modules_mapping: dict | None = None):
    """Load LoRA weights into LoRA-enabled layers."""
    try:
        lora_config = load_lora_config(lora_path)
        target_modules = lora_config.get('target_modules', [])
        
        lora_weights = {}
        
        for file in tqdm(glob(os.path.join(lora_path, "*.safetensors")), desc="Loading LoRA"):
            with safe_open(file, "pt", "cpu") as f:
                for weight_name in f.keys():
                    lora_weights[weight_name] = f.get_tensor(weight_name)
        
        applied_count = 0

        modified_modules = None
        if packed_modules_mapping is not None:
            modified_modules = [v for k, (v, _) in packed_modules_mapping.items() if k in target_modules]
            rev_mapping = {v: k for k, (v, _) in packed_modules_mapping.items()}
            
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                should_apply = True
                
                if modified_modules is not None:
                    modified_module_type = '.'.join(name.split('.')[-2:])  
                    org_module_type = rev_mapping[modified_module_type]
                    org_name = name.replace(modified_module_type, org_module_type)
                    should_apply = any(target in modified_module_type for target in modified_modules)
                elif target_modules:
                    module_type = name.split('.')[-1] if '.' in name else name
                    should_apply = any(target in module_type for target in target_modules)
                 
                if not should_apply:
                    continue
                
                base_patterns = [
                    name,
                    f"base_model.model.{name}",
                    f"model.{name}",
                ] if modified_modules is None else [
                    org_name,
                    f"base_model.model.{org_name}",
                    f"model.{org_name}",
                ]
                
                found_a = found_b = None
                for base_name in base_patterns:
                    lora_a_keys = [
                        f"{base_name}.lora_A.weight",
                        f"{base_name}.lora_A.default.weight",
                        f"{base_name}.lora_A",
                    ]
                    lora_b_keys = [
                        f"{base_name}.lora_B.weight", 
                        f"{base_name}.lora_B.default.weight",
                        f"{base_name}.lora_B",
                    ]
                    
                    for key in lora_a_keys:
                        if key in lora_weights:
                            found_a = lora_weights[key]
                            break
                    for key in lora_b_keys:
                        if key in lora_weights:
                            found_b = lora_weights[key]
                            break
                    
                    if found_a is not None and found_b is not None:
                        break
                
                if found_a is not None and found_b is not None:
                    if hasattr(module, 'tp_size') and module.tp_size > 1:
                        if hasattr(module, 'tp_dim') and module.tp_dim == 0:
                            shard_size = found_b.size(0) // module.tp_size
                            start_idx = module.tp_rank * shard_size
                            found_b = found_b[start_idx:start_idx + shard_size]
                        elif hasattr(module, 'tp_dim') and module.tp_dim == 1:
                            shard_size = found_a.size(1) // module.tp_size
                            start_idx = module.tp_rank * shard_size
                            found_a = found_a[:, start_idx:start_idx + shard_size]
                    
                    try:
                        module.lora_A.data.copy_(found_a)
                        module.lora_B.data.copy_(found_b)
                        applied_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to load LoRA weights for {name}: {e}")
        
        for module in model.modules():
            if hasattr(module, 'merge_lora'):
                module.merge_lora()
        
        logger.info(f"LoRA weights applied to {applied_count} layers and merged")
        
    except Exception as e:
        logger.error(f"Error loading LoRA weights: {e}")
        logger.warning("Continuing with base model only")
    
    return model
