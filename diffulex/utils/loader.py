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


def _load_gptq_awq_weights(model: nn.Module, config: Config):
    """Load GPTQ/AWQ offline quantized weights from checkpoint.
    
    Args:
        model: Model module
        config: Config with model path
        
    Returns:
        Tuple of (loaded_gptq_count, loaded_awq_count, skipped_count)
    """
    loaded_gptq = 0
    loaded_awq = 0
    skipped = 0
    
    # Check if model is configured for GPTQ or AWQ
    weight_attn_dtype = getattr(config, "linear_attn_weight_dtype", "bf16") or "bf16"
    weight_mlp_dtype = getattr(config, "linear_mlp_weight_dtype", "bf16") or "bf16"
    
    use_gptq = weight_attn_dtype.lower() == "gptq" or weight_mlp_dtype.lower() == "gptq"
    use_awq = weight_attn_dtype.lower() == "awq" or weight_mlp_dtype.lower() == "awq"
    
    if not (use_gptq or use_awq):
        return loaded_gptq, loaded_awq, skipped
    
    # Collect all weight names from safetensors files
    all_keys = []
    all_files = list(glob(os.path.join(config.model, "*.safetensors")))
    for file in all_files:
        with safe_open(file, "pt", "cpu") as f:
            all_keys.extend(f.keys())
    
    # Group keys by module prefix
    module_keys: dict[str, dict[str, str]] = {}
    for key in all_keys:
        # Check for GPTQ/AWQ keys: {prefix}.qweight, {prefix}.qzeros, {prefix}.scales, {prefix}.g_idx (GPTQ only)
        if key.endswith(".qweight"):
            prefix = key[:-8]  # Remove ".qweight"
            if prefix not in module_keys:
                module_keys[prefix] = {}
            module_keys[prefix]["qweight"] = key
        elif key.endswith(".qzeros"):
            prefix = key[:-7]  # Remove ".qzeros"
            if prefix not in module_keys:
                module_keys[prefix] = {}
            module_keys[prefix]["qzeros"] = key
        elif key.endswith(".scales"):
            prefix = key[:-7]  # Remove ".scales"
            if prefix not in module_keys:
                module_keys[prefix] = {}
            module_keys[prefix]["scales"] = key
        elif key.endswith(".g_idx"):
            prefix = key[:-6]  # Remove ".g_idx"
            if prefix not in module_keys:
                module_keys[prefix] = {}
            module_keys[prefix]["g_idx"] = key
    
    # Load GPTQ/AWQ weights for each module
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    
    for prefix, key_dict in module_keys.items():
        if "qweight" not in key_dict or "qzeros" not in key_dict or "scales" not in key_dict:
            continue  # Skip incomplete sets
        
        # Map prefix to module name
        module_name = prefix
        for k, (v, _) in packed_modules_mapping.items():
            if k in prefix:
                module_name = prefix.replace(k, v)
                break
        
        # Try to find the module
        try:
            module = None
            # Try exact match first
            try:
                module = dict(model.named_modules())[module_name]
                if not hasattr(module, "set_offline_quantized_weight"):
                    module = None
            except KeyError:
                pass
            
            # Try partial match if exact match failed
            if module is None:
                for name, m in model.named_modules():
                    # Handle different naming conventions
                    if (
                        name == module_name
                        or name.endswith("." + module_name)
                        or module_name.endswith("." + name)
                        or (name.split(".")[-1] == module_name.split(".")[-1])
                    ):
                        if hasattr(m, "set_offline_quantized_weight"):
                            module = m
                            break
            
            if module is None:
                skipped += 1
                continue
            
            # Determine format: check if g_idx exists (GPTQ) or not (AWQ)
            has_g_idx = "g_idx" in key_dict
            if has_g_idx and use_gptq:
                format = "gptq"
            elif not has_g_idx and use_awq:
                format = "awq"
            else:
                # Prefer GPTQ if both are enabled and g_idx exists
                format = "gptq" if (use_gptq and has_g_idx) else ("awq" if use_awq else None)
            
            if format is None:
                skipped += 1
                continue
            
            # Load tensors from safetensors files
            qweight = None
            qzeros = None
            scales = None
            g_idx = None
            
            for file in all_files:
                with safe_open(file, "pt", "cpu") as f:
                    if key_dict["qweight"] in f.keys() and qweight is None:
                        qweight = f.get_tensor(key_dict["qweight"])
                    if key_dict["qzeros"] in f.keys() and qzeros is None:
                        qzeros = f.get_tensor(key_dict["qzeros"])
                    if key_dict["scales"] in f.keys() and scales is None:
                        scales = f.get_tensor(key_dict["scales"])
                    if format == "gptq" and "g_idx" in key_dict and key_dict["g_idx"] in f.keys() and g_idx is None:
                        g_idx = f.get_tensor(key_dict["g_idx"])
                
                # Early exit if all required tensors are loaded
                if qweight is not None and qzeros is not None and scales is not None:
                    if format != "gptq" or g_idx is not None:
                        break
            
            if qweight is None or qzeros is None or scales is None:
                skipped += 1
                continue
            
            # Infer dimensions from tensor shapes
            out_features, packed_in = qweight.shape
            in_features = packed_in * 2  # Packed int4: 2 values per byte (max estimate)
            # Refine in_features from scales shape if available
            if scales.shape[1:] != ():
                # scales is [num_groups, in_features] or [num_groups]
                if len(scales.shape) == 2:
                    in_features = scales.shape[1]
            
            # Default group_size for GPTQ/AWQ is 128
            group_size = 128
            # Infer group_size from scales/qzeros shape
            num_groups = qzeros.shape[0]
            if num_groups > 0:
                estimated_group_size = (out_features + num_groups - 1) // num_groups
                if estimated_group_size > 0:
                    group_size = estimated_group_size
            
            # Handle tensor parallel: if tp_size > 1, we need to handle sharding
            # For MVP, only support TP=1 (tensor_parallel_size=1)
            tp_size = getattr(module, "tp_size", 1)
            if tp_size > 1:
                print(
                    f"Warning: Tensor parallel (TP={tp_size}) is not fully supported for offline quantized weights. "
                    f"Skipping {module_name}. Please provide a TP=1 checkpoint or implement TP sharding logic."
                )
                skipped += 1
                continue
            
            # Set offline quantized weight
            try:
                module.set_offline_quantized_weight(
                    format=format,
                    qweight=qweight,
                    qzeros=qzeros,
                    scales=scales,
                    out_features=out_features,
                    in_features=in_features,
                    group_size=group_size,
                    g_idx=g_idx,
                )
                if format == "gptq":
                    loaded_gptq += 1
                else:
                    loaded_awq += 1
            except Exception as e:
                print(f"Failed to load offline quantized weights for {module_name}: {e}")
                import traceback
                traceback.print_exc()
                skipped += 1
        
        except Exception as e:
            print(f"Error loading offline quantized weights for {prefix}: {e}")
            import traceback
            traceback.print_exc()
            skipped += 1
    
    return loaded_gptq, loaded_awq, skipped


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
    
    # First, try to load offline quantized weights (GPTQ/AWQ)
    loaded_gptq, loaded_awq, skipped_offline = _load_gptq_awq_weights(model, config)
    if loaded_gptq > 0 or loaded_awq > 0:
        print(f"Loaded offline quantized weights: GPTQ={loaded_gptq}, AWQ={loaded_awq}, skipped={skipped_offline}")
    
    # Load base model weights (only for non-offline-quantized layers)
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in tqdm(glob(os.path.join(config.model, "*.safetensors")), desc="Loading base model"):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # Skip GPTQ/AWQ keys (already loaded)
                if any(
                    weight_name.endswith(suffix)
                    for suffix in [".qweight", ".qzeros", ".scales", ".g_idx"]
                ):
                    continue
                
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
                            try:
                            param = model.get_parameter(param_name)
                            weight_loader = getattr(param, "weight_loader", default_weight_loader)
                            weight_loader(param, f.get_tensor(weight_name))
                            except (AttributeError, KeyError):
                                # Try buffer fallback for non-parameter weights
                                try:
                                    buffer = model.get_buffer(param_name)
                                    buffer.copy_(f.get_tensor(weight_name))
                                except (AttributeError, KeyError):
                                    pass
                        else:
                            try:
                            param = model.get_parameter(param_name)
                            weight_loader = partial(getattr(param, "weight_loader"), param, f.get_tensor(weight_name)) 
                            if shard_id is None:
                                weight_loader()
                            else:
                                weight_loader(shard_id)
                            except (AttributeError, KeyError):
                                # Parameter might not exist if offline quantized weights were loaded
                                # Skip it silently
                                pass
                        break
                else:
                    try:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
                    except (AttributeError, KeyError):
                        # Try buffer fallback for non-parameter weights
                        try:
                            buffer = model.get_buffer(weight_name)
                            buffer.copy_(f.get_tensor(weight_name))
                        except (AttributeError, KeyError):
                            pass
    
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
