import transformers
import torch

from peft import PeftModel, PeftConfig
from lm_eval.models.utils import get_dtype

from diffulex.legacy.config import Config
from diffulex.legacy.engine.model_runner import AutoModelRunner

from model_cache.dream.model_dream import DreamModel
from model_cache.dream.configuration_dream import DreamConfig

pretrained = "/data1/ckpts/Dream-org/Dream-v0-Base-7B"
dtype = "auto"
trust_remote_code = True
lora_path = "/data1/xck/ckpt/wx_dream_base/Decoder-ddt_test-20k"
device = "cuda"

target_dtype = get_dtype(dtype)  

# 加载PEFT模型
model_config = DreamConfig.from_pretrained(pretrained)
model = DreamModel.from_pretrained(
    pretrained, 
    config=model_config,
    torch_dtype=target_dtype,
    trust_remote_code=False,
).eval()

config = PeftConfig.from_pretrained(lora_path)
model = PeftModel.from_pretrained(model, lora_path)

if target_dtype is not None and target_dtype != "auto":
    model = model.to(target_dtype)

model = model.to(device)

tokenizer = transformers.AutoTokenizer.from_pretrained(
    pretrained, trust_remote_code=trust_remote_code
)

# 加载D2F模型
temp_config = Config(
    model=pretrained,
    lora_path=lora_path,
    use_lora=True,
    model_name="dream", 
    enforce_eager=True, 
    tensor_parallel_size=1,
    accept_threshold=0.95,
    complete_threshold=0.9,
    add_new_block_threshold=0.1,
)

temp_model = AutoModelRunner.from_config(temp_config, 0, []).model

def get_layer_weights(model, layer_name_pattern):
    """获取特定层的权重用于比较"""
    weights = {}
    for name, param in model.named_parameters():
        if layer_name_pattern in name and "lora_" not in name:
            weights[name] = param.detach().cpu()
    return weights

def compare_layer_weights(model1, model2, layer_pattern):
    """比较特定层的权重"""
    print(f"\n检查包含 '{layer_pattern}' 的层:")
    
    weights1 = get_layer_weights(model1, layer_pattern)
    weights2 = get_layer_weights(model2, layer_pattern)
    
    print(f"  PEFT模型找到 {len(weights1)} 个参数")
    print(f"  D2F模型找到 {len(weights2)} 个参数")
    
    if len(weights1) == 0 and len(weights2) == 0:
        print(f"  两个模型都没有找到包含 '{layer_pattern}' 的层")
        return True
    
    # 找公共参数
    common_keys = set(weights1.keys()) & set(weights2.keys())
    print(f"  公共参数: {len(common_keys)}")
    
    all_same = True
    for key in sorted(common_keys)[:5]:  # 只检查前5个
        w1, w2 = weights1[key], weights2[key]
        if w1.shape != w2.shape:
            print(f"    ❌ {key}: 形状不匹配 {w1.shape} vs {w2.shape}")
            all_same = False
            continue
        
        diff = torch.abs(w1 - w2).max().item()
        if diff > 1e-6:
            print(f"    ❌ {key}: 最大差异 {diff:.2e}")
            all_same = False
        else:
            print(f"    ✅ {key}: 权重相同")
    
    # 显示不匹配的参数名
    only_in_1 = set(weights1.keys()) - set(weights2.keys())
    only_in_2 = set(weights2.keys()) - set(weights1.keys())
    
    if only_in_1:
        print(f"    仅在PEFT模型中: {len(only_in_1)}个, 例如 {list(only_in_1)[:2]}")
    if only_in_2:
        print(f"    仅在D2F模型中: {len(only_in_2)}个, 例如 {list(only_in_2)[:2]}")
    
    return all_same

def print_model_structure(model, name, max_layers=5):
    """打印模型结构用于调试"""
    print(f"\n{name} 模型结构:")
    param_names = list(model.named_parameters())
    print(f"总参数数量: {len(param_names)}")
    
    print("前几个参数:")
    for i, (param_name, param) in enumerate(param_names[:max_layers]):
        print(f"  {param_name}: {param.shape}")
    
    if len(param_names) > max_layers:
        print("  ...")
        print("后几个参数:")
        for param_name, param in param_names[-2:]:
            print(f"  {param_name}: {param.shape}")

# 打印模型结构
print_model_structure(model, "PEFT")
print_model_structure(temp_model, "D2F")

# 比较关键层
print("\n开始比较关键层权重...")
layer_patterns = ["embed_tokens", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"]

all_layers_match = True
for pattern in layer_patterns:
    layer_match = compare_layer_weights(model, temp_model, pattern)
    all_layers_match = all_layers_match and layer_match

if all_layers_match:
    print("\n✅ 关键层权重完全匹配！")
else:
    print("\n❌ 发现权重差异")

# 检查LoRA状态
print("\n检查LoRA状态...")
lora_modules = []
for name, module in temp_model.named_modules():
    if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
        merged_status = getattr(module, 'merged', 'unknown')
        lora_modules.append(f"{name} (merged: {merged_status})")

print(f"D2F模型中的LoRA模块: {len(lora_modules)}")
for lora_mod in lora_modules[:5]:  # 只显示前5个
    print(f"  {lora_mod}")

# 检查PEFT模型的LoRA状态
print("\nPEFT模型LoRA状态:")
peft_lora_count = 0
for name, module in model.named_modules():
    if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
        peft_lora_count += 1

print(f"PEFT模型中的LoRA模块: {peft_lora_count}")

print("\n权重比较完成！")
