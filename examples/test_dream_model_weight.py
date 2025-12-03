import transformers
import torch

from peft import PeftModel, PeftConfig
from lm_eval.models.utils import get_dtype

from diffulex.legacy.config import Config
from diffulex.legacy.models.auto_model import AutoModelLM

from model_cache.dream.model_dream import DreamModel
from model_cache.dream.configuration_dream import DreamConfig

pretrained = "/data1/ckpts/Dream-org/Dream-v0-Base-7B"
dtype = "auto"
trust_remote_code = True
lora_path = "/data1/xck/ckpt/wx_dream_base/Decoder-ddt_test-20k"
device = "cuda"

target_dtype = get_dtype(dtype)  

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

temp_model = AutoModelLM.from_config(temp_config)

def get_layer_weights(model, layer_name_pattern):
    weights = {}
    for name, param in model.named_parameters():
        if layer_name_pattern in name and "lora_" not in name:
            weights[name] = param.detach().cpu()
    return weights

def compare_layer_weights(model1, model2, layer_pattern):
    print(f"检查包含 '{layer_pattern}' 的层:")
    
    weights1 = get_layer_weights(model1, layer_pattern)
    weights2 = get_layer_weights(model2, layer_pattern)
    
    print(f"  PEFT模型找到 {len(weights1)} 个参数")
    print(f"  D2F模型找到 {len(weights2)} 个参数")
    
    if len(weights1) == 0 and len(weights2) == 0:
        print(f"  两个模型都没有找到包含 '{layer_pattern}' 的层")
        return True
    
    common_keys = set(weights1.keys()) & set(weights2.keys())
    print(f"  公共参数: {len(common_keys)}")
    
    all_same = True
    for key in sorted(common_keys)[:3]:
        w1, w2 = weights1[key], weights2[key]
        if w1.shape != w2.shape:
            print(f"    形状不匹配 {key}: {w1.shape} vs {w2.shape}")
            all_same = False
            continue
        
        diff = torch.abs(w1 - w2).max().item()
        if diff > 1e-6:
            print(f"    权重差异 {key}: 最大差异 {diff:.2e}")
            all_same = False
        else:
            print(f"    权重相同 {key}")
    
    only_in_1 = set(weights1.keys()) - set(weights2.keys())
    only_in_2 = set(weights2.keys()) - set(weights1.keys())
    
    if only_in_1:
        print(f"    仅在PEFT模型中: {len(only_in_1)}个参数")
    if only_in_2:
        print(f"    仅在D2F模型中: {len(only_in_2)}个参数")
    
    return all_same

print("开始比较模型权重...")
layer_patterns = ["embed_tokens", "q_proj", "k_proj", "v_proj", "o_proj", "lm_head"]

all_match = True
for pattern in layer_patterns:
    layer_match = compare_layer_weights(model, temp_model, pattern)
    all_match = all_match and layer_match
    print()

if all_match:
    print("所有关键层权重匹配！")
else:
    print("发现权重差异")

print("检查LoRA状态...")
lora_count = 0
merged_count = 0
for name, module in temp_model.named_modules():
    if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
        lora_count += 1
        if hasattr(module, 'merged') and module.merged:
            merged_count += 1

print(f"D2F模型中的LoRA层数量: {lora_count}")
print(f"已合并的LoRA层数量: {merged_count}")

print("\n权重比较完成！")