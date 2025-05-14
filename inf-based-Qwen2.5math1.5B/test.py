import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel # 导入 PeftModel

# --- 配置 ---
# *原始*基础 Qwen1.5-1.8B 模型文件的路径
# 这个路径应该与您 config.json 中的 'base_model_name_or_path' 匹配，
# 或者如果您是从 Hugging Face 下载的，则是其 ID。
# 确保此路径包含完整的基模型，而不仅仅是 checkpoint。
base_model_path = "Qwen2___5-Math-1___5B-Instruct" 
# 如果 "qwen_model/qwen1.5-1.8b" 相对于您的脚本不是正确的路径，
# 请提供完整的绝对路径或正确的相对路径。
# 例如: base_model_path = r"E:\_MyCollegeLife\_3rd_2\inf_model_finetune\qwen_model\qwen1.5-1.8b"
# 或者如果它在 Hugging Face Hub 上: base_model_path = "Qwen/Qwen1.5-1.8B"

# 您的 LoRA 适配器 checkpoint 目录的路径
adapter_path = "qwen2.5-1.5b-math-finetuned-info-theory-loss-eval"

# --- 加载 ---

print(f"正在从以下位置加载基础模型 Tokenizer: {base_model_path}")
# 从基础模型路径加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,
    trust_remote_code=True # Qwen 模型通常需要这个参数
)
print("Tokenizer 加载成功。")

print(f"正在从以下位置加载基础模型: {base_model_path}")
# 加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16, # 或根据您的硬件/需求选择 torch.float16 / None (float32)
    device_map="auto",        # 需要 'accelerate' 库: pip install accelerate
    trust_remote_code=True    # Qwen 模型通常需要这个参数
)
print("基础模型加载成功。")

print(f"正在从以下位置加载 PEFT 适配器: {adapter_path}")
# 加载 LoRA 适配器并将其应用到基础模型上
# PeftModel 会自动在 adapter_path 中查找 'adapter_config.json' (或者 'config.json' 如果只有这个)
# 以及 'adapter_model.bin'/'adapter_model.safetensors'
model = PeftModel.from_pretrained(model, adapter_path)
print("PEFT 适配器加载成功。")

# 可选：合并权重以可能加速推理（合并后无法撤销）
# print("正在合并适配器权重...")
# model = model.merge_and_unload()
# print("权重合并完成。")

# --- 模型准备就绪 ---
# 现在 'model' 就是微调后的模型 (基础模型 + 适配器)
# 您可以继续执行 test.py 脚本的其余逻辑 (生成、评估等)

print("模型已准备好进行推理。")

# 示例：生成文本 (用您实际的测试代码替换)
inputs = tokenizer("熵的基本性质包括非负性，即对于任意随机变量X，其熵H(X)满足H(X)≥0。请用数学表达式说明这一性质，并给出证明过程。", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=2048)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

