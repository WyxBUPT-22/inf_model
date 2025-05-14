# Qwen2.5-Math-Eval-01.py
 
import os
from modelscope import AutoModelForCausalLM, AutoTokenizer
 
# 权重文件目录
model_dir = os.path.join("Qwen2___5-Math-1___5B-Instruct")
print(f'权重目录: {model_dir}')
 
# 初始化模型
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype='auto',
    device_map='auto',
    local_files_only=True,
)
 
# 初始化分词器
tokenizer = AutoTokenizer.from_pretrained(
	model_dir,
	local_files_only=True,
)
 
# Prompt提示词
prompt = '有一个二元对称信道，信道的误码率p=0.06，设该信道以1000个符号/秒的速率传输输入符号，现有一消息序列，共有9500个符号，并设消息中q(0)=q(1)=0.5，问从信息传输的角度来考虑，10秒能否将消息无失真地传送完成?'
messages = [
    {'role': 'system', 'content': '你是一位数学专家，特别擅长解答数学题。'},
    {'role': 'user', 'content': prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer(
    [text],
    return_tensors='pt',
).to(model.device)
 
print(f'开始推理: {prompt}')
 
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=2048,
)
 
print('推理完成.')
 
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
 
response = tokenizer.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)[0]
 
print(f'推理结果: {response}')