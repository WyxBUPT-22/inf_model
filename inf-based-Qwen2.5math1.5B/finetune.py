#!/usr/bin/env python
# coding=utf-8

import os
import torch
import logging
import numpy as np
import math # 用于计算 Perplexity
import re   # 用于清理生成的文本
import json # 用于保存最终指标
from tqdm import tqdm # 用于生成过程的进度条
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    # DataCollatorForLanguageModeling, # *** 不再使用标准的 Causal LM Collator ***
    HfArgumentParser,
    set_seed, # 用于设置随机种子
)
from peft import LoraConfig, get_peft_model, PeftModel # 用于加载 Peft 模型
from datasets import load_dataset, DatasetDict
# from sklearn.model_selection import train_test_split # 不再需要，datasets.DatasetDict.train_test_split 已使用
import nltk # 用于 BLEU/ROUGE 分词
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl # 增强类型提示
from transformers.trainer_utils import get_last_checkpoint # 用于恢复训练
import warnings
from functools import partial # 用于传递 tokenizer 到 map

# --- 从 transformers 导入所需的基础类 ---
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers.data.data_collator import DataCollatorMixin # 引入 Mixin
from typing import Any, Dict, List, Optional, Tuple, Union

# --- 配置日志记录器 ---
# 移除重复的 basicConfig 设置，保留顶部的设置
log_file_handler = logging.FileHandler("finetune_information_theory_loss_eval.log", encoding='utf-8') # 指定 UTF-8
log_stream_handler = logging.StreamHandler()

# 移动 basicConfig 到脚本顶部，确保它首先被调用
logging.basicConfig(
    level=logging.DEBUG, # <--- 在这里设置全局 DEBUG 级别，以便捕获所有信息
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S", # 添加日期格式
    handlers=[log_file_handler, log_stream_handler] # 应用文件和流处理器
)

logger = logging.getLogger(__name__) # 获取 logger 实例

# --- 忽略无关警告 ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 自定义 Data Collator ---
# --- 自定义 Data Collator ---
class CustomDataCollator(DataCollatorMixin):
    """
    一个自定义的数据整理器，显式地分别填充 'input_ids' 和 'labels'。
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] # Defaults will be set in __init__
    max_length: Optional[int]
    pad_to_multiple_of: Optional[int]
    label_pad_token_id: int # Default will be set in __init__
    return_tensors: str # Default will be set in __init__

    # --- <<< ADD THIS __init__ METHOD >>> ---
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        label_pad_token_id: int = -100,
        return_tensors: str = "pt"
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id
        self.return_tensors = return_tensors
        # Optional: Add a log to confirm initialization
        logger.debug(f"CustomDataCollator initialized with: padding={self.padding}, max_length={self.max_length}, pad_to_multiple_of={self.pad_to_multiple_of}, label_pad_token_id={self.label_pad_token_id}")
    # --- <<< END OF __init__ METHOD >>> ---


    def torch_call(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # --- 基础检查 (可选但推荐保留) ---
        batch_input_ids_lengths = []
        batch_labels_lengths = []
        problem_detected = False
        for i, feature in enumerate(features):
             # 检查 feature 是否为字典
            if not isinstance(feature, dict):
                logger.error(f"  Feature {i}: 不是字典! 类型: {type(feature)}")
                problem_detected = True
                continue # 跳过对此 feature 的后续检查
            # 检查 input_ids
            input_ids = feature.get("input_ids")
            if input_ids is None or not isinstance(input_ids, list) or not all(isinstance(x, int) for x in input_ids):
                 logger.error(f"  Feature {i}: 'input_ids' 存在问题 (None, 非列表, 或含非整数)。")
                 problem_detected = True
            else:
                 batch_input_ids_lengths.append(len(input_ids))
            # 检查 labels
            labels = feature.get("labels")
            if labels is None or not isinstance(labels, list) or not all(isinstance(x, int) for x in labels):
                 logger.error(f"  Feature {i}: 'labels' 存在问题 (None, 非列表, 或含非整数)。")
                 problem_detected = True
            else:
                 batch_labels_lengths.append(len(labels))
            # 检查长度一致性 (在所有字段都有效的情况下)
            if not problem_detected and input_ids is not None and labels is not None and len(input_ids) != len(labels): # Added checks for None
                 logger.error(f"  Feature {i}: input_ids 长度 ({len(input_ids)}) 与 labels 长度 ({len(labels)}) 不一致!")
                 problem_detected = True

        logger.debug(f"CustomDataCollator: Input lengths: iids={batch_input_ids_lengths}, lbls={batch_labels_lengths}")
        if problem_detected:
             logger.critical("!!! CustomDataCollator 检测到特征结构问题。请检查预处理。 !!!")
             raise ValueError("在 DataCollator 中检测到特征结构问题")
        # --- 检查结束 ---

        # 1. 分别提取 input_ids 和 labels 列表
        input_ids_list = [feature["input_ids"] for feature in features]
        labels_list = [feature["labels"] for feature in features]

        # 2. 使用 tokenizer.pad 分别填充 input_ids 和 labels
        #    注意：传入字典结构 {"input_ids": list_to_pad}
        #    使用 self.padding, self.max_length, self.pad_to_multiple_of
        batch = {}
        # --- 填充 Input IDs ---
        # <<< Make sure to use self.tokenizer, self.padding etc. here >>>
        padded_input_ids_dict = self.tokenizer.pad(
            {"input_ids": input_ids_list},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
            return_attention_mask=True, # 请求 attention_mask
        )
        batch["input_ids"] = padded_input_ids_dict["input_ids"]
        batch["attention_mask"] = padded_input_ids_dict["attention_mask"]

        # --- 填充 Labels ---
        labels_needs_padding = any(isinstance(f.get("labels"), list) for f in features)
        if labels_needs_padding:
            # 先用 tokenizer 的 pad_token_id 填充
            # <<< Use self attributes here too >>>
            padded_labels_dict = self.tokenizer.pad(
                {"input_ids": labels_list}, # 仍用 "input_ids" 作为 key 让 pad 函数处理
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            labels_tensor = padded_labels_dict["input_ids"]

            # <<< Use self.label_pad_token_id >>>
            labels_tensor[batch["attention_mask"] == 0] = self.label_pad_token_id
            # <<< Use self.tokenizer and self.label_pad_token_id >>>
            if self.tokenizer.pad_token_id is not None: # Check if pad_token_id exists
                 labels_tensor[labels_tensor == self.tokenizer.pad_token_id] = self.label_pad_token_id

            batch["labels"] = labels_tensor
        else:
            logger.warning("当前批次中未找到 'labels' 列表，将不包含 labels。")
            pass


        # 3. 最终检查和日志
        if 'labels' in batch and batch['input_ids'].shape != batch['labels'].shape:
             logger.error(f"!!! CustomDataCollator: Final input_ids shape {batch['input_ids'].shape} != labels shape {batch['labels'].shape} !!!")
             raise ValueError("填充后 input_ids 和 labels 形状不一致")
        if 'attention_mask' in batch and batch['input_ids'].shape != batch['attention_mask'].shape:
             logger.error(f"!!! CustomDataCollator: Final input_ids shape {batch['input_ids'].shape} != attention_mask shape {batch['attention_mask'].shape} !!!")
             raise ValueError("填充后 input_ids 和 attention_mask 形状不一致")

        return batch


# --- 参数配置 ---
model_path = ".cache/modelscope/hub/models/Qwen/Qwen2___5-Math-1___5B-Instruct"  # 模型路径
dataset_path = "finetune_data_converted.json"  # 数据集路径
output_dir = "./qwen2.5-0.5b-math-finetuned-info-theory-loss-eval"  # 输出路径
tensorboard_log_dir = os.path.join(output_dir, "tf-logs") # TensorBoard 日志放在输出目录内
MAX_SEQ_LENGTH = 2048  # 定义统一的最大序列长度
TRAIN_BATCH_SIZE = 2     # 训练批次大小
GRAD_ACCUM_STEPS = 4     # 梯度累积步数 (有效批大小 = 2*4 = 8)
EVAL_BATCH_SIZE = 4      # ***评估批次大小 (用于计算 Loss, 可以稍大) ***
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3
LOGGING_STEPS = 50
SAVE_STEPS = 500
EVAL_STEPS = 500         # 评估 Loss 的频率
WARMUP_STEPS = 100
SEED = 42                # 随机种子，确保可复现性
NUM_PROC_PREFILTER = max(os.cpu_count() // 2, 1) # 预过滤使用的进程数
NUM_PROC_MAP = max(os.cpu_count() // 2, 1)       # 主要映射使用的进程数
# --- 参数配置结束 ---


# --- LaTeX 公式预处理 ---
def preprocess_latex(text):
    """清理和标准化 LaTeX 公式，保留核心结构"""
    if not isinstance(text, str): # 增加健壮性
        return ""
    try:
        # 简化可能由转义错误产生的多个反斜杠
        text = re.sub(r"\\{2,}", r"\\", text)
        # 标准化空白符
        text = re.sub(r"\s+", " ", text)
        # 去除首尾空白和文本内部换行符
        text = text.replace("\n", " ").strip()
        return text
    except Exception as e:
        logger.warning(f"LaTeX 预处理出错: {e} on text: {text[:100]}...")
        return str(text).strip() if text else "" # 保证返回字符串

# --- <<< NEW >>> Pre-filtering function ---
def filter_problematic_samples(example):
    """检查预处理后 instruction 或 output 是否为空，为空则过滤"""
    processed_inp = preprocess_latex(example.get("instruction", "")) # 使用 .get 提供默认值
    processed_tgt = preprocess_latex(example.get("output", ""))

    # 如果处理后instruction或output为空，则返回False (将被过滤掉)
    if not processed_inp or not processed_tgt:
        return False
    # 否则返回True (保留该样本)
    return True

# --- 数据预处理函数 (用于训练和 Loss 评估) ---
def preprocess_function_for_training(examples, tokenizer): # 将 tokenizer 传入
    inputs = examples["instruction"]
    targets = examples["output"]
    model_inputs = {"input_ids": [], "labels": []} # attention_mask 会由 collator 处理
    processed_count = 0
    skipped_long_prompt = 0
    skipped_empty = 0 # 这个计数在预过滤后理论上应为 0，但保留以防万一
    skipped_invalid_structure = 0 # 新增计数器

    for inp, tgt in zip(inputs, targets):
        # 预处理 (理论上预过滤已处理空值，但作为安全检查)
        processed_inp = preprocess_latex(inp)
        processed_tgt = preprocess_latex(tgt)
        if not processed_inp or not processed_tgt:
            # logger.debug("Skipping sample due to empty instruction or output after preprocessing (should have been pre-filtered).")
            skipped_empty += 1
            continue

        # 构建输入格式 "[问题] instruction [回答] output" + EOS
        prompt = f"[问题] {processed_inp} [回答]" # Prompt 部分
        full_text = f"{prompt} {processed_tgt}{tokenizer.eos_token}" # 包含目标和 EOS 的完整文本

        # 分词完整文本
        tokenized_full = tokenizer(full_text, add_special_tokens=False)
        tokenized_prompt = tokenizer(prompt, add_special_tokens=False)

        input_ids_full = tokenized_full["input_ids"]
        input_ids_prompt = tokenized_prompt["input_ids"]
        prompt_len = len(input_ids_prompt)

        # --- 截断检查 ---
        if len(input_ids_full) > MAX_SEQ_LENGTH:
            # 截断 input_ids
            input_ids_full = input_ids_full[:MAX_SEQ_LENGTH]
            # 确保最后一个 token 是 EOS
            if input_ids_full[-1] != tokenizer.eos_token_id:
                input_ids_full[-1] = tokenizer.eos_token_id
            logger.debug(f"截断样本: 原始长度 {len(tokenized_full['input_ids'])}, 截断至 {MAX_SEQ_LENGTH}")

            # 检查截断后 prompt 是否完整 (重要!)
            if prompt_len >= MAX_SEQ_LENGTH:
                # logger.warning(f"跳过样本: Prompt 长度 ({prompt_len}) >= MAX_SEQ_LENGTH ({MAX_SEQ_LENGTH})。考虑增加 MAX_SEQ_LENGTH 或过滤此类样本。")
                skipped_long_prompt += 1
                continue # 跳过这个样本, 因为无法生成有效的 labels

        # 创建标签
        labels = list(input_ids_full) # 复制截断后的 input_ids
        # 屏蔽 prompt 部分，确保不超过 labels 长度
        actual_prompt_len_in_labels = min(prompt_len, len(labels))
        labels[:actual_prompt_len_in_labels] = [-100] * actual_prompt_len_in_labels

        # --- 结构和类型检查 ---
        valid_sample = True
        error_msg = ""

        if not isinstance(input_ids_full, list):
            error_msg = f"input_ids_full 不是列表! 类型: {type(input_ids_full)}. Input: '{inp[:50]}...'"
            valid_sample = False
        elif not all(isinstance(token_id, int) for token_id in input_ids_full):
            non_int_elements = [elem for elem in input_ids_full if not isinstance(elem, int)]
            error_msg = f"input_ids_full 包含非整数元素! 示例: {non_int_elements[:5]}. Input: '{inp[:50]}...'"
            valid_sample = False

        if valid_sample: # 只有在 input_ids 没问题时才检查 labels
            if not isinstance(labels, list):
                error_msg = f"labels 不是列表! 类型: {type(labels)}. Input: '{inp[:50]}...'"
                valid_sample = False
            # 不需要检查空列表了，因为 input_ids_full 经过截断和 EOS 处理，不可能为空
            # elif not labels and input_ids_full: ...
            elif not all(isinstance(label_val, int) for label_val in labels):
                non_int_labels = [elem for elem in labels if not isinstance(elem, int)]
                error_msg = f"labels 包含非整数元素! 示例: {non_int_labels[:5]}. Input: '{inp[:50]}...'"
                valid_sample = False
            # --- <<< 新增：检查 input_ids 和 labels 长度是否一致 >>> ---
            elif len(input_ids_full) != len(labels):
                 error_msg = f"处理后 input_ids 长度 ({len(input_ids_full)}) 与 labels 长度 ({len(labels)}) 不一致! Input: '{inp[:50]}...'"
                 valid_sample = False


        if not valid_sample:
            logger.error(f"跳过结构/类型/长度错误的样本: {error_msg}")
            skipped_invalid_structure += 1 # 增加计数器
            continue # 跳过这个错误的样本

        model_inputs["input_ids"].append(input_ids_full)
        model_inputs["labels"].append(labels)
        processed_count += 1

    # 这部分日志现在主要关注跳过的原因
    if skipped_empty > 0:
        logger.warning(f"在此批次中跳过 {skipped_empty} 个空样本 (应已被预过滤)。")
    if skipped_long_prompt > 0:
        logger.warning(f"在此批次中跳过 {skipped_long_prompt} 个 prompt 过长的样本。")
    if skipped_invalid_structure > 0:
        logger.warning(f"在此批次中跳过 {skipped_invalid_structure} 个结构/类型/长度错误的样本 (详见 ERROR 日志)。")
    # if processed_count == 0 and (skipped_empty > 0 or skipped_long_prompt > 0 or skipped_invalid_structure > 0):
    #      logger.error(f"此批次中的所有样本都被跳过。检查数据质量和 MAX_SEQ_LENGTH。")

    return model_inputs

def main():
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    # 设置随机种子
    set_seed(SEED)
    logger.info(f"随机种子设置为: {SEED}")

    # 检查 GPU
    if not torch.cuda.is_available():
        logger.error("未检测到 GPU，退出程序。")
        exit(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    if torch.cuda.is_available():
        logger.info(f"可用 GPU 数量: {torch.cuda.device_count()}")
        logger.info(f"当前 GPU: {torch.cuda.get_device_name(0)}")

    # --- 加载分词器 ---
    try:
        logger.info(f"加载分词器: {model_path}")
        # 将 tokenizer 定义在 main 开始处，以便后续 filter 和 map 使用
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # --- <<< 移除手动 tokenizer.pad 测试代码 >>> ---
        # logger.info("手动测试 tokenizer.pad() 功能...")
        # ... (测试代码已移除) ...

        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                logger.warning("分词器没有 pad_token。将 pad_token 设置为 eos_token。")
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                logger.error("分词器既没有 pad token 也没有 eos token。无法继续。")
                exit(1)
        # 确保 padding_side 是 right (虽然 CustomCollator 可能覆盖，但保持一致性)
        tokenizer.padding_side = "right"
        logger.info(f"分词器 pad token ID: {tokenizer.pad_token_id}, padding side: {tokenizer.padding_side}")

    except Exception as e:
        logger.error(f"加载分词器失败: {e}", exc_info=True)
        exit(1)

    # --- 加载基座模型 ---
    try:
        logger.info(f"加载基座模型: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            # 考虑移除 attn_implementation 或设置为 'sdpa' (如果支持) 或 'eager'
            # attn_implementation="sdpa", # 例如
            use_cache=False # 训练时必须禁用
        )
        #logger.info(f"模型 attention 实现: {model.config.attn_implementation}") # 打印实际使用的实现

        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        logger.info("模型已启用梯度检查点。")

    except Exception as e:
        logger.error(f"加载模型失败: {e}", exc_info=True)
        exit(1)

    # --- 配置 LoRA ---
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    try:
        logger.info("应用 LoRA 配置...")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    except Exception as e:
        logger.error(f"应用 LoRA 失败: {e}", exc_info=True)
        exit(1)

    # --- 加载、过滤、分割和预过滤数据集 ---
    try:
        logger.info(f"加载数据集: {dataset_path}")
        all_data = load_dataset("json", data_files=dataset_path)

        if "train" in all_data:
            dataset = all_data["train"]
            logger.info("使用 'train' split 作为主数据集。")
        else:
            split_name = next(iter(all_data))
            logger.warning(f"未找到 'train' split，使用数据集中的第一个 split: '{split_name}'")
            dataset = all_data[split_name]

        logger.info(f"原始数据集大小: {len(dataset)}")

        # --- 初始有效性过滤 ---
        original_size = len(dataset)
        def is_valid(example):
            return (example.get("instruction") and isinstance(example["instruction"], str) and
                    example.get("output") and isinstance(example["output"], str))
        dataset = dataset.filter(is_valid, num_proc=NUM_PROC_PREFILTER) # 使用多进程加速过滤
        filtered_size = len(dataset)
        if original_size != filtered_size:
             logger.warning(f"通过 is_valid 过滤掉 {original_size - filtered_size} 个无效样本。")
        if len(dataset) == 0:
            logger.error("初始过滤后数据集为空，请检查数据文件。")
            exit(1)

        # --- 分割数据集 ---
        train_test_split_dict = dataset.train_test_split(test_size=0.2, seed=SEED)
        datasets = DatasetDict({
            "train": train_test_split_dict["train"],
            "test": train_test_split_dict["test"]
        })
        logger.info(f"数据集分割: 训练集 {len(datasets['train'])} 条，测试集 {len(datasets['test'])} 条")

        # --- <<< 应用预过滤步骤 >>> ---
        logger.info(f"开始预过滤数据集 (基于 preprocess_latex 后非空)，使用 {NUM_PROC_PREFILTER} 个进程...")

        original_train_count_prefilter = len(datasets['train'])
        original_test_count_prefilter = len(datasets['test'])

        # 对训练集进行预过滤
        datasets['train'] = datasets['train'].filter(
            filter_problematic_samples,
            num_proc=NUM_PROC_PREFILTER,
            desc="预过滤训练集"
        )
        # 对测试集进行预过滤
        datasets['test'] = datasets['test'].filter(
            filter_problematic_samples,
            num_proc=NUM_PROC_PREFILTER,
            desc="预过滤测试集"
        )

        filtered_train_count_prefilter = len(datasets['train'])
        filtered_test_count_prefilter = len(datasets['test'])

        logger.info(f"预过滤后训练集样本数: {filtered_train_count_prefilter} (移除了 {original_train_count_prefilter - filtered_train_count_prefilter} 个)")
        logger.info(f"预过滤后测试集样本数: {filtered_test_count_prefilter} (移除了 {original_test_count_prefilter - filtered_test_count_prefilter} 个)")

        if filtered_train_count_prefilter == 0:
            logger.error("训练集在预过滤后为空，无法继续训练。")
            exit(1)
        if filtered_test_count_prefilter == 0:
             logger.warning("测试集在预过滤后为空，将无法进行基于 Loss 的评估和生成评估。")
        # --- <<< 预过滤结束 >>> ---

    except Exception as e:
        logger.error(f"加载、过滤或分割数据集失败: {e}", exc_info=True)
        exit(1)


    # --- 分词数据集 (主要映射步骤) ---
    try:
        logger.info(f"开始分词和预处理数据集 (用于训练和 Loss 评估)，使用 {NUM_PROC_MAP} 个进程...")
        # 使用 functools.partial 传递 tokenizer 到 map 函数
        preprocess_fn_with_tokenizer = partial(preprocess_function_for_training, tokenizer=tokenizer)

        tokenized_datasets = datasets.map(
            preprocess_fn_with_tokenizer,
            batched=True,
            remove_columns=datasets["train"].column_names, # 移除原始列
            num_proc=NUM_PROC_MAP,
            desc="运行分词器处理数据集",
            load_from_cache_file=True # 建议保留缓存
        )

        # --- 后续长度过滤 (安全检查) ---
        logger.info("过滤掉分词后可能为空或结构错误的样本 (理论上预处理函数已处理)...") # 更新日志信息
        original_train_count = len(tokenized_datasets['train'])
        original_test_count = len(tokenized_datasets['test'])

        # 过滤掉 input_ids 为空的 (安全检查)
        tokenized_datasets = tokenized_datasets.filter(lambda example: len(example.get('input_ids', [])) > 0, num_proc=NUM_PROC_PREFILTER)
        # 过滤掉 labels 为空的 (安全检查)
        tokenized_datasets = tokenized_datasets.filter(lambda example: len(example.get('labels', [])) > 0, num_proc=NUM_PROC_PREFILTER)
        # 过滤掉 input_ids 和 labels 长度不一致的 (终极安全检查)
        tokenized_datasets = tokenized_datasets.filter(lambda x: len(x['input_ids']) == len(x['labels']), num_proc=NUM_PROC_PREFILTER)


        filtered_train_count = len(tokenized_datasets['train'])
        filtered_test_count = len(tokenized_datasets['test'])

        logger.info(f"分词和最终过滤后训练集样本数: {filtered_train_count} (移除了 {original_train_count - filtered_train_count} 个)")
        logger.info(f"分词和最终过滤后测试集样本数: {filtered_test_count} (移除了 {original_test_count - filtered_test_count} 个)")

        if filtered_train_count == 0:
            logger.error("训练集在分词和过滤后为空，无法继续训练。")
            exit(1)
        if filtered_test_count == 0:
            logger.warning("测试集在分词和过滤后为空。将无法进行基于 Loss 的评估。")

        # 抽查一个样本
        if filtered_train_count > 0:
            sample = tokenized_datasets['train'][0]
            logger.info(f"抽样样本 keys: {list(sample.keys())}") # 应该只剩 input_ids, labels
            logger.info(f"抽样样本 input_ids 长度: {len(sample['input_ids'])}")
            logger.info(f"抽样样本 labels 长度: {len(sample['labels'])}")
            # logger.info(f"抽样样本解码 input_ids: {tokenizer.decode(sample['input_ids'][:50])}...")
            # logger.info(f"抽样样本解码 labels (非 -100): {tokenizer.decode([l for l in sample['labels'] if l != -100][:50])}...")

    except Exception as e:
        logger.error(f"分词或后续过滤数据集失败: {e}", exc_info=True)
        exit(1)

    # --- <<< 使用 CustomDataCollator 进行动态 Padding >>> ---
    data_collator = CustomDataCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 # 可以根据需要调整或移除
    )
    logger.info("使用 CustomDataCollator。")


    # --- 训练参数 ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=False,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        fp16=True,
        logging_steps=LOGGING_STEPS,
        log_level="info",
        logging_strategy="steps",
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        eval_strategy="no", # 保持 steps，但在 Trainer 初始化时检查 eval_dataset
        #eval_steps=EVAL_STEPS,
        load_best_model_at_end=False, # 保持 True，但在 Trainer 初始化时检查
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to=["tensorboard"],
        logging_dir=tensorboard_log_dir,
        seed=SEED,
        data_seed=SEED,
        remove_unused_columns=True, # <<<--- 必须为 True，配合 CustomDataCollator
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        group_by_length=False, # <-- 初始测试设置为 False
        #group_by_length=True, # <-- 如果 False 成功，可以尝试改回 True 以提高效率
        #length_column_name="input_ids", # <-- 如果 group_by_length=True，指定长度列 (tokenized 后才有)
        dataloader_num_workers= 0,
        dataloader_pin_memory=True,
    )
    logger.info(f"训练参数: {training_args.to_dict()}")

    # --- 简化的 compute_metrics ---
    def compute_metrics(eval_pred):
        logger.debug("compute_metrics called. Relying on Trainer's automatic eval_loss logging.")
        # Trainer 会自动计算并记录 eval_loss
        return {}

    # --- 内存监控回调 ---
    class MemoryCallback(TrainerCallback):
        def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            if torch.cuda.is_available() and state.is_local_process_zero and state.global_step > 0 and state.global_step % args.logging_steps == 0:
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"Step {state.global_step} | Mem: {allocated:.2f} GiB allocated, {reserved:.2f} GiB reserved")

    # --- 初始化 Trainer ---
    try:
        # 确保测试集非空才传入 eval_dataset
        eval_dataset_for_trainer = tokenized_datasets["test"] if filtered_test_count > 0 else None
        final_eval_strategy = training_args.eval_strategy
        final_load_best = training_args.load_best_model_at_end

        if eval_dataset_for_trainer is None:
             logger.warning("测试集为空，Trainer 将不会进行评估步骤 (eval_steps 将被忽略)。")
             # 如果没有评估集，evaluation_strategy 必须是 "no"
             if final_eval_strategy != "no":
                 logger.warning(f"将 evaluation_strategy 从 '{final_eval_strategy}' 修改为 'no'。")
                 final_eval_strategy = "no"
             if final_load_best:
                 logger.warning("将 load_best_model_at_end 修改为 False。")
                 final_load_best = False
             # 更新 TrainingArguments 实例中的值 (如果需要显式修改)
             training_args.evaluation_strategy = final_eval_strategy
             training_args.load_best_model_at_end = final_load_best


        trainer = Trainer(
            model=model,
            args=training_args, # 使用可能已修改的 training_args
            train_dataset=tokenized_datasets["train"],
            eval_dataset=eval_dataset_for_trainer,
            tokenizer=tokenizer,
            data_collator=data_collator, # <-- 使用 CustomDataCollator
            compute_metrics=compute_metrics,
            callbacks=[MemoryCallback()]
        )
        logger.info("Trainer 初始化成功。")
    except Exception as e:
        logger.error(f"初始化 Trainer 失败: {e}", exc_info=True)
        exit(1)

    # --- 开始训练 ---
    train_successful = False
    try:
        logger.info("*" * 50)
        logger.info("开始训练...")
        logger.info("*" * 50)

        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        resume_from_checkpoint = None
        if last_checkpoint is not None:
            logger.info(f"检测到检查点，将在 {last_checkpoint} 恢复训练。")
            resume_from_checkpoint = last_checkpoint
        elif os.path.exists(training_args.output_dir) and os.listdir(training_args.output_dir):
             # 检查是否包含 adapter_config.json 等 LoRA 文件，而不仅仅是 checkpoint-* 目录
             is_adapter_dir = any(f == 'adapter_config.json' for f in os.listdir(training_args.output_dir))
             if not is_adapter_dir and any(f.startswith('checkpoint-') for f in os.listdir(training_args.output_dir)):
                 logger.warning(f"输出目录 {training_args.output_dir} 包含 checkpoint 但找不到根目录下的 adapter 文件。可能需要清理或更改 output_dir。")
             elif not is_adapter_dir:
                 logger.info(f"输出目录 {training_args.output_dir} 非空但似乎不包含有效 checkpoint 或 adapter。")


        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        train_successful = True

        logger.info("训练完成，保存最终模型状态 (LoRA adapter)...")
        # trainer.save_model() 会保存 adapter 到 output_dir
        # 如果 load_best_model_at_end=True，则保存的是最佳模型
        # 如果为 False 或没有评估，则保存的是训练结束时的模型
        trainer.save_model(output_dir)
        trainer.save_state()
        tokenizer.save_pretrained(output_dir)
        logger.info(f"最终模型适配器 (adapter) 已保存至: {output_dir}")

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        train_metrics_path = os.path.join(output_dir, "train_results.json")
        trainer.save_metrics("train", metrics) # 直接传递 metrics 字典
        logger.info(f"训练指标已保存至: {train_metrics_path}")

        # 进行基于 Loss 的最终评估 (仅当有评估集且执行了评估)
        # 注意: 如果 eval_dataset_for_trainer 为 None 或 evaluation_strategy="no", 则 evaluate 不会执行
        if training_args.do_eval and eval_dataset_for_trainer is not None and training_args.evaluation_strategy != "no":
            logger.info("进行最终 Loss 评估...")
            # load_best_model_at_end=True 时，trainer.model 已经是最佳模型
            eval_metrics = trainer.evaluate(eval_dataset=eval_dataset_for_trainer)

            try:
                perplexity = math.exp(eval_metrics["eval_loss"])
                eval_metrics["perplexity"] = perplexity
                logger.info(f"最终评估 Perplexity: {perplexity:.4f}")
            except KeyError:
                 logger.warning("无法在评估结果中找到 'eval_loss' 来计算 Perplexity。")
            except OverflowError:
                 eval_metrics["perplexity"] = float("inf")
                 logger.warning("计算 Perplexity 时发生溢出 (loss 可能过高)。")

            trainer.log_metrics("eval", eval_metrics)
            eval_metrics_path = os.path.join(output_dir, "eval_results_loss.json")
            trainer.save_metrics("eval", eval_metrics) # 直接传递 metrics 字典
            logger.info(f"Loss 评估指标已保存至: {eval_metrics_path}")
        elif training_args.evaluation_strategy == "no":
             logger.info("跳过最终 Loss 评估，因为 evaluation_strategy='no'。")

    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}", exc_info=True)
        if trainer is not None:
            logger.warning("训练异常终止，尝试保存当前模型状态...")
            try:
                error_save_path = os.path.join(output_dir, "checkpoint-last-on-error")
                trainer.save_model(error_save_path)
                tokenizer.save_pretrained(error_save_path)
                logger.info(f"当前模型状态已尝试保存至: {error_save_path}")
            except Exception as save_e:
                logger.error(f"尝试保存失败模型时出错: {save_e}")
    finally:
        if torch.cuda.is_available():
            logger.info("训练流程结束（或异常终止），清理 CUDA 缓存...")
            # del model # 尝试显式删除模型和 trainer (在生成评估前统一处理)
            # del trainer
            torch.cuda.empty_cache()
            logger.info("CUDA 缓存已清理。")

    # --- 方法一：训练后单独进行生成评估 ---
    # 条件：训练成功，并且有测试数据用于生成
    # 注意：即使没有基于loss的评估，只要训练成功且有测试集，也可以进行生成评估
    # 使用原始的、未分词的、但经过预过滤的测试集
    raw_test_dataset_for_gen = datasets["test"] # 使用预过滤后的原始文本数据
    if train_successful and len(raw_test_dataset_for_gen) > 0:
        logger.info("=" * 50)
        logger.info("训练流程成功结束。开始进行基于生成的最终评估...")
        logger.info("=" * 50)

        # 加载用于评估的模型
        # 如果 load_best_model_at_end=True 且执行了评估，trainer.model 是最佳模型
        # 否则，它是训练结束时的模型。或者我们可以总是从保存的 output_dir 加载
        logger.info(f"从 {output_dir} 加载最终保存的 PEFT 模型进行生成评估...")
        try:
            # 释放 GPU 显存
            if 'model' in locals(): del model # 删除训练时的模型对象
            if 'trainer' in locals(): del trainer
            if torch.cuda.is_available(): torch.cuda.empty_cache()

            # 重新加载基础模型和最佳 adapter
            base_model_for_eval = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map="auto", # 重新加载到 GPU
                #attn_implementation="sdpa", # 保持与训练时一致或设为推理优化选项
            )
            model_to_eval = PeftModel.from_pretrained(base_model_for_eval, output_dir)
            # model_to_eval = model_to_eval.merge_and_unload() # 可选: 合并权重（需要足够内存，推理更快）
            # model_to_eval.to(device) # 如果 device_map="auto" 不需要这行
            tokenizer_to_eval = AutoTokenizer.from_pretrained(output_dir) # 使用保存的 tokenizer
            # 确保 pad token 设置正确
            if tokenizer_to_eval.pad_token is None and tokenizer_to_eval.eos_token is not None:
                 tokenizer_to_eval.pad_token = tokenizer_to_eval.eos_token
                 tokenizer_to_eval.pad_token_id = tokenizer_to_eval.eos_token_id

            model_to_eval.eval() # 设置为评估模式
            current_model = model_to_eval
            current_tokenizer = tokenizer_to_eval
            logger.info("从磁盘加载最终模型进行生成评估完成。")
        except Exception as load_e:
             logger.error(f"加载最终模型进行生成评估失败: {load_e}", exc_info=True)
             current_model = None # 标记模型加载失败


        if current_model and current_tokenizer:
            all_preds = []
            all_labels = []

            generation_config = {
                "max_new_tokens": MAX_SEQ_LENGTH // 2, # 限制新生成 token 数量
                "do_sample": False,
                "num_beams": 1,
                "pad_token_id": current_tokenizer.pad_token_id,
                "eos_token_id": current_tokenizer.eos_token_id,
            }
            logger.info(f"生成参数: {generation_config}")

            # --- 检查 nltk punkt 分词器 ---
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                logger.info("未找到 NLTK punkt 分词器。正在下载...")
                try:
                    nltk.download("punkt", quiet=True)
                    logger.info("NLTK punkt 下载成功。")
                except Exception as e:
                    logger.error(f"下载 NLTK punkt 失败。错误: {e}")

            logger.info(f"开始为 {len(raw_test_dataset_for_gen)} 个测试样本生成预测文本...")
            inference_batch_size = EVAL_BATCH_SIZE # 使用评估批次大小进行推理
            for i in tqdm(range(0, len(raw_test_dataset_for_gen), inference_batch_size), desc="Generating Predictions"):
                end_index = min(i + inference_batch_size, len(raw_test_dataset_for_gen))
                batch_instructions = raw_test_dataset_for_gen[i:end_index]["instruction"]
                batch_label_texts = raw_test_dataset_for_gen[i:end_index]["output"]

                # 对文本进行最后的清理 (虽然已预过滤，但保险起见)
                batch_instructions = [preprocess_latex(instr) for instr in batch_instructions]
                batch_label_texts = [preprocess_latex(lbl) for lbl in batch_label_texts]

                # 构建 Prompts
                prompts = [f"[问题] {instr} [回答]" for instr in batch_instructions if instr] # 确保 instr 非空
                current_labels = [lbl for instr, lbl in zip(batch_instructions, batch_label_texts) if instr and lbl] # 保持对应

                if not prompts:
                    logger.debug(f"Skipping batch starting at index {i}: no valid prompts after final check.")
                    continue

                original_padding_side = current_tokenizer.padding_side
                current_tokenizer.padding_side = "left" # 生成时推荐左填充
                inputs = current_tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_SEQ_LENGTH // 2, # 限制输入 prompt 长度, 留足生成空间
                    return_attention_mask=True
                ).to(device) # 确保输入在 GPU 上
                current_tokenizer.padding_side = original_padding_side

                try:
                    # 使用 FP16 推理 (TrainingArguments 可能不再可用，手动设置)
                    with torch.cuda.amp.autocast(enabled=True):
                        with torch.no_grad():
                            outputs = current_model.generate(
                                input_ids=inputs["input_ids"],
                                attention_mask=inputs["attention_mask"],
                                **generation_config
                            )

                    # 只解码生成的部分 (更精确)
                    generated_ids = outputs[:, inputs['input_ids'].shape[1]:]
                    batch_preds = current_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                    # 对解码结果进行 strip
                    batch_preds = [pred.strip() for pred in batch_preds]


                    # # 旧的解码逻辑 (保留作为注释，以防新逻辑有问题)
                    # decoded_outputs = current_tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    # batch_preds_old = []
                    # for idx, full_decoded_text in enumerate(decoded_outputs):
                    #     input_prompt_text = current_tokenizer.decode(inputs['input_ids'][idx], skip_special_tokens=True)
                    #     answer_marker = "[回答]"
                    #     marker_pos = full_decoded_text.rfind(answer_marker) # 用 rfind 从后找，可能更稳定
                    #     if marker_pos != -1:
                    #         pred_text = full_decoded_text[marker_pos + len(answer_marker):].strip()
                    #     else:
                    #          # 尝试从头找
                    #          marker_pos = full_decoded_text.find(answer_marker)
                    #          if marker_pos != -1:
                    #               pred_text = full_decoded_text[marker_pos + len(answer_marker):].strip()
                    #          else:
                    #              logger.warning(f"生成结果中未找到 '[回答]' 标记 (样本索引约 {i+idx}): '{full_decoded_text[:150]}...'。将使用完整解码文本作为预测。")
                    #              pred_text = full_decoded_text.strip() # Fallback: 使用整个解码结果
                    #     batch_preds_old.append(pred_text)
                    # # print("New preds:", batch_preds)
                    # # print("Old preds:", batch_preds_old)

                    all_preds.extend(batch_preds)
                    all_labels.extend(current_labels) # current_labels 已过滤匹配 prompts

                except Exception as gen_e:
                    logger.error(f"批量生成预测时出错 (批次起始索引 {i}): {gen_e}", exc_info=True) # 打印完整 Traceback
                    all_preds.extend(["<GENERATION_ERROR>"] * len(current_labels)) # 标记错误
                    all_labels.extend(current_labels)

            # --- 计算 ROUGE/BLEU ---
            logger.info(f"生成完成 ({len(all_preds)} predictions generated)。开始计算 ROUGE/BLEU 指标...")
            rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            smoothie = SmoothingFunction().method4

            bleu_scores = []
            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []
            valid_pairs_count = 0

            for pred, label in zip(all_preds, all_labels):
                if isinstance(pred, str) and isinstance(label, str) and pred != "<GENERATION_ERROR>" and label: # 确保标签非空，预测非错误标记
                    # 如果预测为空字符串也计算，这反映模型生成了空内容
                    try:
                        # ROUGE
                        rouge_scores_single = rouge.score(label, pred)
                        rouge1_scores.append(rouge_scores_single['rouge1'].fmeasure)
                        rouge2_scores.append(rouge_scores_single['rouge2'].fmeasure)
                        rougeL_scores.append(rouge_scores_single['rougeL'].fmeasure)

                        # BLEU
                        reference_tokens = [nltk.word_tokenize(label)] # 使用 NLTK 分词
                        prediction_tokens = nltk.word_tokenize(pred)   # 使用 NLTK 分词
                        bleu_scores.append(sentence_bleu(reference_tokens, prediction_tokens, smoothing_function=smoothie))

                        valid_pairs_count += 1
                    except ValueError as ve:
                         logger.warning(f"计算指标时遇到 ValueError: {ve}. Pred='{pred[:50]}...', Label='{label[:50]}...'")
                    except Exception as metric_e:
                        logger.warning(f"计算指标时出错: Pred='{pred[:50]}...', Label='{label[:50]}...': {metric_e}")
                else:
                    # logger.debug(f"跳过指标计算: Pred='{pred}', Label='{label}'")
                    pass # 跳过无效对

            # --- 聚合和记录最终生成指标 ---
            if valid_pairs_count > 0:
                final_metrics = {
                    "final_bleu": np.mean(bleu_scores) * 100,
                    "final_rouge1": np.mean(rouge1_scores) * 100,
                    "final_rouge2": np.mean(rouge2_scores) * 100,
                    "final_rougeL": np.mean(rougeL_scores) * 100,
                    "evaluated_pairs_count": valid_pairs_count,
                    "total_test_samples_for_gen": len(raw_test_dataset_for_gen),
                    "generation_config": generation_config
                }
                logger.info(f"最终生成评估指标 (基于 {valid_pairs_count} 个有效对):")
                for key, value in final_metrics.items():
                    if isinstance(value, (float, np.floating)): # 检查 numpy float
                         logger.info(f"  {key}: {value:.4f}")
                    else:
                         logger.info(f"  {key}: {value}")

                final_metrics_path = os.path.join(output_dir, "final_generation_metrics.json")
                try:
                    # 自定义 JSON 编码器处理 numpy 类型
                    class NpEncoder(json.JSONEncoder):
                        def default(self, obj):
                            if isinstance(obj, np.integer): return int(obj)
                            if isinstance(obj, np.floating): return float(obj)
                            if isinstance(obj, np.ndarray): return obj.tolist()
                            return super(NpEncoder, self).default(obj)
                    with open(final_metrics_path, "w", encoding='utf-8') as f:
                        json.dump(final_metrics, f, indent=4, cls=NpEncoder)
                    logger.info(f"最终生成指标已保存至: {final_metrics_path}")
                except Exception as write_e:
                    logger.error(f"保存最终生成指标失败: {write_e}")
            else:
                logger.warning("未能成功计算任何有效的生成指标。请检查生成过程或测试数据。")

    elif not train_successful:
         logger.error("训练未成功完成，跳过最终生成评估。")
    elif len(datasets.get("test", [])) == 0 : # 使用 .get 以防 'test' 不存在
         logger.warning("测试集为空，跳过最终生成评估。")


    logger.info("=" * 50)
    logger.info("脚本执行结束。")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()
