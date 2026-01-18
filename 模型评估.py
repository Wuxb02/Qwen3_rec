# %% [markdown]
# # 模型对比评估完整脚本 (vLLM 串行防 OOM 版 - 带输入输出日志)
# 
# **核心修正点**: 
# 1. **OOM防护**: 串行加载模型，用完即毁。
# 2. **长文本支持**: max_model_len 提升至 8192。
# 3. **思考链支持**: 移除 stop 符限制，并在后处理中自动过滤 <think> 标签。
# 4. **日志增强**: 推理完成后自动打印前5条完整的输入与输出。

# %%
# Cell 1: 导入必要的库
import json
import gc
import torch
import time
import re
import pandas as pd
import numpy as np
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# 设置 pandas 显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)

# %%
# Cell 2: 配置与数据加载
print("=" * 60)
print("步骤 1: 准备环境与数据")
print("=" * 60)

# === 路径配置 ===
BASE_MODEL_PATH = "./Qwen/Qwen3-8B"
FINETUNED_MODEL_PATH = "./merged_models/qwen3_mind_news_recommend" 
TEST_DATA_PATH = "./data/processed/mind_test.json"
RESULT_SAVE_PATH = "./evaluation_results_full.json"

# === 加载测试数据 ===
print(f"正在读取测试集: {TEST_DATA_PATH}")
with open(TEST_DATA_PATH, "r", encoding='utf-8') as f:
    test_data = json.load(f)

# === 采样设置 ===
TEST_SAMPLE_SIZE = None 
test_samples = test_data[:TEST_SAMPLE_SIZE]
print(f"本次评估样本数: {len(test_samples)}")

# === 构建 Prompts ===
def build_prompt(sample):
    return f"<|im_start|>system\n{sample['instruction']}<|im_end|>\n<|im_start|>user\n{sample['input']}<|im_end|>\n<|im_start|>assistant\n"

print("正在构建 Prompts...")
prompts = [build_prompt(s) for s in test_samples]
labels = [s["output"].strip() for s in test_samples]
print(f"✓ Prompts 构建完成，共 {len(prompts)} 条")

print("正在按长度排序 Prompts (优化 Batch 效率)...")
# 同时记录原始索引，以便后续和 labels 对齐（如果 labels 顺序很重要）
prompts_with_index = list(enumerate(prompts))
prompts_with_index.sort(key=lambda x: len(x[1]), reverse=True) 

prompts = [p for i, p in prompts_with_index]
sorted_indices = [i for i, p in prompts_with_index]
# 注意：如果后续需要和原始 labels 对应，记得 evaluation 时要 re-map 回去
# 或者简单点：直接把 labels 也同步排序
labels = [labels[i] for i in sorted_indices]

# === 采样参数 (修正版) ===
sampling_params = SamplingParams(
    temperature=0.1,    
    max_tokens=1024,     # 给足空间让模型思考
    stop=["<|im_end|>", "<|endoftext|>"],            # 移除 \n 停止符，防止思考过程中断
    stop_token_ids=[151645, 151643] # Qwen 的特殊 token id，确保读完即停
)

# %%
# Cell 3: 定义显存清理工具函数
def clean_gpu_memory():
    """强制清理显存"""
    print("   [System] 正在清理显存...")
    gc.collect()
    torch.cuda.empty_cache()
    try:
        from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except:
        pass
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"   [System] 显存状态: 已用 {allocated:.2f}GB")

# %%
# Cell 4: 评估基座模型 (Phase 1)
print("\n" + "=" * 60)
print("步骤 2: 评估基座模型 (Base Model)")
print("=" * 60)

# 1. 加载模型 (修正 max_model_len)
print(f"正在加载基座模型: {BASE_MODEL_PATH}")
llm_base = LLM(
    model=BASE_MODEL_PATH,
    enable_prefix_caching=True,
    gpu_memory_utilization=0.9,
    tensor_parallel_size=2,
    max_model_len=8192,      # <--- 修正点：支持长文本
    trust_remote_code=True,
    dtype="bfloat16"
)

# 2. 推理
print("开始基座模型推理...")
start_time = time.time()
outputs_base = llm_base.generate(prompts, sampling_params, use_tqdm=True)
elapsed_base = time.time() - start_time

# 3. 提取结果 (原始文本)
pred_base_raw = [o.outputs[0].text for o in outputs_base]

# === 打印前5条完整内容 ===
print("\n" + "★" * 50)
print("【基座模型 (Base) - 推理结果采样展示 (前5条)】")
for i in range(min(2, len(prompts))):
    print(f"\n>>> Sample {i+1} <<<")
    print(f"【Input / Prompt】:\n{prompts[i].strip()}")
    print("-" * 30)
    print(f"【Output / Generation】:\n{pred_base_raw[i].strip()}")
    print("=" * 50)
print("★" * 50 + "\n")
# ========================

print(f"✓ 基座模型评估完成 (耗时: {elapsed_base:.2f}s)")

# 4. 彻底销毁模型
try:
    destroy_model_parallel()
except:
    pass
del llm_base
clean_gpu_memory()
print("✓ 基座模型已完全卸载")

# %%
# Cell 5: 评估微调模型 (Phase 2)
print("\n" + "=" * 60)
print("步骤 3: 评估微调模型 (Finetuned Model)")
print("=" * 60)

# 1. 加载模型 (修正 max_model_len)
print(f"正在加载微调模型: {FINETUNED_MODEL_PATH}")
llm_finetuned = LLM(
    model=FINETUNED_MODEL_PATH,
    enable_prefix_caching=True,
    gpu_memory_utilization=0.9,
    tensor_parallel_size=2,
    max_model_len=8192,      # <--- 修正点：保持一致
    trust_remote_code=True,
    dtype="bfloat16"
)

# 2. 推理
print("开始微调模型推理...")
start_time = time.time()
outputs_finetuned = llm_finetuned.generate(prompts, sampling_params, use_tqdm=True)
elapsed_finetuned = time.time() - start_time

# 3. 提取结果 (原始文本)
pred_finetuned_raw = [o.outputs[0].text for o in outputs_finetuned]

# === 打印前5条完整内容 ===
print("\n" + "★" * 50)
print("【微调模型 (Finetuned) - 推理结果采样展示 (前5条)】")
for i in range(min(2, len(prompts))):
    print(f"\n>>> Sample {i+1} <<<")
    print(f"【Input / Prompt】:\n{prompts[i].strip()}")
    print("-" * 30)
    print(f"【Output / Generation】:\n{pred_finetuned_raw[i].strip()}")
    print("=" * 50)
print("★" * 50 + "\n")
# ========================

print(f"✓ 微调模型评估完成 (耗时: {elapsed_finetuned:.2f}s)")

# 4. 彻底销毁模型
try:
    destroy_model_parallel()
except:
    pass
del llm_finetuned
clean_gpu_memory()
print("✓ 微调模型已完全卸载")

# %%
# Cell 6: 结果清洗与评估函数 (修正版)
def parse_prediction(pred_text):
    """
    智能提取答案：移除 <think>...</think> 并提取 Yes/No
    """
    if not isinstance(pred_text, str):
        return "No"
    
    # 移除 <think> 标签及其内容
    clean_text = re.sub(r'<think>.*?</think>', '', pred_text, flags=re.DOTALL)
    if '<think>' in clean_text: # 处理未闭合的情况
        clean_text = clean_text.split('<think>')[0]
        
    text = clean_text.lower().strip()
    
    # 模糊匹配
    if "yes" in text: return "Yes"
    if "no" in text: return "No"
    
    return "No" # 默认兜底

def evaluate_comprehensive(predictions, labels):
    # 1. 清洗数据
    clean_preds = [parse_prediction(p) for p in predictions]
    clean_labels = [parse_prediction(l) for l in labels] # 标签通常很干净，但也处理一下以防万一

    # Debug: 打印前3条清洗效果
    print("[Debug] 清洗效果示例 (Parse Check):")
    for i in range(min(3, len(predictions))):
        print(f"  Raw: {repr(predictions[i][:50])}...")
        print(f"  Cln: {clean_preds[i]}")

    # 2. 计算混淆矩阵
    cm = confusion_matrix(clean_labels, clean_preds, labels=["No", "Yes"])
    tn, fp, fn, tp = cm.ravel()

    # 3. 计算指标
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    
    # 正类 (Yes)
    p_pos = tp / (tp + fp) if (tp + fp) > 0 else 0
    r_pos = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_pos = 2 * p_pos * r_pos / (p_pos + r_pos) if (p_pos + r_pos) > 0 else 0
    
    # 负类 (No)
    p_neg = tn / (tn + fn) if (tn + fn) > 0 else 0
    r_neg = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_neg = 2 * p_neg * r_neg / (p_neg + r_neg) if (p_neg + r_neg) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "confusion_matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
        "positive_class": {"precision": p_pos, "recall": r_pos, "f1_score": f1_pos},
        "negative_class": {"precision": p_neg, "recall": r_neg, "f1_score": f1_neg},
        "macro_avg": {"f1_score": (f1_pos + f1_neg) / 2}
    }

print("\n正在计算对比指标...")
# 注意：这里传入的是 raw 数据，交给函数内部去清洗
metrics_base = evaluate_comprehensive(pred_base_raw, labels)
metrics_finetuned = evaluate_comprehensive(pred_finetuned_raw, labels)
print("✓ 指标计算完成")

# %%
# Cell 7: 生成并展示对比报告
print("\n" + "=" * 80)
print("最终对比报告")
print("=" * 80)

# 1. 核心指标对比表
comparison_data = {
    "指标": ["准确率", "F1 (Yes)", "Recall (Yes)", "Precision (Yes)", "F1 (No)", "Macro F1"],
    "基座模型": [
        metrics_base['accuracy'], metrics_base['positive_class']['f1_score'], 
        metrics_base['positive_class']['recall'], metrics_base['positive_class']['precision'],
        metrics_base['negative_class']['f1_score'], metrics_base['macro_avg']['f1_score']
    ],
    "微调模型": [
        metrics_finetuned['accuracy'], metrics_finetuned['positive_class']['f1_score'],
        metrics_finetuned['positive_class']['recall'], metrics_finetuned['positive_class']['precision'],
        metrics_finetuned['negative_class']['f1_score'], metrics_finetuned['macro_avg']['f1_score']
    ]
}

df_metrics = pd.DataFrame(comparison_data)
df_metrics["提升"] = df_metrics["微调模型"] - df_metrics["基座模型"]

# 格式化输出
pd.options.display.float_format = '{:,.4f}'.format
print("\n【核心指标概览】")
print(df_metrics.to_string(index=False))

# 2. 混淆矩阵
print("\n【混淆矩阵对比】")
print(f"{'':<10} {'TP':<8} {'TN':<8} {'FP':<8} {'FN':<8}")
cm_b = metrics_base['confusion_matrix']
cm_f = metrics_finetuned['confusion_matrix']
print(f"{'基座':<10} {cm_b['TP']:<8} {cm_b['TN']:<8} {cm_b['FP']:<8} {cm_b['FN']:<8}")
print(f"{'微调':<10} {cm_f['TP']:<8} {cm_f['TN']:<8} {cm_f['FP']:<8} {cm_f['FN']:<8}")

# 3. 速度
print("\n【推理速度】")
print(f"基座: {len(prompts)/elapsed_base:.2f} samples/s")
print(f"微调: {len(prompts)/elapsed_finetuned:.2f} samples/s")

# %%
# Cell 8: 保存结果
final_results = {
    "metrics_base": metrics_base,
    "metrics_finetuned": metrics_finetuned,
    "inference_time": {"base": elapsed_base, "finetuned": elapsed_finetuned}
}
with open(RESULT_SAVE_PATH, "w", encoding='utf-8') as f:
    json.dump(final_results, f, indent=4)
print(f"\n✓ 结果已保存至: {RESULT_SAVE_PATH}")