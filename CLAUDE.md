# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个基于 Qwen3-8B 的新闻推荐模型微调项目,使用 MIND 数据集训练模型来预测用户是否会点击特定新闻。项目采用 LoRA 微调方法进行监督式微调(SFT),并集成了 vLLM 进行高性能推理评估。

**注**:
- 项目已从 ChatGLM-4-9B 迁移到 Qwen3-8B
- 完善了数据处理流程,现在包含 train/val/test 三个数据集
- 集成 vLLM 实现 18-25 倍评估加速

## 运行环境

- Python 解释器路径: `D:\anaconda\python.exe`
- 基础模型: Qwen3-8B (从 ModelScope 下载到 `./Qwen/Qwen3-8B`)
- 微调后的适配器: `./saves/qwen3_mind_news_recommend`

## 核心命令

### 0. 模型下载 (首次运行)
从 ModelScope 下载 Qwen3-8B 模型:
```bash
# 安装 ModelScope
pip install modelscope

# 运行下载脚本
"D:\anaconda\python.exe" download_qwen_model.py
```

下载完成后,模型将保存在 `./Qwen/Qwen3-8B/` 目录(约 28GB)。

### 1. 数据预处理
使用 Jupyter Notebook 运行数据预处理:
```bash
"D:\anaconda\python.exe" -m jupyter notebook "数据预处理.ipynb"
```

数据预处理流程:
- 读取 MIND 数据集的 train/val/test 三个目录的 `news.tsv` 和 `behaviors.tsv`
- 构建新闻 ID 到标题的映射
- 生成训练样本,格式包含用户点击历史、喜欢/不喜欢的新闻、目标新闻
- 输出到:
  - `./data/processed/mind_train.json` (约 125,000 样本)
  - `./data/processed/mind_val.json` (约 35,000 样本)
  - `./data/processed/mind_test.json` (约 31,000 样本)

### 2. 模型训练
使用 LLaMA-Factory 框架进行训练:
```bash
# 安装 LLaMA-Factory (如未安装)
pip install llmtuner

# 启动训练
llamafactory-cli train new_train.yaml
```

训练配置 ([new_train.yaml](new_train.yaml)):
- 模型: Qwen3-8B (从 `./Qwen/Qwen3-8B` 加载)
- 模板: qwen
- 微调方法: LoRA
- LoRA target: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj (7层)
- 数据集: mind (train) + 10% 自动划分作为验证集
- 批次大小: 1 per device, 梯度累积 8 步
- 学习率: 1.0e-4, Cosine 调度器
- 训练轮数: 3 epochs
- 最大样本数: 1000 (可调整或去掉以使用全部数据)
- 输出目录: `./saves/qwen3_mind_news_recommend`

### 3. LoRA 权重合并 (vLLM 加速推理必需)
合并 LoRA 适配器到基座模型,生成完整的微调模型供 vLLM 使用:
```bash
# 安装依赖
pip install peft vllm

# 合并权重
"D:\anaconda\python.exe" merge_lora_weights.py
```

合并后的模型将保存到 `./merged_models/qwen3_mind_news_recommend/` (约 28GB)。

### 4. 模型推理与评估

#### 4.1 方式一: vLLM 批量推理 (推荐, 18-25倍加速)
```bash
"D:\anaconda\python.exe" -m jupyter notebook "模型对比评估_vllm.ipynb"
```

vLLM 评估脚本功能:
- 使用 vLLM 进行高性能批量推理
- 同时测试基座模型和合并后的微调模型
- 评估速度提升 18-25 倍 (vs. API 方式)
- 显存利用率提升 2-3 倍
- 对比两个模型在测试集上的准确率
- 生成详细的性能对比报告(包含推理速度)
- 输出错误案例分析
- 结果保存到 `evaluation_results_vllm.json`

**性能对比**:
- 1000 样本评估: 2-5 分钟 (vs. API 方式 30-50 分钟)
- 全量 31,000 样本: 25-50 分钟 (vs. API 方式 20+ 小时)
- 吞吐量: 10-30 samples/sec (vs. API 方式 0.3-0.5 samples/sec)

#### 4.2 方式二: API 服务推理 (备选)
```bash
# 启动基座模型服务 (端口 8000)
llamafactory-cli api \
  --model_name_or_path ./Qwen/Qwen3-8B \
  --template qwen \
  --port 8000

# 启动微调模型服务 (端口 8001)
llamafactory-cli api \
  --model_name_or_path ./Qwen/Qwen3-8B \
  --adapter_name_or_path ./saves/qwen3_mind_news_recommend \
  --template qwen \
  --finetuning_type lora \
  --port 8001

# 运行评估
"D:\anaconda\python.exe" -m jupyter notebook "模型对比评估.ipynb"
```

API 评估脚本功能:
- 通过 OpenAI SDK 调用 API 服务
- 同时测试基座模型和微调模型
- 对比两个模型在测试集上的准确率
- 生成详细的性能对比报告
- 输出错误案例分析
- 结果保存到 `evaluation_results.json`

## 数据结构

### MIND 数据集结构
```
data/
├── MIND/
│   ├── train/
│   │   ├── news.tsv        # 训练集新闻 (101,527条)
│   │   └── behaviors.tsv   # 训练集用户行为
│   ├── val/
│   │   ├── news.tsv        # 验证集新闻
│   │   └── behaviors.tsv   # 验证集用户行为
│   └── test/
│       ├── news.tsv        # 测试集新闻 (120,959条)
│       └── behaviors.tsv   # 测试集用户行为
└── processed/
    ├── mind_train.json     # 预处理后的训练样本 (~125,000条)
    ├── mind_val.json       # 预处理后的验证样本 (~35,000条)
    └── mind_test.json      # 预处理后的测试样本 (~31,000条)
```

### 样本格式
每个样本包含:
- `instruction`: 推荐任务的系统提示词
- `input`: 用户点击历史 + 喜欢的新闻 + 不喜欢的新闻 + 目标新闻
- `output`: "Yes." 或 "No."
- `用户ID`: 用户标识符

## LoRA 配置

LoRA 微调参数:
- `r`: 8 (LoRA rank)
- `lora_alpha`: 16
- `target_modules`: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
- `lora_dropout`: 0.0

## 关键特性

1. **用户历史处理**: 如果用户点击历史超过3条,只保留最近3条新闻
2. **样本构建策略**: 曝光新闻列表中,前面的新闻用于构建喜欢/不喜欢列表,最后一个新闻作为预测目标
3. **输出格式**: 模型输出严格限定为 "Yes." 或 "No." 两种格式
4. **数据集划分**: 使用 MIND 原生的 train/val/test 三部分数据
5. **模型对比**: 支持微调前后模型的性能对比评估
6. **vLLM 加速**: 集成 vLLM 实现 18-25 倍推理加速,显著提升评估效率
7. **双推理模式**: 支持 vLLM 批量推理和 API 服务推理两种方式

## 文件清单

核心文件:
- [download_qwen_model.py](download_qwen_model.py) - 从 ModelScope 下载 Qwen3-8B 模型
- [数据预处理.ipynb](数据预处理.ipynb) - 处理 train/val/test 数据集
- [dataset_info.json](dataset_info.json) - LLaMA-Factory 数据集注册配置
- [new_train.yaml](new_train.yaml) - 训练配置文件
- [news_inference.yaml](news_inference.yaml) - 推理配置文件
- [merge_lora_weights.py](merge_lora_weights.py) - 合并 LoRA 权重到基座模型 (vLLM 必需)
- [模型对比评估_vllm.ipynb](模型对比评估_vllm.ipynb) - vLLM 加速评估 (推荐, 18-25倍加速)
- [模型对比评估.ipynb](模型对比评估.ipynb) - API 方式对比评估 (备选)
- [模型预测.ipynb](模型预测.ipynb) - 原有的推理脚本

## 注意事项

### 通用注意事项
1. 所有 Python 命令必须使用完整的 Anaconda Python 路径
2. 路径包含空格时需要用双引号包裹
3. Qwen3-8B 模型约 28GB,下载前确保磁盘空间充足
4. 训练前确保 `max_samples` 参数符合需求(默认 1000,可根据资源调整)
5. 数据预处理时处理 val 数据集可能需要较长时间

### 显存优化建议
如遇到显存不足,可以:
- 减少 `lora_target` 只保留 `q_proj,v_proj`
- 使用 4-bit 量化: 在 yaml 中添加 `quantization_bit: 4`
- 增加 `gradient_accumulation_steps`
- 降低 vLLM 的 `gpu_memory_utilization` 到 0.8

### vLLM 特定注意事项
1. **LoRA 权重合并必需**: vLLM 推理前必须先运行 `merge_lora_weights.py` 合并权重
2. **合并显存需求**: 合并过程需要约 40GB 显存,确保资源充足
3. **合并是一次性操作**: 合并后无需重复,可直接使用合并后的模型
4. **vLLM 批量推理**: vLLM 自动批处理,推荐用于评估全量数据
5. **Prompt 格式**: vLLM 需要使用 Qwen chat 格式: `<|im_start|>system\n...<|im_end|>`

### 推理方式选择
- **vLLM 方式 (推荐)**:
  - 优势: 18-25 倍加速, 显存利用率高, 适合批量评估
  - 劣势: 需要合并权重(一次性操作), 初始化较慢
  - 适用场景: 全量评估, 批量预测, 性能要求高

- **API 方式 (备选)**:
  - 优势: 无需合并权重, 启动快速, 适合小规模测试
  - 劣势: 推理速度慢 20+ 倍, 资源利用率低
  - 适用场景: 快速验证, 小样本测试, 调试

### vLLM 常见问题
1. **CUDA out of memory**: 降低 `gpu_memory_utilization` 或 `max_model_len`
2. **加载失败**: 确保合并脚本成功运行,检查 `./merged_models/` 目录
3. **输出格式错误**: 检查 prompt 格式,调整 `stop` 参数和 `temperature`
4. **性能未达预期**: 增加 `max_num_batched_tokens` 和 `max_num_seqs` 参数

## 相关文档

- 实施计划: [C:\Users\wxb55\.claude\plans\humming-juggling-pearl.md](C:\Users\wxb55\.claude\plans\humming-juggling-pearl.md)
- 项目手册: [项目手册：SFT推荐模型微调.pdf](项目手册：SFT推荐模型微调.pdf)
- 研究论文: [A Survey on Large Language Models for Recommendation.pdf](A Survey on Large Language Models for Recommendation.pdf)
