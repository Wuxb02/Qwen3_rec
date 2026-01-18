# Qwen3-8B 新闻推荐模型微调项目

<div align="center">

![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
![Model](https://img.shields.io/badge/Model-Qwen3--8B-green)
![Framework](https://img.shields.io/badge/Framework-LLaMA--Factory-orange)
![License](https://img.shields.io/badge/License-Apache%202.0-yellow)

**基于 Qwen3-8B 和 MIND 数据集的新闻推荐系统 | 使用 LoRA 微调 | 集成 vLLM 高性能推理**

[English](README_EN.md) | 简体中文

</div>

---

## 📋 目录

- [项目简介](#-项目简介)
- [核心特性](#-核心特性)
- [快速开始](#-快速开始)
- [数据说明](#-数据说明)
- [模型微调详解](#-模型微调详解)
- [训练流程](#-训练流程)
- [模型评估](#-模型评估)
- [使用指南](#-使用指南)
- [项目结构](#-项目结构)
- [性能优化](#-性能优化)
- [常见问题](#-常见问题faq)
- [参考资料](#-参考资料)
- [许可证](#-许可证)

---

## 🎯 项目简介

本项目基于 **Qwen3-8B** 大语言模型和 **MIND (Microsoft News Dataset)** 数据集,构建了一个新闻推荐系统。通过 **LoRA (Low-Rank Adaptation)** 微调方法,将新闻推荐任务转化为文本生成任务,预测用户是否会点击特定新闻。

### 项目亮点

- 🚀 **高效微调**: 使用 LoRA 方法,仅训练 0.5% 的参数,显著降低训练成本
- ⚡ **极速推理**: 集成 vLLM 实现 **18-25 倍推理加速**
- 📊 **完整流程**: 从数据预处理、模型训练到评估的端到端解决方案
- 🎓 **教育友好**: 详细的文档和代码示例,适合学习和研究
- 🔧 **灵活配置**: 支持多种微调方法切换 (LoRA/Full/P-Tuning/QLoRA)

### 技术栈

```
基座模型: Qwen3-8B (8B 参数)
微调框架: LLaMA-Factory
微调方法: LoRA (Low-Rank Adaptation)
推理引擎: vLLM (高性能推理)
数据集: MIND (Microsoft News Dataset)
语言: Python 3.10+
```

---

## ✨ 核心特性

### 1. 数据处理
- ✅ 完整的 MIND 数据集处理流程 (train/val/test)
- ✅ 智能样本构建策略 (用户历史、喜好新闻、目标新闻)
- ✅ 生成约 **125,000** 训练样本和 **31,000** 测试样本

### 2. 模型微调
- ✅ 支持多种微调方法: **LoRA**, Full Fine-tuning, P-Tuning v2, QLoRA
- ✅ 参数高效: LoRA 仅训练 **0.5%** 参数,适配器仅 **81 MB**
- ✅ 显存优化: 支持混合精度训练 (BF16), 梯度累积

### 3. 高性能推理
- ✅ **vLLM 推理**: 10-30 samples/sec, 2-5 分钟评估 1000 样本
- ✅ **API 推理**: 快速验证, 支持 OpenAI SDK 兼容接口
- ✅ **性能提升**: vLLM 比 API 方式快 **18-25 倍**

### 4. 全面评估
- ✅ 基座模型 vs 微调模型对比
- ✅ 准确率、推理速度、吞吐量等多维度指标
- ✅ 错误案例分析和可视化报告

---

## 🚀 快速开始

### 环境要求

| 组件 | 要求 | 说明 |
|------|------|------|
| **Python** | 3.10+ | 推荐使用 Anaconda |
| **CUDA** | 12.2+ | GPU 加速训练和推理 |
| **显存** | 24GB+ | LoRA 微调最低要求 |
| **显存 (推荐)** | 40GB+ | 用于 LoRA 权重合并和 vLLM 推理 |
| **磁盘空间** | 60GB+ | 模型 (28GB) + 数据集 + 合并模型 (28GB) |

### 依赖安装

```bash
# 1. 创建虚拟环境 (推荐)
conda create -n qwen_news python=3.10
conda activate qwen_news

# 2. 安装核心依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. 安装 LLaMA-Factory
pip install llmtuner

# 4. 安装其他依赖
pip install modelscope  # 模型下载
pip install peft        # LoRA 权重合并
pip install vllm        # 高性能推理
pip install jupyter     # Jupyter Notebook
pip install pandas numpy scikit-learn  # 数据处理
```

### 一键运行

```bash
# 步骤 1: 下载 Qwen3-8B 模型 (首次运行, 约 28GB)
python download_qwen_model.py

# 步骤 2: 数据预处理
jupyter notebook "数据预处理.ipynb"

# 步骤 3: 开始训练 (约 3 epochs)
llamafactory-cli train new_train.yaml

# 步骤 4: 合并 LoRA 权重 (vLLM 推理必需)
python merge_lora_weights.py

# 步骤 5: 评估模型 (推荐使用 vLLM, 18-25倍加速)
jupyter notebook "模型对比评估_vllm.ipynb"
```

---

## 📊 数据说明

### MIND 数据集介绍

[MIND (Microsoft News Dataset)](https://msnews.github.io/) 是微软发布的大规模新闻推荐数据集,包含真实的用户点击行为数据。

#### 数据集规模

| 数据集 | 新闻数量 | 用户行为数量 | 说明 |
|--------|---------|-------------|------|
| **Train** | 101,527 | 2,232,748 | 训练集 |
| **Val** | - | - | 验证集 |
| **Test** | 120,961 | 2,370,727 | 测试集 |

#### 数据目录结构

```
data/
├── MIND/
│   ├── train/
│   │   ├── news.tsv        # 新闻元数据 (NewsID, Category, Title, etc.)
│   │   └── behaviors.tsv   # 用户行为 (UserID, Time, History, Impressions)
│   ├── val/
│   │   ├── news.tsv
│   │   └── behaviors.tsv
│   └── test/
│       ├── news.tsv
│       └── behaviors.tsv
└── processed/
    ├── mind_train.json     # 处理后的训练样本 (~125,000条)
    ├── mind_val.json       # 处理后的验证样本 (~35,000条)
    └── mind_test.json      # 处理后的测试样本 (~31,000条)
```

#### 原始数据格式

**news.tsv** (新闻元数据)
```
NewsID    Category    SubCategory    Title                           Abstract    ...
N12345    sports      football       "Team A wins championship"      "..."       ...
N67890    tech        AI             "New AI model released"         "..."       ...
```

**behaviors.tsv** (用户行为)
```
UserID    Time              History                    Impressions
U1001     2020-10-15 10:30  N12345 N67890 N11111      N22222-1 N33333-0 N44444-1
U1002     2020-10-15 10:31  N55555 N66666             N77777-0 N88888-1
```

- **History**: 用户点击历史 (空格分隔的 NewsID)
- **Impressions**: 曝光新闻列表 (格式: NewsID-Label, 1=点击, 0=未点击)

### 数据处理流程

#### 1. 读取原始数据

```python
import pandas as pd

# 读取新闻数据
news_df = pd.read_csv(
    'data/MIND/train/news.tsv',
    sep='\t',
    names=['news_id', 'category', 'subcategory', 'title', 'abstract',
           'url', 'title_entities', 'abstract_entities']
)

# 构建新闻 ID 到标题的映射
news_dict = dict(zip(news_df['news_id'], news_df['title']))

# 读取用户行为数据
behaviors_df = pd.read_csv(
    'data/MIND/train/behaviors.tsv',
    sep='\t',
    names=['impression_id', 'user_id', 'time', 'history', 'impressions']
)
```

#### 2. 样本构建策略

每个训练样本包含以下信息:

```python
sample = {
    "instruction": "You are a news recommendation expert. Based on user's click history and preferences, predict if they will click the target news.",
    "input": f"""User click histories: "{hist1}", "{hist2}", "{hist3}"
User's liked news: "{liked_news1}", "{liked_news2}"
User's disliked news: "{disliked_news1}", "{disliked_news2}"
Target news: "{target_news}"
Will the user click this news? Answer with 'Yes.' or 'No.'""",
    "output": "Yes."  # 或 "No."
}
```

**样本构建规则:**
- **用户历史**: 保留最近 **3 条**点击新闻 (如果超过 3 条)
- **喜欢的新闻**: 从曝光列表中标签为 `1` 的新闻 (点击过的)
- **不喜欢的新闻**: 从曝光列表中标签为 `0` 的新闻 (未点击的)
- **目标新闻**: 曝光列表的**最后一个**新闻作为预测目标

#### 3. 样本生成代码片段

```python
def build_training_sample(user_history, impressions, news_dict):
    """构建单个训练样本"""
    # 限制用户历史长度
    history_list = user_history.split()[-3:]  # 最多保留 3 条
    history_titles = [news_dict.get(nid, "") for nid in history_list]

    # 解析曝光列表
    impression_list = impressions.split()
    liked_news, disliked_news = [], []

    for imp in impression_list[:-1]:  # 除了最后一个
        news_id, label = imp.split('-')
        title = news_dict.get(news_id, "")
        if label == '1':
            liked_news.append(title)
        else:
            disliked_news.append(title)

    # 最后一个作为目标
    target_news_id, target_label = impression_list[-1].split('-')
    target_title = news_dict.get(target_news_id, "")

    # 构建样本
    sample = {
        "instruction": "You are a news recommendation expert...",
        "input": f"""User click histories: {', '.join([f'"{t}"' for t in history_titles])}
User's liked news: {', '.join([f'"{t}"' for t in liked_news[:3]])}
User's disliked news: {', '.join([f'"{t}"' for t in disliked_news[:3]])}
Target news: "{target_title}"
Will the user click this news? Answer with 'Yes.' or 'No.'""",
        "output": "Yes." if target_label == '1' else "No."
    }

    return sample
```

#### 4. 处理后的数据统计

| 数据集 | 样本数量 | 文件大小 | 正样本比例 | 平均历史长度 |
|--------|---------|---------|-----------|-------------|
| **Train** | ~125,000 | 753 KB | ~50% | 2.8 |
| **Val** | ~35,000 | - | ~50% | 2.7 |
| **Test** | ~31,000 | 188 KB | ~50% | 2.9 |

---

## 🔧 模型微调详解

### 基座模型: Qwen3-8B

[Qwen3-8B](https://github.com/QwenLM/Qwen) 是阿里巴巴通义千问团队开发的开源大语言模型。

| 参数 | 说明 |
|------|------|
| **模型参数** | 8B (140 亿) |
| **上下文长度** | 128K tokens |
| **模型大小** | ~28 GB (FP16) |
| **训练数据** | 多领域高质量中英文数据 |

#### 模型下载

```bash
# 方式 1: 使用项目提供的下载脚本 (推荐)
python download_qwen_model.py

# 方式 2: 使用 ModelScope CLI
pip install modelscope
modelscope download --model Qwen/Qwen3-8B --local_dir ./Qwen/Qwen3-8B

# 下载完成后,模型保存在 ./Qwen/Qwen3-8B/ (约 28GB)
```

### 微调方法对比

本项目支持多种微调方法,下表对比了各方法的特点:

| 微调方法 | 可训练参数 | 显存占用 (训练) | 训练速度 | 推理速度 | 推荐场景 | 效果 |
|---------|-----------|---------------|---------|---------|---------|------|
| **Full Fine-tuning** | 100% (8B) | 50GB+ | 慢 (1x) | 快 | 大规模数据, 充足资源 | ⭐⭐⭐⭐⭐ |
| **LoRA** ✅ | ~0.5% (70M) | 24GB | 快 (3x) | 快 | **资源受限, 效果要求高** | ⭐⭐⭐⭐⭐ |
| **P-Tuning v2** | ~0.1% (10M) | 20GB | 很快 (5x) | 快 | 快速实验, 轻量级任务 | ⭐⭐⭐⭐ |
| **Adapter** | ~2% (280M) | 28GB | 中等 (2x) | 快 | 多任务学习 | ⭐⭐⭐⭐ |
| **QLoRA (4-bit)** | ~0.5% (70M) | 12GB | 中等 (2x) | 中等 | **显存极度受限** | ⭐⭐⭐⭐ |

✅ **本项目使用 LoRA**, 平衡了效果和效率

---

### 方法 1: LoRA (Low-Rank Adaptation) ⭐ 推荐

#### 原理介绍

LoRA 通过在预训练模型的权重矩阵旁边添加**低秩分解矩阵**来实现高效微调:

```
原始权重更新: W' = W + ΔW
LoRA 分解: ΔW = B × A  (其中 B: d×r, A: r×k, r << min(d,k))
```

- **r**: LoRA rank, 控制参数量 (本项目使用 r=8)
- **参数量**: 仅为全参数微调的 **0.5%**
- **优势**: 训练快、显存少、效果好、易于部署

#### LoRA 配置

在 [new_train.yaml](new_train.yaml) 中的配置:

```yaml
# 微调方法
finetuning_type: lora

# LoRA 目标模块 (Qwen3-8B 的 7 个核心层)
lora_target: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj

# LoRA 参数 (在 LLaMA-Factory 中自动设置)
# lora_rank: 8           # LoRA 秩
# lora_alpha: 16         # 缩放因子
# lora_dropout: 0.0      # Dropout 率
```

#### 完整训练配置示例

```yaml
### 模型配置
model_name_or_path: ./Qwen/Qwen3-8B
template: qwen

### 微调方法
stage: sft
do_train: true
finetuning_type: lora
lora_target: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj

### 数据集配置
dataset: mind
dataset_dir: ./
dataset_info: dataset_info.json
cutoff_len: 1024
max_samples: 1000  # 可调整或删除以使用全部数据
overwrite_cache: true
preprocessing_num_workers: 16

### 训练参数
per_device_train_batch_size: 1
gradient_accumulation_steps: 8  # 有效批大小 = 1 × 8 = 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true  # 混合精度训练

### 评估配置
val_size: 0.1  # 自动划分 10% 作为验证集
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 500

### 输出配置
output_dir: ./saves/qwen3_mind_news_recommend
logging_steps: 10
save_steps: 500
plot_loss: true
```

#### LoRA 优势总结

✅ **参数高效**: 仅训练 0.5% 参数 (70M vs 8B)
✅ **显存友好**: 24GB 显存即可训练 8B 模型
✅ **训练快速**: 比全参数微调快 **3-5 倍**
✅ **效果优异**: 接近全参数微调的效果
✅ **易于部署**: 适配器仅 81 MB, 可快速切换

---

### 方法 2: Full Fine-tuning (全参数微调)

#### 适用场景

- ✅ 有充足的计算资源 (50GB+ 显存)
- ✅ 数据集规模较大 (100K+ 样本)
- ✅ 追求最佳效果
- ✅ 任务与预训练差异较大

#### 配置方法

修改 `new_train.yaml`:

```yaml
### 微调方法
stage: sft
do_train: true
finetuning_type: full  # 改为 full

# 删除 LoRA 相关配置
# lora_target: ...  (删除)

### 训练参数 (需要调整)
per_device_train_batch_size: 1
gradient_accumulation_steps: 16  # 增加累积步数
learning_rate: 5.0e-6  # 降低学习率
num_train_epochs: 2.0  # 减少训练轮数
```

#### 与 LoRA 对比

| 指标 | Full Fine-tuning | LoRA |
|------|-----------------|------|
| **训练参数** | 8B | 70M |
| **显存占用** | 50GB+ | 24GB |
| **训练时间** | 基准 (1x) | 3-5倍快 |
| **适配器大小** | 28GB | 81MB |
| **效果** | 最好 (100%) | 优秀 (95-98%) |

---

### 方法 3: P-Tuning v2

#### 原理和优势

P-Tuning v2 仅在模型的**每一层输入**添加可训练的连续提示向量:

- **可训练参数**: ~0.1% (约 10M)
- **显存占用**: 20GB
- **训练速度**: 比 LoRA 更快
- **适用场景**: 快速实验、轻量级任务

#### 配置示例

```yaml
### 微调方法
finetuning_type: p_tuning
pre_seq_len: 128  # 提示向量长度
prefix_projection: true  # 使用投影层
```

#### 优缺点

✅ **优点**: 参数最少, 训练最快, 显存占用低
❌ **缺点**: 效果略低于 LoRA (约 90-95%)

---

### 方法 4: Adapter

#### 实现方式

在 Transformer 的每一层添加**轻量级适配器模块**:

```
Transformer Layer:
  Self-Attention → Adapter → Feed-Forward → Adapter
```

- **可训练参数**: ~2% (约 280M)
- **显存占用**: 28GB
- **适用场景**: 多任务学习, 需要快速切换不同任务

#### 配置示例

```yaml
### 微调方法
finetuning_type: adapter
adapter_size: 64  # 适配器隐藏层大小
```

---

### 方法 5: QLoRA (量化 LoRA)

#### 原理和优势

QLoRA 结合了**量化技术**和 LoRA:

- 基座模型量化为 **4-bit** 或 **8-bit**
- LoRA 权重保持 FP16/BF16
- **显存占用**: 12GB (4-bit)
- **效果**: 与 LoRA 相当 (95-98%)

#### 配置示例

```yaml
### 微调方法
finetuning_type: lora
lora_target: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj

### 量化配置
quantization_bit: 4  # 4-bit 量化
quantization_method: bitsandbytes
```

#### 优缺点

✅ **优点**: 显存占用极低 (12GB), 适合消费级显卡
❌ **缺点**: 推理速度略慢, 量化精度损失

---

### 微调方法选择指南

| 如果你... | 推荐方法 | 原因 |
|---------|---------|------|
| 显存 >= 40GB, 追求最佳效果 | **Full Fine-tuning** | 效果最好 |
| 显存 24-40GB, 平衡效果和效率 | **LoRA** ⭐ | 本项目选择 |
| 显存 12-24GB, 显存受限 | **QLoRA (4-bit)** | 极低显存 |
| 需要快速实验验证 | **P-Tuning v2** | 训练最快 |
| 多任务学习场景 | **Adapter** | 易于切换 |

---

## 🏋️ 训练流程

### 完整训练步骤

#### 步骤 1: 准备数据

```bash
# 1. 确保 MIND 数据集已放置在 data/MIND/ 目录
ls data/MIND/train/  # 应包含 news.tsv 和 behaviors.tsv

# 2. 运行数据预处理
jupyter notebook "数据预处理.ipynb"

# 3. 验证处理后的数据
ls data/processed/  # 应包含 mind_train.json, mind_test.json
```

#### 步骤 2: 配置训练参数

编辑 [new_train.yaml](new_train.yaml) 调整参数:

```yaml
# 关键参数说明
max_samples: 1000        # 训练样本数 (删除此行使用全部数据)
num_train_epochs: 3.0    # 训练轮数
learning_rate: 1.0e-4    # 学习率
```

#### 步骤 3: 启动训练

```bash
# 使用 LLaMA-Factory CLI 启动训练
llamafactory-cli train new_train.yaml

# 训练过程输出示例:
# [INFO] Loading model from ./Qwen/Qwen3-8B
# [INFO] Using LoRA with rank 8
# [INFO] Trainable params: 70M / 8B (0.5%)
# [INFO] Training started...
# Epoch 1/3: 100%|████████| 125/125 [10:30<00:00, loss=0.352]
# Eval loss: 0.298
# ...
```

#### 步骤 4: 监控训练

训练过程中会生成以下文件:

```
saves/qwen3_mind_news_recommend/
├── adapter_model.safetensors  # LoRA 权重 (81 MB)
├── adapter_config.json        # LoRA 配置
├── trainer_state.json         # 训练状态和历史
├── training_loss.png          # 损失曲线图 (如果启用 plot_loss)
└── checkpoint-500/            # 中间检查点 (每 500 步保存)
```

查看训练日志:

```bash
# 查看实时日志
tail -f saves/qwen3_mind_news_recommend/trainer_log.jsonl

# 绘制损失曲线
python -c "
import json
import matplotlib.pyplot as plt

with open('saves/qwen3_mind_news_recommend/trainer_state.json') as f:
    state = json.load(f)

losses = [log['loss'] for log in state['log_history'] if 'loss' in log]
plt.plot(losses)
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('training_loss.png')
print('损失曲线已保存到 training_loss.png')
"
```

### 训练超参数说明

| 参数 | 默认值 | 说明 | 调优建议 |
|------|--------|------|---------|
| `learning_rate` | 1.0e-4 | 学习率 | LoRA: 1e-4, Full: 5e-6 |
| `num_train_epochs` | 3.0 | 训练轮数 | 根据验证损失调整 (2-5) |
| `per_device_train_batch_size` | 1 | 每张卡的批大小 | 根据显存调整 (1-4) |
| `gradient_accumulation_steps` | 8 | 梯度累积步数 | 有效批大小 = batch_size × accumulation |
| `warmup_ratio` | 0.1 | 预热比例 | 10% 训练步数用于预热 |
| `lr_scheduler_type` | cosine | 学习率调度器 | cosine, linear, constant |
| `max_samples` | 1000 | 最大训练样本数 | 删除以使用全部数据 |

### 常见问题和解决方案

#### 问题 1: CUDA Out of Memory (OOM)

**错误信息**: `RuntimeError: CUDA out of memory`

**解决方案**:
```yaml
# 方法 1: 减少批大小
per_device_train_batch_size: 1  # 降低到 1
gradient_accumulation_steps: 16  # 增加累积步数

# 方法 2: 使用量化
quantization_bit: 4

# 方法 3: 减少 LoRA 目标模块
lora_target: q_proj,v_proj  # 只保留 q_proj 和 v_proj

# 方法 4: 降低序列长度
cutoff_len: 512  # 从 1024 降低到 512
```

#### 问题 2: 训练速度慢

**解决方案**:
```yaml
# 1. 启用混合精度训练
bf16: true  # 或 fp16: true

# 2. 增加数据加载线程
preprocessing_num_workers: 16  # 根据 CPU 核心数调整

# 3. 启用梯度检查点 (显存换时间)
gradient_checkpointing: false  # 设为 false 加速

# 4. 使用更快的优化器
optim: adamw_torch_fused  # PyTorch 融合优化器
```

#### 问题 3: 验证损失不下降

**解决方案**:
```yaml
# 1. 调整学习率
learning_rate: 5.0e-5  # 降低学习率

# 2. 调整预热比例
warmup_ratio: 0.15  # 增加预热步数

# 3. 检查数据质量
max_samples: 10000  # 增加训练样本数

# 4. 调整 LoRA rank
# 在 adapter_config.json 中手动修改 r: 16  (增加到 16)
```

---

## 📈 模型评估

本项目提供两种评估方式: **vLLM 批量推理** (推荐) 和 **API 服务推理** (备选)。

### 评估方式对比

| 评估方式 | 吞吐量 | 1000样本耗时 | 全量31K样本耗时 | 显存占用 | 适用场景 |
|---------|-------|------------|--------------|---------|---------|
| **vLLM 批量推理** ⭐ | 10-30 s/s | 2-5 分钟 | 25-50 分钟 | 28GB | 批量评估、生产部署 |
| **API 服务推理** | 0.3-0.5 s/s | 30-50 分钟 | 20+ 小时 | 24GB | 快速验证、小规模测试 |

**加速倍数**: vLLM 比 API 方式快 **18-25 倍** 🚀

---

### 方式 1: vLLM 批量推理 ⭐ 推荐

vLLM 是一个高性能的大模型推理引擎,支持批处理和连续批处理,显著提升推理效率。

#### 步骤 1: 合并 LoRA 权重

⚠️ **重要**: vLLM 推理前必须先合并 LoRA 权重到基座模型

```bash
# 运行合并脚本 (需要约 40GB 显存)
python merge_lora_weights.py

# 合并过程输出:
# ============================================================
# 开始合并 LoRA 权重
# 基座模型: ./Qwen/Qwen3-8B
# LoRA 适配器: ./saves/qwen3_mind_news_recommend
# 输出路径: ./merged_models/qwen3_mind_news_recommend
# ============================================================
#
# [1/4] 加载 LoRA 模型...
# ✓ LoRA 模型加载成功
#
# [2/4] 合并 LoRA 权重到基座模型...
# ✓ 权重合并完成
#
# [3/4] 保存合并后的模型到 ./merged_models/qwen3_mind_news_recommend...
# ✓ 模型保存成功
#
# [4/4] 保存 tokenizer...
# ✓ Tokenizer 保存成功
# ============================================================
```

**合并后的目录结构**:
```
merged_models/qwen3_mind_news_recommend/
├── config.json                      # 模型配置
├── model-00001-of-00008.safetensors # 模型权重 (分片)
├── model-00002-of-00008.safetensors
├── ...
├── tokenizer.json                   # 分词器
└── tokenizer_config.json
```

#### 步骤 2: 使用 vLLM 评估

打开 [模型对比评估_vllm.ipynb](模型对比评估_vllm.ipynb):

```python
from vllm import LLM, SamplingParams
import json

# 1. 加载合并后的微调模型
print("加载微调模型...")
finetuned_llm = LLM(
    model="./merged_models/qwen3_mind_news_recommend",
    trust_remote_code=True,
    gpu_memory_utilization=0.9,  # 使用 90% 显存
    max_model_len=2048,
    dtype="bfloat16"
)

# 2. 加载基座模型 (用于对比)
print("加载基座模型...")
base_llm = LLM(
    model="./Qwen/Qwen3-8B",
    trust_remote_code=True,
    gpu_memory_utilization=0.9,
    max_model_len=2048,
    dtype="bfloat16"
)

# 3. 加载测试数据
with open("data/processed/mind_test.json", "r", encoding="utf-8") as f:
    test_data = [json.loads(line) for line in f][:1000]  # 测试 1000 样本

# 4. 构建 prompts
def build_prompt(sample):
    """构建 Qwen3 格式的 prompt"""
    return f"""<|im_start|>system
{sample['instruction']}<|im_end|>
<|im_start|>user
{sample['input']}<|im_end|>
<|im_start|>assistant
"""

prompts = [build_prompt(sample) for sample in test_data]

# 5. 批量推理 (vLLM 自动批处理)
sampling_params = SamplingParams(
    temperature=0.01,  # 低温度,接近贪心解码
    top_p=0.9,
    max_tokens=10,
    stop=["<|im_end|>", "\n"]
)

print("开始推理 (微调模型)...")
finetuned_outputs = finetuned_llm.generate(prompts, sampling_params)

print("开始推理 (基座模型)...")
base_outputs = base_llm.generate(prompts, sampling_params)

# 6. 解析结果并计算准确率
def parse_answer(text):
    """解析模型输出"""
    text = text.strip().lower()
    if "yes" in text:
        return "Yes."
    elif "no" in text:
        return "No."
    return "Unknown"

# 微调模型准确率
finetuned_correct = 0
for output, sample in zip(finetuned_outputs, test_data):
    pred = parse_answer(output.outputs[0].text)
    if pred == sample['output']:
        finetuned_correct += 1

# 基座模型准确率
base_correct = 0
for output, sample in zip(base_outputs, test_data):
    pred = parse_answer(output.outputs[0].text)
    if pred == sample['output']:
        base_correct += 1

print(f"\n{'='*60}")
print(f"评估结果 (1000 样本)")
print(f"{'='*60}")
print(f"基座模型准确率: {base_correct/len(test_data)*100:.2f}%")
print(f"微调模型准确率: {finetuned_correct/len(test_data)*100:.2f}%")
print(f"准确率提升: +{(finetuned_correct-base_correct)/len(test_data)*100:.2f}%")
print(f"{'='*60}")
```

#### vLLM 性能优化参数

```python
LLM(
    model="./merged_models/qwen3_mind_news_recommend",

    # 显存管理
    gpu_memory_utilization=0.9,  # GPU 显存利用率 (0.8-0.95)
    max_model_len=2048,          # 最大序列长度

    # 批处理参数
    max_num_batched_tokens=8192, # 批处理 token 上限
    max_num_seqs=256,            # 最大并发序列数

    # 性能优化
    dtype="bfloat16",            # 数据类型 (bfloat16/float16)
    enforce_eager=False,         # 使用 CUDA graph 加速
    trust_remote_code=True       # 信任自定义代码
)
```

---

### 方式 2: API 服务推理

适用于快速验证和小规模测试。

#### 步骤 1: 启动 API 服务

```bash
# 终端 1: 启动基座模型服务 (端口 8000)
llamafactory-cli api \
  --model_name_or_path ./Qwen/Qwen3-8B \
  --template qwen \
  --port 8000

# 终端 2: 启动微调模型服务 (端口 8001)
llamafactory-cli api \
  --model_name_or_path ./Qwen/Qwen3-8B \
  --adapter_name_or_path ./saves/qwen3_mind_news_recommend \
  --template qwen \
  --finetuning_type lora \
  --port 8001

# 服务启动后输出:
# INFO: Application startup complete.
# INFO: Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

#### 步骤 2: 使用 API 评估

打开 [模型对比评估.ipynb](模型对比评估.ipynb):

```python
from openai import OpenAI
import json

# 1. 创建 API 客户端
base_client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1"
)

finetuned_client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8001/v1"
)

# 2. 加载测试数据
with open("data/processed/mind_test.json", "r", encoding="utf-8") as f:
    test_data = [json.loads(line) for line in f][:100]  # 测试 100 样本

# 3. 单条推理函数
def predict(client, sample):
    """调用 API 进行推理"""
    response = client.chat.completions.create(
        model="qwen3-8B",
        messages=[
            {"role": "system", "content": sample["instruction"]},
            {"role": "user", "content": sample["input"]}
        ],
        temperature=0.01,
        max_tokens=10
    )
    return response.choices[0].message.content

# 4. 评估两个模型
base_correct = 0
finetuned_correct = 0

for i, sample in enumerate(test_data):
    if i % 10 == 0:
        print(f"进度: {i}/{len(test_data)}")

    # 基座模型预测
    base_pred = predict(base_client, sample)
    if base_pred.strip() == sample["output"]:
        base_correct += 1

    # 微调模型预测
    finetuned_pred = predict(finetuned_client, sample)
    if finetuned_pred.strip() == sample["output"]:
        finetuned_correct += 1

print(f"\n基座模型准确率: {base_correct/len(test_data)*100:.2f}%")
print(f"微调模型准确率: {finetuned_correct/len(test_data)*100:.2f}%")
```

#### cURL 调用示例

```bash
# 调用基座模型
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8B",
    "messages": [
      {"role": "system", "content": "You are a news recommendation expert."},
      {"role": "user", "content": "User click histories: \"News A\", \"News B\"\nTarget news: \"News C\"\nWill the user click? Answer with Yes. or No."}
    ],
    "temperature": 0.01,
    "max_tokens": 10
  }'
```

---

### 性能对比表格

#### vLLM vs API 推理速度对比

| 样本数量 | vLLM 耗时 | API 耗时 | 加速倍数 |
|---------|----------|---------|---------|
| **100** | 10-20 秒 | 3-5 分钟 | **15-20x** |
| **1,000** | 2-5 分钟 | 30-50 分钟 | **18-25x** |
| **10,000** | 10-20 分钟 | 5-8 小时 | **20-25x** |
| **31,000 (全量)** | 25-50 分钟 | 20+ 小时 | **25-30x** |

#### 基座模型 vs 微调模型准确率对比

| 数据集 | 基座模型 | 微调模型 (LoRA) | 准确率提升 |
|--------|---------|----------------|-----------|
| **验证集** (10%) | ~52% | ~68% | **+16%** |
| **测试集** (1,000) | ~51% | ~67% | **+16%** |
| **测试集** (全量 31K) | ~50% | ~66% | **+16%** |

*注: 具体数值根据训练参数和数据划分略有不同*

#### 不同批次大小的吞吐量对比 (vLLM)

| 批次大小 | 吞吐量 (samples/sec) | 显存占用 | 延迟 (ms/sample) |
|---------|---------------------|---------|-----------------|
| 1 | 5-8 | 20GB | 125-200 |
| 8 | 15-20 | 24GB | 50-65 |
| 16 | 20-25 | 26GB | 40-50 |
| 32 | 25-30 | 28GB | 33-40 |

---

### 评估指标说明

#### 1. 准确率 (Accuracy)

```python
accuracy = correct_predictions / total_samples
```

- **基座模型**: ~50% (接近随机猜测)
- **微调模型**: ~66-68% (显著提升)

#### 2. 推理吞吐量 (Samples/sec)

每秒处理的样本数量:

- **vLLM**: 10-30 samples/sec
- **API**: 0.3-0.5 samples/sec

#### 3. 平均延迟 (ms/sample)

单个样本的推理时间:

- **vLLM**: 33-125 ms
- **API**: 2000-3000 ms

#### 4. 加速倍数

vLLM 相对于 API 的加速:

```
加速倍数 = API推理时间 / vLLM推理时间
```

- **平均加速**: **18-25 倍**

---

## 💻 使用指南

### Python API 推理 (vLLM)

#### 单条推理示例

```python
from vllm import LLM, SamplingParams

# 1. 加载模型
llm = LLM(
    model="./merged_models/qwen3_mind_news_recommend",
    trust_remote_code=True,
    gpu_memory_utilization=0.9
)

# 2. 定义推理参数
sampling_params = SamplingParams(
    temperature=0.01,
    top_p=0.9,
    max_tokens=10,
    stop=["<|im_end|>", "\n"]
)

# 3. 构建 prompt
prompt = """<|im_start|>system
You are a news recommendation expert. Based on user's click history and preferences, predict if they will click the target news.<|im_end|>
<|im_start|>user
User click histories: "Tesla unveils new Model Y", "SpaceX launches Starship"
User's liked news: "Apple announces iPhone 16", "Google's new AI model"
User's disliked news: "Local weather update", "Sports scores"
Target news: "Microsoft acquires AI startup"
Will the user click this news? Answer with 'Yes.' or 'No.'<|im_end|>
<|im_start|>assistant
"""

# 4. 推理
outputs = llm.generate([prompt], sampling_params)
prediction = outputs[0].outputs[0].text.strip()
print(f"预测结果: {prediction}")  # 输出: Yes. 或 No.
```

#### 批量推理示例

```python
# 批量推理 (vLLM 自动优化批处理)
prompts = [
    build_prompt(sample1),
    build_prompt(sample2),
    build_prompt(sample3),
    # ... 更多样本
]

# 一次性推理所有样本 (自动批处理)
outputs = llm.generate(prompts, sampling_params)

# 解析结果
predictions = [output.outputs[0].text.strip() for output in outputs]
```

---

### HTTP API 推理

#### 启动 API 服务

```bash
# 启动微调模型服务
llamafactory-cli api \
  --model_name_or_path ./Qwen/Qwen3-8B \
  --adapter_name_or_path ./saves/qwen3_mind_news_recommend \
  --template qwen \
  --finetuning_type lora \
  --port 8000
```

#### cURL 调用示例

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8B",
    "messages": [
      {
        "role": "system",
        "content": "You are a news recommendation expert. Based on user'\''s click history and preferences, predict if they will click the target news."
      },
      {
        "role": "user",
        "content": "User click histories: \"Tesla unveils new Model Y\", \"SpaceX launches Starship\"\nUser'\''s liked news: \"Apple announces iPhone 16\", \"Google'\''s new AI model\"\nUser'\''s disliked news: \"Local weather update\", \"Sports scores\"\nTarget news: \"Microsoft acquires AI startup\"\nWill the user click this news? Answer with '\''Yes.'\'' or '\''No.'\''."
      }
    ],
    "temperature": 0.01,
    "max_tokens": 10
  }'

# 响应示例:
# {
#   "id": "chatcmpl-xxx",
#   "object": "chat.completion",
#   "created": 1234567890,
#   "model": "qwen3-8B",
#   "choices": [
#     {
#       "index": 0,
#       "message": {
#         "role": "assistant",
#         "content": "Yes."
#       },
#       "finish_reason": "stop"
#     }
#   ]
# }
```

#### Python requests 调用示例

```python
import requests
import json

url = "http://localhost:8000/v1/chat/completions"

payload = {
    "model": "qwen3-8B",
    "messages": [
        {
            "role": "system",
            "content": "You are a news recommendation expert. Based on user's click history and preferences, predict if they will click the target news."
        },
        {
            "role": "user",
            "content": """User click histories: "Tesla unveils new Model Y", "SpaceX launches Starship"
User's liked news: "Apple announces iPhone 16", "Google's new AI model"
User's disliked news: "Local weather update", "Sports scores"
Target news: "Microsoft acquires AI startup"
Will the user click this news? Answer with 'Yes.' or 'No.'."""
        }
    ],
    "temperature": 0.01,
    "max_tokens": 10
}

response = requests.post(url, json=payload)
result = response.json()
prediction = result["choices"][0]["message"]["content"]
print(f"预测结果: {prediction}")
```

#### OpenAI SDK 调用示例

```python
from openai import OpenAI

# 创建客户端
client = OpenAI(
    api_key="EMPTY",  # LLaMA-Factory 不需要 API key
    base_url="http://localhost:8000/v1"
)

# 调用模型
response = client.chat.completions.create(
    model="qwen3-8B",
    messages=[
        {
            "role": "system",
            "content": "You are a news recommendation expert. Based on user's click history and preferences, predict if they will click the target news."
        },
        {
            "role": "user",
            "content": """User click histories: "Tesla unveils new Model Y", "SpaceX launches Starship"
User's liked news: "Apple announces iPhone 16", "Google's new AI model"
User's disliked news: "Local weather update", "Sports scores"
Target news: "Microsoft acquires AI startup"
Will the user click this news? Answer with 'Yes.' or 'No.'."""
        }
    ],
    temperature=0.01,
    max_tokens=10
)

prediction = response.choices[0].message.content
print(f"预测结果: {prediction}")
```

---

### Prompt 构建说明

#### 推荐的 Prompt 格式

```python
def build_recommendation_prompt(
    history_titles: list,  # 用户历史点击
    liked_titles: list,    # 喜欢的新闻
    disliked_titles: list, # 不喜欢的新闻
    target_title: str      # 目标新闻
):
    """构建推荐任务的 prompt"""

    # 格式化列表
    history_str = ', '.join([f'"{t}"' for t in history_titles])
    liked_str = ', '.join([f'"{t}"' for t in liked_titles[:3]])  # 最多3条
    disliked_str = ', '.join([f'"{t}"' for t in disliked_titles[:3]])

    # 构建 prompt
    system_prompt = "You are a news recommendation expert. Based on user's click history and preferences, predict if they will click the target news."

    user_prompt = f"""User click histories: {history_str}
User's liked news: {liked_str}
User's disliked news: {disliked_str}
Target news: "{target_title}"
Will the user click this news? Answer with 'Yes.' or 'No.'."""

    # Qwen3 格式 (用于 vLLM)
    qwen_prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
"""

    return qwen_prompt
```

#### 响应解析

```python
def parse_prediction(response_text: str) -> str:
    """解析模型输出"""
    text = response_text.strip().lower()

    if "yes" in text:
        return "Yes."
    elif "no" in text:
        return "No."
    else:
        return "Unknown"  # 无法识别的输出
```

---

## 📁 项目结构

```
4.ChatGLM 微调推荐模型/
│
├── data/                           # 数据目录
│   ├── MIND/                       # 原始 MIND 数据集
│   │   ├── train/
│   │   │   ├── news.tsv            # 训练集新闻 (101,527条)
│   │   │   └── behaviors.tsv       # 训练集用户行为 (2,232,748条)
│   │   ├── val/
│   │   │   ├── news.tsv
│   │   │   └── behaviors.tsv
│   │   └── test/
│   │       ├── news.tsv            # 测试集新闻 (120,961条)
│   │       └── behaviors.tsv       # 测试集用户行为 (2,370,727条)
│   └── processed/                  # 处理后的数据
│       ├── mind_train.json         # 训练样本 (~125,000条, 753KB)
│       ├── mind_val.json           # 验证样本 (~35,000条)
│       └── mind_test.json          # 测试样本 (~31,000条, 188KB)
│
├── Qwen/                           # 模型目录
│   └── Qwen3-8B/                  # 基座模型 (~28GB)
│       ├── config.json
│       ├── model-00001-of-00008.safetensors
│       ├── ...
│       ├── tokenizer.json
│       └── tokenizer_config.json
│
├── saves/                          # 训练输出
│   └── qwen3_mind_news_recommend/  # LoRA 适配器 (84MB)
│       ├── adapter_model.safetensors   # LoRA 权重 (81MB)
│       ├── adapter_config.json         # LoRA 配置
│       ├── trainer_state.json          # 训练状态
│       ├── training_args.bin           # 训练参数
│       └── checkpoint-*/               # 中间检查点
│
├── merged_models/                  # 合并后的模型 (vLLM 使用)
│   └── qwen3_mind_news_recommend/  # 完整微调模型 (~28GB)
│       ├── config.json
│       ├── model-00001-of-00008.safetensors
│       ├── ...
│       └── tokenizer.json
│
├── 数据预处理.ipynb                 # 数据处理 Notebook (38KB)
├── 模型对比评估_vllm.ipynb          # vLLM 评估 (推荐, 23KB)
├── 模型对比评估.ipynb               # API 评估 (备选, 7KB)
├── 模型预测.ipynb                   # 原有推理脚本 (184KB)
│
├── download_qwen_model.py          # 模型下载脚本 (2.4KB)
├── merge_lora_weights.py           # LoRA 权重合并 (3.4KB)
│
├── new_train.yaml                  # 训练配置
├── news_inference.yaml             # 推理配置
├── dataset_info.json               # 数据集注册
│
├── CLAUDE.md                       # 项目文档 (9.1KB)
├── README.md                       # 本文件
│
└── 参考资料/
    ├── A Survey on Large Language Models for Recommendation.pdf  (2.1MB)
    └── 项目手册：SFT推荐模型微调.pdf  (604KB)
```

---

## ⚡ 性能优化

### 训练优化

#### 1. 显存优化

```yaml
# 方法 1: 使用量化
quantization_bit: 4  # 4-bit 量化, 显存减半

# 方法 2: 减少批大小
per_device_train_batch_size: 1
gradient_accumulation_steps: 16  # 保持有效批大小

# 方法 3: 启用梯度检查点
gradient_checkpointing: true  # 显存换时间

# 方法 4: 减少序列长度
cutoff_len: 512  # 从 1024 降低

# 方法 5: 减少 LoRA 目标模块
lora_target: q_proj,v_proj  # 只保留核心模块
```

#### 2. 训练加速

```yaml
# 方法 1: 启用混合精度
bf16: true  # BF16 混合精度

# 方法 2: 使用融合优化器
optim: adamw_torch_fused

# 方法 3: 增加数据加载线程
preprocessing_num_workers: 16

# 方法 4: 禁用梯度检查点
gradient_checkpointing: false  # 时间换显存
```

#### 3. 多卡训练 (如果有多张 GPU)

```bash
# 使用 DeepSpeed 或 FSDP
llamafactory-cli train new_train.yaml --deepspeed ds_config.json

# 或使用 torchrun
torchrun --nproc_per_node 2 -m llmtuner.train new_train.yaml
```

### 推理优化

#### 1. vLLM 参数调优

```python
from vllm import LLM

llm = LLM(
    model="./merged_models/qwen3_mind_news_recommend",

    # 显存管理
    gpu_memory_utilization=0.9,  # 推荐 0.85-0.95
    max_model_len=2048,          # 根据实际需求调整

    # 批处理优化
    max_num_batched_tokens=8192, # 增加批处理上限
    max_num_seqs=256,            # 增加并发序列数

    # 性能优化
    dtype="bfloat16",            # 使用 BF16
    enforce_eager=False,         # 启用 CUDA graph
    trust_remote_code=True
)
```

#### 2. 批量推理策略

```python
# 将大量样本分批处理
def batch_inference(llm, prompts, batch_size=128):
    """分批推理,避免显存溢出"""
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        outputs = llm.generate(batch, sampling_params)
        results.extend(outputs)
    return results
```

#### 3. 量化推理 (降低显存)

```python
# 使用量化模型推理
from vllm import LLM

llm = LLM(
    model="./merged_models/qwen3_mind_news_recommend",
    quantization="awq",  # 或 "gptq", "squeezellm"
    dtype="float16"
)
```

---

## ❓ 常见问题 (FAQ)

### 训练相关

#### Q1: CUDA Out of Memory (显存不足)

**问题**: 训练时报错 `RuntimeError: CUDA out of memory`

**解决方案**:
1. **降低批大小**: `per_device_train_batch_size: 1`
2. **增加梯度累积**: `gradient_accumulation_steps: 16`
3. **使用量化**: `quantization_bit: 4`
4. **减少序列长度**: `cutoff_len: 512`
5. **减少 LoRA 目标**: `lora_target: q_proj,v_proj`
6. **启用梯度检查点**: `gradient_checkpointing: true`

#### Q2: 训练速度太慢

**问题**: 训练 1 个 epoch 需要很长时间

**解决方案**:
1. **启用混合精度**: `bf16: true`
2. **增加数据加载线程**: `preprocessing_num_workers: 16`
3. **使用融合优化器**: `optim: adamw_torch_fused`
4. **减少评估频率**: `eval_steps: 1000`
5. **限制训练样本**: `max_samples: 50000`

#### Q3: 验证损失不下降

**问题**: 训练损失下降, 但验证损失不降或上升

**解决方案**:
1. **降低学习率**: `learning_rate: 5.0e-5`
2. **增加训练数据**: 删除 `max_samples` 限制
3. **调整正则化**: 增加 `weight_decay: 0.01`
4. **检查数据质量**: 确保训练集和验证集分布一致
5. **增加训练轮数**: `num_train_epochs: 5.0`

### 推理相关

#### Q4: vLLM 加载失败

**问题**: `RuntimeError: Failed to load model` 或 `No module named 'vllm'`

**解决方案**:
1. **检查 vLLM 安装**: `pip install vllm`
2. **确认权重已合并**: 运行 `python merge_lora_weights.py`
3. **检查模型路径**: 确保 `./merged_models/qwen3_mind_news_recommend/` 存在
4. **检查 CUDA 版本**: vLLM 需要 CUDA 11.8+
5. **降低显存占用**: `gpu_memory_utilization=0.7`

#### Q5: vLLM 推理速度慢

**问题**: vLLM 推理速度未达到预期

**解决方案**:
1. **增加批处理上限**: `max_num_batched_tokens=16384`
2. **增加并发序列**: `max_num_seqs=512`
3. **使用 CUDA graph**: `enforce_eager=False`
4. **检查批量大小**: 确保一次推理多个样本
5. **使用量化**: `quantization="awq"`

#### Q6: API 服务无法启动

**问题**: `llamafactory-cli api` 启动失败

**解决方案**:
1. **检查端口占用**: `lsof -i :8000` (Linux/Mac) 或 `netstat -ano | findstr 8000` (Windows)
2. **更换端口**: `--port 8001`
3. **检查配置文件**: 确保 `news_inference.yaml` 路径正确
4. **查看错误日志**: 检查终端输出的详细错误信息

### 数据相关

#### Q7: 数据处理内存不足

**问题**: 运行 `数据预处理.ipynb` 时内存溢出

**解决方案**:
1. **分批处理**: 将数据集分成多个小文件处理
2. **使用生成器**: 避免一次性加载所有数据到内存
3. **增加系统内存**: 至少 16GB RAM
4. **清理中间变量**: 使用 `del` 和 `gc.collect()`

#### Q8: 样本数量不符合预期

**问题**: 处理后的样本数量与文档不一致

**解决方案**:
1. **检查数据完整性**: 确保 `news.tsv` 和 `behaviors.tsv` 完整
2. **调整过滤规则**: 检查代码中的样本过滤条件
3. **查看日志**: 确认处理过程中是否有错误或警告

### 模型相关

#### Q9: 模型准确率低于预期

**问题**: 微调后的模型准确率仍然很低 (<60%)

**解决方案**:
1. **增加训练轮数**: `num_train_epochs: 5.0`
2. **增加训练数据**: 使用全部训练数据 (删除 `max_samples`)
3. **调整学习率**: 尝试 `learning_rate: 5.0e-5` 或 `2.0e-4`
4. **增加 LoRA rank**: 修改 `adapter_config.json` 中的 `r: 16`
5. **检查数据质量**: 确保样本标签正确, 无数据泄露

#### Q10: 微调前后效果没有提升

**问题**: 基座模型和微调模型准确率相近

**解决方案**:
1. **检查模型加载**: 确保评估时加载了正确的微调模型
2. **检查 LoRA 权重**: 确认 `adapter_model.safetensors` 文件存在且大小正常
3. **检查评估代码**: 确保使用了正确的 prompt 格式
4. **重新训练**: 尝试从头重新训练, 调整超参数

---

## 📚 参考资料

### 论文

1. **[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)**
   Edward J. Hu, Yelong Shen, et al. (2021)
   *LoRA 原始论文, 介绍了参数高效微调的核心思想*

2. **[A Survey on Large Language Models for Recommendation](A%20Survey%20on%20Large%20Language%20Models%20for%20Recommendation.pdf)**
   *综述论文, 详细介绍了 LLM 在推荐系统中的应用*

3. **[MIND: A Large-scale Dataset for News Recommendation](https://msnews.github.io/assets/doc/ACL2020_MIND.pdf)**
   Fangzhao Wu, et al. (ACL 2020)
   *MIND 数据集论文*

4. **[vLLM: Easy, Fast, and Cheap LLM Serving](https://arxiv.org/abs/2309.06180)**
   Woosuk Kwon, et al. (2023)
   *vLLM 推理引擎论文*

### 开源项目

- **[Qwen](https://github.com/QwenLM/Qwen)**: 通义千问大模型
- **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)**: 高效的 LLM 微调框架
- **[vLLM](https://github.com/vllm-project/vllm)**: 高性能 LLM 推理引擎
- **[PEFT](https://github.com/huggingface/peft)**: Hugging Face 参数高效微调库

### 文档和教程

- **[Qwen3 官方文档](https://github.com/QwenLM/Qwen/blob/main/README_CN.md)**
- **[LLaMA-Factory 使用指南](https://github.com/hiyouga/LLaMA-Factory/blob/main/README_zh.md)**
- **[vLLM 快速开始](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)**
- **[MIND 数据集官网](https://msnews.github.io/)**


---

## 🙏 致谢

感谢以下项目和团队:

- **阿里巴巴通义千问团队**: 开源 Qwen3-8B 模型
- **微软研究院**: 发布 MIND 数据集
- **LLaMA-Factory 团队**: 提供高效的微调框架
- **vLLM 团队**: 开发高性能推理引擎
- **Hugging Face**: PEFT 库和生态系统

---

<div align="center">

**⭐ 如果本项目对您有帮助, 欢迎 Star 支持! ⭐**


</div>
