# ATAE: Audio Token Attention Enhancement

[English](#english) | [中文](#中文)

---

## 中文

**ATAE（Audio Token Attention Enhancement）** 是一种在音频-语言模型的 **decoder self-attention** 中，对 **audio token 对应的 key 位置** 注入可调 logit bias 的方法，用于引导模型更关注音频内容。本仓库提供基于 [MOSS-Audio-8B-Thinking](https://huggingface.co/OpenMOSS/MOSS-Audio-8B-Thinking) 的完整实验流水线：推理 → 答案解析/重匹配 → 准确率评估。

本仓库是 [DCASE 2026 Task 5 Audio-Dependent Question Answering](https://dcase.community/challenge2026/task-audio-dependent-question-answering) 比赛的参赛仓库，我们最终排名第四。

> 本仓库由内部 `bias/workshop` 实验代码整理而来，**未修改**原始 workshop 目录中的任何文件。

### 方法简介

- 在指定 decoder 层 `L` 的 self-attention 上注册 forward hook
- 对 attention logits 中对应 **audio key** 的位置加上常数 bias `b`
- 模型需使用 **eager attention**（非 SDPA/Flash），以便 hook 能修改 4D floating-point `attention_mask`
- 适用于音频多选题（MCQ）：输入 question + choices + audio，输出模型回答

在 DCASE 2026 Task 5 Dev Set（1607 条）上，MOSS-Audio-8B-Thinking 的典型结果：

| 设置 | 准确率 |
|------|--------|
| Baseline（无 bias，L0 b0.0） | 64.84% |
| ATAE 最佳（L0 b2.0） | 65.84% |

### 目录结构

```
atae/
├── atae/                     # Python 包
│   ├── bias_core.py          # ATAE 注入核心（ATAEInjector）
│   ├── inference.py          # 推理
│   ├── backfill_parsed_answer.py
│   ├── postprocess_predictions.py
│   └── evaluate.py
├── scripts/
│   └── run_pipeline.sh       # 推理 → 后处理 → 评估
├── data/                     # 数据目录（见 data/README.md）
├── results/                  # 预测输出
└── logs/                     # 日志
```

### 环境准备

**1. 克隆本仓库**

```bash
git clone <your-repo-url> atae
cd atae
```

**2. Python 依赖**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**3. MOSS-Audio 模型代码**（不在 PyPI 上，需单独克隆）

```bash
git clone https://github.com/OpenMOSS/MOSS-Audio.git
export MOSS_AUDIO_DIR=/path/to/MOSS-Audio
```

**4. 模型权重**

从 Hugging Face 下载 [MOSS-Audio-8B-Thinking](https://huggingface.co/OpenMOSS/MOSS-Audio-8B-Thinking)：

```bash
export MODEL_PATH=/path/to/MOSS-Audio-8B-Thinking
```

**5. 评估数据**

见 [`data/README.md`](data/README.md)。默认期望：

- `data/dev.jsonl`
- `data/dev_audios/*.wav`

```bash
export DATA_DIR="$(pwd)/data"
```

### 快速开始

**Smoke test（2 条样本，需 GPU）：**

```bash
export MODEL_PATH=/path/to/MOSS-Audio-8B-Thinking
export MOSS_AUDIO_DIR=/path/to/MOSS-Audio
export DATA_DIR=/path/to/your/data

MAX_SAMPLES=2 TARGET_LAYER=0 BIAS_VALUE=2.0 bash scripts/run_pipeline.sh
```

**完整 dev set 单次实验：**

```bash
TARGET_LAYER=24 BIAS_VALUE=1.0 bash scripts/run_pipeline.sh
```

**Baseline（不注入 bias，设 `BIAS_VALUE=0`）：**

```bash
TARGET_LAYER=0 BIAS_VALUE=0 bash scripts/run_pipeline.sh
```

**仅后处理 + 评估（已有预测文件）：**

```bash
SKIP_INFERENCE=1 OUTPUT_JSONL=results/dev_single_setting_L24_b1.0.jsonl bash scripts/run_pipeline.sh
```

### 主要环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `MODEL_PATH` | MOSS-Audio 模型权重路径 | （必填） |
| `MOSS_AUDIO_DIR` | MOSS-Audio 源码根目录 | （必填） |
| `DATA_DIR` | 数据根目录 | `./data` |
| `TARGET_LAYER` | ATAE 注入的 decoder 层索引 | `0` |
| `BIAS_VALUE` | attention logit bias 数值；`0` 即为 baseline | `2.0` |
| `MAX_SAMPLES` | 限制样本数，`0` 表示全量 | `0` |
| `RESUME` | 从已有 jsonl 断点续跑 | `1` |
| `DO_SAMPLE` | `1` 采样，`0` greedy | `0` |
| `MAX_NEW_TOKENS` | 最大生成长度 | `1024` |

### 直接使用 Python CLI

```bash
export MOSS_AUDIO_DIR=/path/to/MOSS-Audio
python -m atae.inference \
  --model-path "$MODEL_PATH" \
  --input-jsonl data/dev.jsonl \
  --audio-root data \
  --output-jsonl results/test.jsonl \
  --target-layer 24 \
  --bias-value 1.0 \
  --max-samples 10
```

---

## English

**ATAE (Audio Token Attention Enhancement)** injects a tunable **additive bias on audio key positions** in the **decoder self-attention** of audio-language models, encouraging the model to attend more to audio tokens. This repo provides a full pipeline built on [MOSS-Audio-8B-Thinking](https://huggingface.co/OpenMOSS/MOSS-Audio-8B-Thinking): inference → answer parsing/rematch → accuracy evaluation.

### Quick start

```bash
pip install -r requirements.txt
export MODEL_PATH=/path/to/MOSS-Audio-8B-Thinking
export MOSS_AUDIO_DIR=/path/to/MOSS-Audio
export DATA_DIR=/path/to/data

MAX_SAMPLES=2 TARGET_LAYER=0 BIAS_VALUE=2.0 bash scripts/run_pipeline.sh
```

Baseline: set `BIAS_VALUE=0` (e.g. `TARGET_LAYER=0 BIAS_VALUE=0 bash scripts/run_pipeline.sh`).

See [`data/README.md`](data/README.md) for dataset layout.

### License

[MIT License](LICENSE)
