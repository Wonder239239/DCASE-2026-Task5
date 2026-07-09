# ATAE: Audio Token Attention Enhancement

**ATAE (Audio Token Attention Enhancement)** injects a tunable additive bias on audio key positions in the decoder self-attention of audio-language models, encouraging the model to attend more to audio tokens. This repository provides a full experimental pipeline built on [MOSS-Audio-8B-Thinking](https://huggingface.co/OpenMOSS/MOSS-Audio-8B-Thinking): inference, answer parsing/rematching, and accuracy evaluation.

This is the submission repository for the [DCASE 2026 Task 5 Audio-Dependent Question Answering](https://dcase.community/challenge2026/task-audio-dependent-question-answering) challenge. Our team ranked fourth.

This repository was organized from internal `bias/workshop` experiment code and does not modify any files in the original workshop directory.

## Method

- Register a forward hook on the self-attention module of a target decoder layer `L`.
- Add a constant bias `b` to attention logits at audio key positions.
- Use eager attention instead of SDPA/Flash attention, so the hook can modify the 4D floating-point `attention_mask`.
- Apply the method to audio multiple-choice QA: question, choices, and audio in; model answer out.

Typical results on the DCASE 2026 Task 5 Dev Set with 1607 samples:

| Setting | Accuracy |
|---------|----------|
| Baseline, no bias, L0 b0.0 | 64.84% |
| ATAE best, L0 b2.0 | 65.84% |

## Project Layout

```text
atae/
├── atae/                     # Python package
│   ├── bias_core.py          # ATAE injection core: ATAEInjector
│   ├── inference.py          # Inference
│   ├── backfill_parsed_answer.py
│   ├── postprocess_predictions.py
│   └── evaluate.py
├── scripts/
│   └── run_pipeline.sh       # Inference, post-processing, and evaluation
├── data/                     # Data directory; see data/README.md
├── results/                  # Prediction outputs
└── logs/                     # Logs
```

## Setup

**1. Clone this repository**

```bash
git clone <your-repo-url> atae
cd atae
```

**2. Install Python dependencies**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**3. Clone the MOSS-Audio model code**

MOSS-Audio is not available on PyPI and needs to be cloned separately:

```bash
git clone https://github.com/OpenMOSS/MOSS-Audio.git
export MOSS_AUDIO_DIR=/path/to/MOSS-Audio
```

**4. Download model weights**

Download [MOSS-Audio-8B-Thinking](https://huggingface.co/OpenMOSS/MOSS-Audio-8B-Thinking) from Hugging Face:

```bash
export MODEL_PATH=/path/to/MOSS-Audio-8B-Thinking
```

**5. Prepare evaluation data**

See [`data/README.md`](data/README.md). The default expected layout is:

- `data/dev.jsonl`
- `data/dev_audios/*.wav`

```bash
export DATA_DIR="$(pwd)/data"
```

## Quick Start

**Full dev set run:**

```bash
TARGET_LAYER=0 BIAS_VALUE=2.0 bash scripts/run_pipeline.sh
```

**Baseline run:**

Set `BIAS_VALUE=0` to disable bias injection.

```bash
TARGET_LAYER=0 BIAS_VALUE=0 bash scripts/run_pipeline.sh
```

**Post-processing and evaluation only:**

Use this when a prediction file already exists.

```bash
SKIP_INFERENCE=1 OUTPUT_JSONL=results/dev_single_setting_L0_b2.0.jsonl bash scripts/run_pipeline.sh
```

## Main Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Path to MOSS-Audio model weights | Required |
| `MOSS_AUDIO_DIR` | Root directory of the MOSS-Audio source code | Required |
| `DATA_DIR` | Data root directory | `./data` |
| `TARGET_LAYER` | Decoder layer index for ATAE injection | `0` |
| `BIAS_VALUE` | Attention logit bias value; `0` means baseline | `2.0` |
| `MAX_SAMPLES` | Limit the number of samples; `0` means all samples | `0` |
| `RESUME` | Resume from an existing JSONL output | `1` |
| `DO_SAMPLE` | `1` for sampling, `0` for greedy decoding | `0` |
| `MAX_NEW_TOKENS` | Maximum generation length | `1024` |

## Python CLI

```bash
export MOSS_AUDIO_DIR=/path/to/MOSS-Audio
python -m atae.inference \
  --model-path "$MODEL_PATH" \
  --input-jsonl data/dev.jsonl \
  --audio-root data \
  --output-jsonl results/test.jsonl \
  --target-layer 0 \
  --bias-value 2.0 \
  --max-samples 10
```

## License

[MIT License](LICENSE)
