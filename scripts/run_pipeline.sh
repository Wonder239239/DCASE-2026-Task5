#!/bin/bash
# Inference -> backfill/rematch -> evaluate accuracy.
#
# Examples:
#   export MODEL_PATH=/path/to/MOSS-Audio-8B-Thinking
#   export MOSS_AUDIO_DIR=/path/to/MOSS-Audio
#   export DATA_DIR=/path/to/DCASE2026-Task5-DevSet
#   bash scripts/run_pipeline.sh
#   TARGET_LAYER=24 BIAS_VALUE=1.0 bash scripts/run_pipeline.sh
#   TARGET_LAYER=0 BIAS_VALUE=0 bash scripts/run_pipeline.sh   # baseline
#   MAX_SAMPLES=20 bash scripts/run_pipeline.sh
#   SKIP_INFERENCE=1 OUTPUT_JSONL=results/...jsonl bash scripts/run_pipeline.sh

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
MOSS_AUDIO_DIR="${MOSS_AUDIO_DIR:-}"

BACKFILL_SCRIPT="${REPO_ROOT}/atae/backfill_parsed_answer.py"
POSTPROCESS_SCRIPT="${REPO_ROOT}/atae/postprocess_predictions.py"
INFER_SCRIPT="${REPO_ROOT}/atae/inference.py"
EVAL_SCRIPT="${REPO_ROOT}/atae/evaluate.py"

MODEL_PATH="${MODEL_PATH:-}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
TARGET_LAYER="${TARGET_LAYER:-0}"
BIAS_VALUE="${BIAS_VALUE:-2.0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
DO_SAMPLE="${DO_SAMPLE:-0}"
RESUME="${RESUME:-1}"

SKIP_INFERENCE="${SKIP_INFERENCE:-0}"
SKIP_POSTPROCESS="${SKIP_POSTPROCESS:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

INPUT_JSONL="${INPUT_JSONL:-${DATA_DIR}/dev.jsonl}"
AUDIO_ROOT="${AUDIO_ROOT:-${DATA_DIR}}"
GOLD_JSONL="${GOLD_JSONL:-${DATA_DIR}/dev.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${REPO_ROOT}/results/dev_single_setting_L${TARGET_LAYER}_b${BIAS_VALUE}.jsonl}"

export PYTHONNOUSERSITE=1
export REPO_ROOT MOSS_AUDIO_DIR MODEL_PATH
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/hf}"
export TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/.cache/torch}"
export TMPDIR="${TMPDIR:-${REPO_ROOT}/.cache/tmp}"
mkdir -p "${REPO_ROOT}/results" "${REPO_ROOT}/logs"
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}"
cd "${REPO_ROOT}"

echo "========== ATAE Pipeline =========="
echo "MODEL_PATH=${MODEL_PATH}"
echo "MOSS_AUDIO_DIR=${MOSS_AUDIO_DIR}"
echo "TARGET_LAYER=${TARGET_LAYER}  BIAS_VALUE=${BIAS_VALUE}"
echo "INPUT_JSONL=${INPUT_JSONL}"
echo "OUTPUT_JSONL=${OUTPUT_JSONL}"
echo "SKIP_INFERENCE=${SKIP_INFERENCE}  SKIP_POSTPROCESS=${SKIP_POSTPROCESS}  SKIP_EVAL=${SKIP_EVAL}"
echo "==================================================="

if [[ ! -f "${INPUT_JSONL}" ]]; then
  echo "Error: input not found: ${INPUT_JSONL}" >&2
  exit 1
fi

if [[ "${SKIP_INFERENCE}" != "1" ]]; then
  if [[ -z "${MODEL_PATH}" ]]; then
    echo "Error: MODEL_PATH is not set." >&2
    exit 1
  fi
  if [[ -z "${MOSS_AUDIO_DIR}" ]]; then
    echo "Error: MOSS_AUDIO_DIR is not set (path to MOSS-Audio source for model code)." >&2
    exit 1
  fi

  echo
  echo "[1/3] Running ATAE inference ..."
  INFER_CMD=(
    "${PYTHON}" "${INFER_SCRIPT}"
    --model-path "${MODEL_PATH}"
    --input-jsonl "${INPUT_JSONL}"
    --audio-root "${AUDIO_ROOT}"
    --output-jsonl "${OUTPUT_JSONL}"
    --target-layer "${TARGET_LAYER}"
    --bias-value "${BIAS_VALUE}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
  )
  if [[ "${MAX_SAMPLES}" != "0" ]]; then
    INFER_CMD+=(--max-samples "${MAX_SAMPLES}")
  fi
  if [[ "${RESUME}" == "1" ]]; then
    INFER_CMD+=(--resume)
  fi
  if [[ "${DO_SAMPLE}" == "1" ]]; then
    INFER_CMD+=(--do-sample)
  fi
  echo "Running: ${INFER_CMD[*]}"
  "${INFER_CMD[@]}"
else
  echo "[1/3] Skipping inference."
fi

if [[ ! -f "${OUTPUT_JSONL}" ]]; then
  echo "Error: prediction file not found: ${OUTPUT_JSONL}" >&2
  exit 1
fi

if [[ "${SKIP_POSTPROCESS}" != "1" ]]; then
  echo
  echo "[2/3] Backfilling and rematching parsed_answer ..."
  "${PYTHON}" "${BACKFILL_SCRIPT}" "${OUTPUT_JSONL}"
  "${PYTHON}" "${POSTPROCESS_SCRIPT}" run-all --pred "${OUTPUT_JSONL}"
else
  echo "[2/3] Skipping postprocess."
fi

if [[ "${SKIP_EVAL}" != "1" ]]; then
  echo
  echo "[3/3] Computing accuracy ..."
  "${PYTHON}" "${EVAL_SCRIPT}" --pred "${OUTPUT_JSONL}" --gold "${GOLD_JSONL}"
else
  echo "[3/3] Skipping evaluation."
fi

echo "Pipeline finished."
echo "Predictions: ${OUTPUT_JSONL}"
