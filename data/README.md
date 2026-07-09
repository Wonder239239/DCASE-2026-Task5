# Data

This repository does **not** ship evaluation audio or labels. Prepare your own MCQ jsonl in the following format (one object per line):

```json
{
  "id": "dev_0001",
  "question_text": "What characterizes the vocal performance in the track?",
  "multi_choice": ["option A", "option B", "option C", "option D"],
  "answer": "option B",
  "audio_path": "dev_audios/dev_0001.wav"
}
```

## DCASE 2026 Task 5 Dev Set (used in our experiments)

1. Download the official dev set from the [DCASE 2026 Task 5](https://dcase.community/challenge2026/) page.
2. Place files here:
   - `data/dev.jsonl` — questions with `multi_choice` and `answer`
   - `data/dev_audios/` — wav files referenced by `audio_path`

Then run pipelines with:

```bash
export DATA_DIR="$(pwd)/data"
```

Or override explicitly:

```bash
INPUT_JSONL=/path/to/dev.jsonl AUDIO_ROOT=/path/to/dev_audios GOLD_JSONL=/path/to/dev.jsonl bash scripts/run_pipeline.sh
```
