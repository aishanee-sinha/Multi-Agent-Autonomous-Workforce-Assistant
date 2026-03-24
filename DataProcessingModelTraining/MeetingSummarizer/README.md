# Phase 4 — Meeting Summarizer: Qwen2.5-14B Fine-tuning & Evaluation

End-to-end pipeline for fine-tuning `Qwen/Qwen2.5-14B-Instruct` on meeting transcripts and evaluating it against baseline models using heuristic, semantic, and LLM-as-judge metrics.

---

## Overview

```
Raw meeting transcripts (AMI + MeetingBank + QMSum)
        ↓
Data preprocessing & deduplication → train/val/test splits
        ↓
QLoRA fine-tuning on RTX 4090 (Phase4_Meeting_Summarizer_14B.ipynb)
        ↓
Batched inference on A100 Colab — 4 models × 762 meetings
        ↓
3-tier evaluation: Heuristic G-Eval · BERTScore · DeepEval GPT-4o
        ↓
Comparison charts · Markdown summaries · ICS files · Action items CSV
```

---

## Notebooks

| Notebook | Purpose | Hardware |
|---|---|---|
| `Phase4_Meeting_Summarizer_14B.ipynb` | Data prep + training + local eval | RTX 4090 (24 GB) |
| `Phase4_Evaluation_A100_Colab.ipynb` | Multi-model inference + full evaluation | A100 (40 GB, Colab Pro) |

---

## Hardware & Software Requirements

### Training (Phase4_Meeting_Summarizer_14B.ipynb)
- NVIDIA RTX 4090 (24 GB VRAM) or equivalent
- Python 3.10+, CUDA 12.1
- ~50 GB free disk (base model download ~28 GB)

### Evaluation (Phase4_Evaluation_A100_Colab.ipynb)
- Google Colab Pro with A100 GPU (40 GB)
- Background execution enabled: Runtime → Change runtime type → Background execution
- Google Drive with project folder at `/MyDrive/298-Capstone Project/DATA-298B/`
- OpenAI API key with GPT-4o access (optional — DeepEval cells skipped if not provided)

---

## Installation

### Training notebook
```bash
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.41.2 peft==0.11.1 accelerate==0.32.0 bitsandbytes==0.43.1
pip install datasets rouge-score bert-score nltk scipy sentencepiece tqdm
pip install icalendar dateparser protobuf
```

### Evaluation notebook (Cell 1 handles this automatically)
```bash
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.26.4          # pin before flash-attn to avoid binary mismatch
pip install triton==2.3.1
pip install bitsandbytes==0.43.1   # 0.44.1 breaks on triton.ops
pip install transformers==4.41.2 peft==0.11.1 accelerate==0.32.0
pip install flash-attn==2.5.9.post1 --no-build-isolation
pip install rouge-score bert-score deepeval datasets
```

---

## Part 1 — Data Preprocessing

**Notebook:** `Phase4_Meeting_Summarizer_14B.ipynb` | **Cells:** 4A–4D

### Datasets

| Source | Description | Samples |
|---|---|---|
| AMI corpus | In-person meeting transcripts (Cell 4A) | ~137 |
| MeetingBank | Municipal council meetings from HuggingFace `lytang/MeetingBank-transcript` (Cell 4B) | ~1,366 |
| QMSum | Query-based meeting summarization loaded directly from GitHub (removed from HuggingFace) (Cell 4C) | ~232 |

### Processing steps (Cell 4D)
- Merge all three sources into a unified schema with fields: `meeting_id`, `transcript`, `summary_abstract`, `summary_decisions`, `summary_problems`, `summary_actions`, `action_items`
- Deduplicate using transcript fingerprinting (first 300 chars hash)
- Shuffle with fixed seed and split 70/15/15 into train/val/test
- Train set uses transcript chunking (overlapping windows) to maximise samples from long transcripts → ~32,341 training samples
- Val and test use one sample per meeting (no chunking)

### Output format
Each record is a JSONL line with five structured target sections:
```
ABSTRACT:      concise paragraph summary
DECISIONS:     bullet list of decisions made
PROBLEMS:      bullet list of problems or risks
ACTIONS:       [Owner] - Task - Due: Deadline
ACTIONS_JSON:  [{"owner": "...", "task": "...", "deadline": "...", "discussed_at_sec": 0.0}]
```

---

## Part 2 — Model Training

**Notebook:** `Phase4_Meeting_Summarizer_14B.ipynb` | **Cells:** 5–9

### Model configuration

```python
BASE_MODEL     = "Qwen/Qwen2.5-14B-Instruct"
QUANTIZATION   = "4-bit NF4, double quant, bfloat16 compute dtype"
LORA_R         = 64
LORA_ALPHA     = 128
LORA_DROPOUT   = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
```

### Training configuration

```python
EPOCHS         = 3
BATCH_SIZE     = 1
GRAD_ACCUM     = 8        # effective batch size = 8
LEARNING_RATE  = 1e-4
MAX_SEQ_LEN    = 4096
WARMUP_STEPS   = calculated from dataset size
OPTIMIZER      = adamw_torch_fused
BF16           = True
```

### Cell-by-cell

| Cell | Description |
|---|---|
| Cell 5 | Label-masked dataset and collator — prompt tokens are masked from loss computation so the model only learns to predict the structured output |
| Cell 6 | Load Qwen2.5-14B-Instruct with 4-bit NF4 QLoRA, attach LoRA adapter |
| Cell 7 | Build HuggingFace Dataset objects for train/val/test |
| Cell 8 | Run fine-tuning with HuggingFace Trainer |
| Cell 9 | Save LoRA adapter + tokenizer to `phase4_outputs/phase4_model/final_model/` |

### Training results

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 0 | 0.0608 | 0.2390 |
| 1 | 0.0063 | 0.2423 |
| 2 | 0.0040 | 0.2873 |

Validation loss increased after Epoch 1 indicating overfitting. Best generalization at Epoch 0. Training time ~51 hours on RTX 4090.

---

## Part 3 — Inference

**Notebook:** `Phase4_Evaluation_A100_Colab.ipynb` | **Cells:** 6–7c

Four models are evaluated on the same 762-meeting test set. All inference runs in **background threads** with **resumable JSONL caches** on Drive — disconnects lose no work.

### Drive folder structure required
```
DATA-298B/
├── phase4_model/final_model/     Qwen LoRA adapter
├── proto_final_model/            FLAN-T5 LoRA adapter (Phase 3 prototype)
└── phase4_outputs/test.jsonl     762-meeting test split
```

### Models and inference settings

| Cell | Model | Quantization | Batch | Est. time |
|---|---|---|---|---|
| Cell 6 | Qwen2.5-14B Fine-tuned (v1) | 4-bit NF4 | 6 | ~1.5 hrs |
| Cell 6c | Qwen2.5-14B Fine-tuned (v2) | 4-bit NF4 | 6 | ~17 min (100 only) |
| Cell 7 | Qwen2.5-14B Base | 4-bit NF4 | 6 | ~1.5 hrs |
| Cell 7c | FLAN-T5-large Base + FT | None (bf16) | 16 | ~16 min |

### Prompt variants
- **v1** — rigid structured prompt enforcing all five section headers
- **v2** — natural language prompt with same structure but flexible phrasing, sampling (temperature=0.3, top-p=0.9) instead of greedy decoding

### Monitoring
Run Cell 6b, 6d, or 7b anytime to check progress without interrupting inference.

---

## Part 4 — Evaluation

**Notebook:** `Phase4_Evaluation_A100_Colab.ipynb` | **Cells:** 8–11

Three evaluation tiers are applied:

### Tier 1 — Heuristic G-Eval + ROUGE (Cell 8, all 762 meetings)

| Metric | Method |
|---|---|
| Coherence | Fraction of expected sections present |
| Consistency | ROUGE-L F1 vs reference |
| Fluency | Fraction of sentences with 8–45 words |
| Relevance | ROUGE-1 F1 vs reference |
| Action Quality | Fraction of action items with non-TBD owners |
| ROUGE-1/2/L | Stemmed n-gram overlap |

### Tier 2 — BERTScore (Cell 9, all 762 meetings)
Semantic similarity using `roberta-large` F1 between generated and reference summaries.

### Tier 3 — DeepEval LLM-as-Judge (Cells 10b/10c, 100 meetings per model)

GPT-4o evaluates the same 100 meetings for all models ensuring fair comparison:

| Dimension | Weight | What it measures |
|---|---|---|
| Coherence | 25% | Logical structure, readable sections |
| Consistency | 25% | Factual grounding, no hallucination |
| Fluency | 20% | Clear grammatical English |
| Relevance | 20% | Coverage of key topics and decisions |
| Action Quality | 10% | Owner and deadline present per action |

All DeepEval cells have per-meeting resume support — re-running after a quota interruption skips already scored meetings.

### DeepEval cost estimate

| Job | Cell | Meetings | Est. cost |
|---|---|---|---|
| Qwen Base + FT v1 | Cell 10 (commented) | 100 each | ~$6 |
| FLAN-T5 Base + FT | Cell 10b | 100 each | ~$3 |
| Qwen FT v2 | Cell 10c | 100 | ~$3 |
| **Total** | | | **~$12** |

---

## Part 5 — Results & Outputs

### Key findings

**Heuristic G-Eval** — Fine-tuned Qwen wins on structured output metrics:
- Qwen FT: Action Quality 0.85, ROUGE-1 0.60, ROUGE-L 0.54
- Fine-tuning successfully taught structured section adherence and action item extraction

**DeepEval GPT-4o** — Base Qwen wins on natural language quality:
- Qwen Base: Overall 0.77, Coherence 0.88, Fluency 0.88
- Qwen FT v1: Overall 0.38, Consistency dropped to 0.13
- Fine-tuning hurt naturalness and factual grounding

**Conclusion:** The two evaluation methods reveal a trade-off — fine-tuning achieves its design goal (structured output) but at the cost of natural language quality. This divergence demonstrates that heuristic evaluation alone is insufficient to capture full output quality.

### Saved artifacts

```
phase4_eval_outputs/
├── ft_inference_cache.jsonl          Qwen FT outputs (762 meetings)
├── base_inference_cache.jsonl        Qwen Base outputs (762 meetings)
├── flant5_ft_cache.jsonl             FLAN-T5 FT outputs (762 meetings)
├── flant5_base_cache.jsonl           FLAN-T5 Base outputs (762 meetings)
├── ft_inference_cache_v2.jsonl       Qwen FT v2 outputs (100 meetings)
├── full_evaluation_results.json      aggregated metrics all models
├── csv/
│   ├── deepeval_results.csv          Qwen DeepEval scores
│   ├── deepeval_flant5_cache.jsonl   FLAN-T5 DeepEval scores
│   ├── deepeval_ft_v2_cache.jsonl    Qwen FT v2 DeepEval scores
│   └── action_items_todo_list.csv    all extracted action items
├── plots/
│   ├── all_models_heuristic_geval.png
│   ├── all_models_final_comparison.png
│   └── geval_overall_distribution.png
└── summaries/                        per-meeting markdown summaries
```

---

## Quick Start — Evaluation Only (inference already done)

If inference caches are already on Drive, run these cells in order:

```
Cell 2         → remount Drive
Cell 3         → imports
Cell 4         → utilities + V2 prompt
Cell 5         → load test records
Cell 5b        → reload all 4 inference caches from Drive
Cell 8         → heuristic G-Eval scores
Recovery Cell  → reload DeepEval scores from Drive CSV/JSONL
Cell 11        → print comparison tables
Cell 7d        → generate final 5-model comparison chart
Cell 13        → save summaries, ICS, CSV, final JSON
```

Total time to regenerate all charts after reconnect: ~10 minutes, no GPU or API calls needed.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `No module named 'triton.ops'` | Pin `triton==2.3.1` and `bitsandbytes==0.43.1` |
| `numpy.dtype size changed` | Pin `numpy==1.26.4` before installing flash-attn |
| `Cannot copy out of meta tensor` | Use `device_map="cuda:0"` and `low_cpu_mem_usage=True` instead of `device_map="auto"` |
| Inference stuck, no progress | Model is still loading (~10 min for 14B). Check log file in `phase4_eval_outputs/` |
| DeepEval `insufficient_quota` | Top up OpenAI balance and re-run — resume support skips already scored meetings |
| `New files found: 0` in Apps Script | Run `resetProcessedFiles()` and increase age limit from 24 to 72 hrs |
