# Hugging Face Machine Translation Assignment Repository (DASC7606A-B-A2)

This project guides you through building, fine-tuning, and evaluating a neural machine translation (NMT) system using Hugging Face libraries. You'll source datasets from the Hub, load community pretrained models and tokenizers, construct a training pipeline, and evaluate using BLEU.

## 🎯 Learning Objectives

By completing this project, you will:

- **Use the Hugging Face Hub for datasets**: discover, download, and prepare translation datasets
- **Load pretrained models and tokenizers** from the community (e.g., MarianMT, mT5, M2M100)
- **Build training pipelines** using `transformers`, `datasets`, `accelerate`, and `evaluate`
- **Fine-tune a translation model** end-to-end for a chosen language pair (zh->en)
- **Evaluate translation quality** using BLEU (via `sacrebleu`) and report results

## 📚 Course Structure

The workflow is organized into 5 files:

1. **Dataset Sourcing & Preparation** (`dataset.py`)
   - Explore HF Datasets, load splits, perform filtering/mapping, and train/validation/test preparation
2. **Model & Tokenizer Setup** (`model.py`)
   - Select a suitable base model from the Hub and initialize tokenizer and config
3. **Training Pipeline** (`trainer.py`)
   - Configure `TrainingArguments`, data collators, metrics, logging, and checkpointing
4. **Run & Monitor Training** (`main.py`)
   - Orchestrate end-to-end training and validation, with periodic evaluation
5. **Evaluation & Reporting** (`evaluation.py`)
   - Compute BLEU on the held-out test set; save artifacts and summary

## 🛠️ Setup Instructions

### Prerequisites

- **Python**: 3.13 or higher
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Storage**: 5–15GB free space (datasets + checkpoints)
- **GPU**: CUDA-compatible GPU recommended for reasonable training time

### Installation

1. Install dependencies using uv (recommended):
```bash
uv sync
```

Or using pip:
```bash
pip install -e .
```

2. Verify installation:
```bash
python -c "import transformers, datasets, evaluate; print('HF stack ready')"
jupyter notebook --version
```

## 🚀 Quick Start

1. **Launch training** with defaults:
```bash
python main.py
```

1. **Launch with uv:**
```bash
uv run python main.py
```

1. **Outputs**:
   - Check `runs/` or `results/` for logs and metrics
   - Check `checkpoints/` for saved models

## 📊 Project Structure

```
DASC7606A-B-A2/
├── main.py                 # Entry point (do NOT modify this file)
├── dataset.py              # HF Datasets loading & preprocessing
├── model.py                # Model & tokenizer initialization from Hub
├── trainer.py              # Training arguments & Seq2SeqTrainer setup
├── evaluation.py           # BLEU computation & reporting via sacrebleu
├── utils.py                # Utilities (seeding, I/O helpers, logging)
├── constants.py            # Optional constants (default config)
├── pyproject.toml          # Project dependencies
├── uv.lock                 # Dependency lock file
└── README.md               # This file
```

## 🧪 Assignment Requirements

- **Goal**: Fine-tune a suitable HF model on a translation dataset and achieve competitive BLEU.
- **What you can modify**: You may change any files in the repo **except** `main.py`, part of `evaluation.py` and `utils.py`. The test set must not be modified.
- **What to improve**:
  - **Enriching Datasets**: Use a more varied dataset (no leakage of the test set into training)
  - **Base model choice**: Select an appropriate pretrained model for your language pair
  - **Training pipeline**: Tune hyperparameters (batch size, LR, epochs, schedulers, label smoothing, gradient accumulation)
  - **Data processing**: Tokenization lengths, filtering, cleaning, language codes, special tokens
  - **Advanced HF features**: Mixed precision (`fp16`/`bf16`), gradient checkpointing, LoRA/PEFT, better data collators, scheduler choices, early stopping

### Example Accepted Datasets/Models

- Datasets: `wmt14`, `wmt16`, `wmt19`, `opus100`, `ted_talks_iwslt`, etc. (via HF Datasets)
- Models: MarianMT (Helsinki-NLP/opus-mt-xx-yy), mT5, MBART-50, M2M100, NLLB-200 (ensure your pair is supported), and even **LLMs**, etc.

## 🌐 Submission: Packaging Scripts

Name your archive using your student ID (e.g., `30300xxxxx.zip`) and include:

```text
30300xxxxx.zip
├── main.py
├── dataset.py
├── model.py
├── trainer.py
├── evaluation.py
├── utils.py
├── constants.py
└── pyproject.toml
```

- **Code Files**: All modified code files (exclude large checkpoints and raw datasets)
- **Submission Format**: Zip archive with your student ID as the filename

### Submission Deadline

- Deadline: Nov. 30 (23:59 GMT +8), 2025
- Late Policy: All submissions later than the deadline will **NOT** be accepted

## 📈 Grading Criteria

We will re-run `main.py` and evaluate on the fixed test set. BLEU (SacreBLEU) is the primary metric.

**Important Considerations:**

1. **Error-Free Execution**: Your code must run without errors (and avoid GPU OOM on the provided environment)
2. **Correct Data Usage**: Do not alter or leak the test set into training
3. **No Personal Pre-trained Model**: "Downloads last month" of loaded pre-trained model on HuggingFace should be **GREATER THAN 10**
4. **Reasonable Performance**: Achieve competitive BLEU given the chosen model and setup
5. **Runtime**: Complete in a reasonable time budget (≤ 12 hours with **ONE GPU on HKU GPU Farm**)

**BLEU-based Grading:**

- **BLEU ≥ 25**: 100%
- **BLEU ≥ 24**: 90%
- **BLEU ≥ 23**: 80%
- **BLEU ≥ 22**: 70%
- **BLEU ≥ 21**: 60%
- **BLEU ≥ 20**: 50%
- **BLEU < 20 / Fail to reproduce / Overtime**: 0%

## ⚙️ Configuration

Common hyperparameters can be configured in your modules (for reference):

- **Batch Size**: 8–64 per device (use gradient accumulation to increase effective batch size)
- **Learning Rate**: 1e-5 – 5e-4
- **Epochs**: 1–10 (depending on model size and dataset scale)
- **Max Sequence Lengths**: 128–256 typical for sentence-level MT
- **Advanced**: label smoothing, scheduler (`cosine`, `linear`), warmup ratio/steps, weight decay

## 🔧 Key Technologies

- **[Transformers](https://huggingface.co/docs/transformers/index)**: Pretrained models, tokenizers, Trainer API
- **[Datasets](https://huggingface.co/docs/datasets/index)**: Streaming, dataset mapping, filtering, caching
- **[Evaluate](https://huggingface.co/docs/evaluate/index)**: Metric loading and computation
- **[SacreBLEU](https://github.com/mjpost/sacrebleu)**: Standardized BLEU evaluation
- **[Accelerate](https://huggingface.co/docs/accelerate/index)**: Efficient multi-GPU/mixed precision training

## 🐛 Troubleshooting

### Common Issues

1. **Memory / OOM**:
   - Reduce `per_device_train_batch_size`, enable `gradient_checkpointing`, increase `gradient_accumulation_steps`
   - Use `--fp16 True` (or `--bf16 True` if supported)

2. **Slow Training**:
   - Use a smaller base model or shorter max lengths
   - Ensure GPU is utilized; avoid excessive evaluation frequency

3. **Poor BLEU**:
   - Verify language codes and tokenization
   - Increase training steps/epochs or try a stronger base model
   - Improve data cleaning and max sequence lengths

4. **Hub/Dataset Issues**:
   - Check network access; try dataset streaming
   - Confirm dataset config and split names

## 📄 Notes

- You are encouraged to explore more Hugging Face capabilities that may improve performance (e.g., LLM base models, PEFT/LoRA, better tokenization strategies, curriculum sampling), provided `main.py`, `utils.py`, `compute_metrics()` and the test set remain unchanged.

## 📘 Example Inference Snippet

```python
from transformers import pipeline

translator = pipeline(
    task="translation_zh_to_en", 
    model="Helsinki-NLP/opus-mt-zh-en"
)

print(translator("你好，今天怎么样？")[0]["translation_text"])
```
