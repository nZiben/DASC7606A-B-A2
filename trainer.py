# trainer.py

import torch
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from constants import OUTPUT_DIR
from evaluation import compute_metrics


def create_training_arguments() -> Seq2SeqTrainingArguments:
    """
    Create and return the training arguments for the model.

    Режим "inference only":
    - train() ничего не делает (см. InferenceOnlyTrainer),
    - аргументы нужны только для корректной работы evaluate().
    """
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,

        # Тренировки не будет, но Trainer всё равно просит эти параметры.
        num_train_epochs=1,
        max_steps=0,  # фактически "нет шагов обучения"

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,

        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=0,

        logging_steps=1,
        save_steps=50,

        # В процессе "обучения" валидацию не гоняем
        evaluation_strategy="no",

        save_total_limit=1,
        load_best_model_at_end=False,

        metric_for_best_model="bleu",
        greater_is_better=True,

        max_grad_norm=1.0,

        # predict_with_generate Trainer'у по сути не нужен,
        # т.к. мы делаем свою evaluate(), но пусть будет.
        predict_with_generate=True,

        # Для 3.3B-модели: fp16, чтобы влезть в память
        fp16=True,
        gradient_accumulation_steps=1,

        # gradient_checkpointing выключен, чтобы не ловить странные баги.
        gradient_checkpointing=False,

        dataloader_num_workers=4,

        # Чтобы Trainer не пытался логировать в WandB и т.п.
        report_to="none",

        # Хотим видеть прогресс, если надо
        disable_tqdm=False,

        # Эти поля не используются в нашей кастомной evaluate,
        # но оставим их "разумными", на случай если кто-то вызовет
        # стандартный evaluate() без переопределения.
        generation_max_length=256,
        generation_num_beams=4,
    )

    return training_args


def create_data_collator(tokenizer, model):
    """
    Create data collator for sequence-to-sequence tasks.
    """
    return DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


class InferenceOnlyTrainer(Seq2SeqTrainer):
    """
    Trainer, который:
      - полностью пропускает обучение,
      - использует КАСТОМНЫЙ, лёгкий по памяти evaluation-loop,
        вместо стандартного Trainer.evaluate() + Accelerate.
    """

    def train(self, *args, **kwargs):
        # "Фиктивное" обучение: просто ставим модель в eval-режим.
        self.model.eval()
        return None

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        """
        Кастомная реализация evaluate, ближе к прошлому multi_model_eval:

        - простой DataLoader (без Accelerate),
        - model.generate(...) с параметрами, которые мы
          использовали для facebook/nllb-200-3.3B:
            * max_new_tokens=128
            * num_beams=4
            * do_sample=False
            * no_repeat_ngram_size=3
            * early_stopping=True
        - затем compute_metrics(...) из evaluation.py
        """

        # Выбираем датасет: если явно передали, используем его, иначе self.eval_dataset
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        device = self.args.device
        self.model.to(device)
        self.model.eval()

        # 🔧 Удаляем 'translation' перед collate,
        # иначе DataCollatorForSeq2Seq ломается на nested dict.
        def collate_fn(features):
            if "translation" in features[0]:
                features = [
                    {k: v for k, v in f.items() if k != "translation"}
                    for f in features
                ]
            return self.data_collator(features)

        dataloader = DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=collate_fn,
        )

        all_preds = []
        all_labels = []

        # Простой цикл без Accelerate
        for batch in dataloader:
            # labels остаются на CPU
            labels = batch["labels"].clone()
            all_labels.append(labels)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.no_grad():
                # ❗ ПАРАМЕТРЫ ГЕНЕРАЦИИ, МАКСИМАЛЬНО БЛИЗКИЕ
                # К ПРОШЛОМУ multi_model_eval
                generated_tokens = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=128,
                    num_beams=4,
                    do_sample=False,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )

            all_preds.append(generated_tokens.cpu())

        if not all_preds:
            return {}

        # ==== ПАДДИНГ ПЕРЕД КОНКАТЕНАЦИЕЙ ====
        pad_token_id = self.tokenizer.pad_token_id or 0
        ignore_index = -100

        # Предсказания: паддим pad_token_id
        max_pred_len = max(t.size(1) for t in all_preds)
        padded_preds = []
        for t in all_preds:
            pad_len = max_pred_len - t.size(1)
            if pad_len > 0:
                pad = torch.full(
                    (t.size(0), pad_len),
                    pad_token_id,
                    dtype=t.dtype,
                )
                t = torch.cat([t, pad], dim=1)
            padded_preds.append(t)
        preds_tensor = torch.cat(padded_preds, dim=0)

        # Лейблы: паддим ignore_index (-100),
        # как это делает HuggingFace Trainer.
        max_label_len = max(t.size(1) for t in all_labels)
        padded_labels = []
        for t in all_labels:
            pad_len = max_label_len - t.size(1)
            if pad_len > 0:
                pad = torch.full(
                    (t.size(0), pad_len),
                    ignore_index,
                    dtype=t.dtype,
                )
                t = torch.cat([t, pad], dim=1)
            padded_labels.append(t)
        labels_tensor = torch.cat(padded_labels, dim=0)
        # ==== КОНЕЦ ПАДДИНГА ====

        preds_np = preds_tensor.numpy()
        labels_np = labels_tensor.numpy()

        # Используем заданный в задании compute_metrics (НЕ меняем его)
        metrics = compute_metrics((preds_np, labels_np), self.tokenizer)

        # Префиксуем ключи, как делает Trainer (test_bleu, eval_bleu и т.п.)
        metrics = {f"{metric_key_prefix}_{k}": v for k, v in metrics.items()}

        return metrics


def build_trainer(model, tokenizer, tokenized_datasets) -> Seq2SeqTrainer:
    """
    Build and return the trainer object for (pseudo) training and evaluation.
    """
    data_collator = create_data_collator(tokenizer, model)
    training_args: Seq2SeqTrainingArguments = create_training_arguments()

    return InferenceOnlyTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer),
    )
