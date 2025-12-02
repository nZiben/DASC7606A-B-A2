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
    - аргументы нужны только для корректной работы evaluate(),
    - дополнительно ограничиваем длину генерации, чтобы не ловить OOM.
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

        # Мы САМИ будем вызывать generate() в кастомной evaluate,
        # так что этот флаг Trainer'а нам не важен, но пусть будет.
        predict_with_generate=True,

        # Для 3.3B-модели: fp16, чтобы влезть в память
        fp16=True,
        gradient_accumulation_steps=1,

        # ВАЖНО: gradient_checkpointing выключен, чтобы не было сигнатурных багов.
        gradient_checkpointing=False,

        dataloader_num_workers=4,

        # Чтобы Trainer не пытался логировать в WandB и т.п.
        report_to="none",

        # Хотим видеть прогресс, если надо
        disable_tqdm=False,

        # 🔥 Ограничиваем длину и отключаем beam search по умолчанию.
        generation_max_length=64,
        generation_num_beams=1,
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
      - использует КАСТОМНЫЙ, легкий по памяти evaluation-loop,
        вместо стандартного Trainer.evaluate() + Accelerate.
    """

    def train(self, *args, **kwargs):
        self.model.eval()
        return None

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        """
        Кастомная реализация evaluate, похожая на твой скрипт:
        - простой DataLoader
        - ручной вызов model.generate(...)
        - потом compute_metrics(...)
        """

        # Выбираем датасет: если явно передали, используем его, иначе self.eval_dataset
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        device = self.args.device
        self.model.to(device)
        self.model.eval()

        # 🔧 ВАЖНО: удаляем 'translation' из фичей перед collate,
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
            labels = batch["labels"].clone()
            all_labels.append(labels)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.no_grad():
                generated_tokens = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.args.generation_max_length or 64,
                    num_beams=self.args.generation_num_beams or 1,
                )

            all_preds.append(generated_tokens.cpu())

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

        # Лейблы: паддим ignore_index (-100)
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
