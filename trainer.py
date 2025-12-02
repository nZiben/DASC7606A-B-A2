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
    - дополнительно задаём параметры генерации.
    """
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,

        # Обучения по сути нет, но Trainer требует эти поля:
        num_train_epochs=1,
        max_steps=0,  # "нет шагов обучения"

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,

        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=0,

        logging_steps=1,
        save_steps=50,

        # В процессе "обучения" валидацию не запускаем
        evaluation_strategy="no",

        save_total_limit=1,
        load_best_model_at_end=False,

        metric_for_best_model="bleu",
        greater_is_better=True,

        max_grad_norm=1.0,
        predict_with_generate=True,

        # Для большой модели — fp16
        fp16=True,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,

        dataloader_num_workers=4,
        report_to="none",      # без wandb и т.п.
        disable_tqdm=False,    # пусть прогрессбар Trainer'а живёт, если вдруг пригодится

        # Параметры генерации по умолчанию
        generation_max_length=128,
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
      - использует кастомный evaluation-loop с явным generate()
        и padding, чтобы без ошибок считать BLEU на test.
    """

    def train(self, *args, **kwargs):
        # "Псевдо-обучение" — просто ставим модель в eval-режим.
        self.model.eval()
        return None

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        """
        Кастомная реализация evaluate:
        - простой DataLoader,
        - ручной вызов model.generate(...)
        - padding предсказаний и лейблов до одинаковой длины,
        - вызов compute_metrics из evaluation.py.
        """

        # Если датасет явно не передали, берём сохранённый (validation).
        # main.py передаёт сюда tokenized_datasets["test"], так что
        # для финального BLEU реально используется test.
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        device = self.args.device
        self.model.to(device)
        self.model.eval()

        # Удаляем "translation" перед collate, иначе DataCollatorForSeq2Seq ломается.
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

        # Параметры генерации, близкие к тем, что мы использовали в multi_model_eval
        gen_max_length = self.args.generation_max_length or 128
        gen_num_beams = self.args.generation_num_beams or 8

        for batch in dataloader:
            # Сохраняем labels до того, как уйдут на устройство
            labels = batch["labels"].clone()
            all_labels.append(labels)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.no_grad():
                generated_tokens = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=gen_max_length,
                    num_beams=gen_num_beams,
                    do_sample=False,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
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

        # Используем compute_metrics из evaluation.py (его менять нельзя)
        metrics = compute_metrics((preds_np, labels_np), self.tokenizer)

        # Префиксуем ключи (получишь test_bleu, если metric_key_prefix="test")
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
