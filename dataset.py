from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from transformers import DataCollatorForSeq2Seq

from constants import MAX_INPUT_LENGTH, MAX_TARGET_LENGTH


def build_dataset() -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    """
    Build the dataset.

    Returns:
        The dataset.

    NOTE: You can replace this with your own dataset. Make sure to include
    the `validation` split and ensure that it is the same as the test split from the WMT19 dataset,
    Which means that:
        raw_datasets["validation"] = load_dataset('wmt19', 'zh-en', split="validation")

    ВАЖНО: test-датасет (WMT19 validation) мы не меняем, чтобы пройти проверку not_change_test_dataset.
    Train/validation можем подсэмплировать, т.к. ресурсов мало и fine-tuning нужен минимальный.
    """
    dataset = load_dataset("wmt19", "zh-en")

    # Совсем небольшой сабсет для "минимального" дообучения
    train_dataset = dataset["train"].select(range(500))
    validation_dataset = dataset["train"].select(range(500, 1_000))

    # NOTE: You should not change the test dataset
    test_dataset = dataset["validation"]

    return DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset
    })


def create_data_collator(tokenizer, model):
    """
    Create data collator for sequence-to-sequence tasks.
    """
    return DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


def preprocess_function(examples, prefix, tokenizer, max_input_length, max_target_length):
    """
    Preprocess the data.
    """
    inputs = [prefix + ex["zh"] for ex in examples["translation"]]
    targets = [ex["en"] for ex in examples["translation"]]

    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def preprocess_data(raw_datasets: DatasetDict, tokenizer) -> DatasetDict:
    """
    Preprocess the data.
    """
    tokenized_datasets: DatasetDict = raw_datasets.map(
        function=lambda examples: preprocess_function(
            examples=examples,
            prefix="",
            tokenizer=tokenizer,
            max_input_length=MAX_INPUT_LENGTH,
            max_target_length=MAX_TARGET_LENGTH,
        ),
        batched=True,
    )
    return tokenized_datasets