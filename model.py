from typing import Union

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from constants import MODEL_CHECKPOINT


TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


def initialize_tokenizer() -> TokenizerType:
    """
    Initialize tokenizer for sequence-to-sequence tasks.
    """
    tokenizer: TokenizerType = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    # Для NLLB-200 задаём языковые коды zh→en
    if "nllb-200" in MODEL_CHECKPOINT:
        tokenizer.src_lang = "zho_Hans"
        tokenizer.tgt_lang = "eng_Latn"

    return tokenizer


def initialize_model() -> PreTrainedModel:
    """
    Initialize the base seq2seq model for inference.

    ВАЖНО: грузим модель сразу в float16, чтобы она занимала меньше памяти
    на 16 ГБ GPU.
    """
    model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_CHECKPOINT,
        torch_dtype="auto",   # или torch.float16, если хочешь жёстко
        low_cpu_mem_usage=True,
    )

    # Для NLLB-200 задаём принудительный BOS для английского
    if "nllb-200" in MODEL_CHECKPOINT:
        tok = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
        tok.src_lang = "zho_Hans"
        tok.tgt_lang = "eng_Latn"
        if hasattr(tok, "lang_code_to_id") and "eng_Latn" in tok.lang_code_to_id:
            model.config.forced_bos_token_id = tok.lang_code_to_id["eng_Latn"]

    return model
