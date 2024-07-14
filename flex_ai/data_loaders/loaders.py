from flex_ai.common import enums
from flex_ai.utils.collators import ensure_spaces_between_chat_completions_separators
from flex_ai.utils.datasets import convert_instruction_dataset_to_chat_dataset
from flex_ai.data_loaders.files import read_jsonl
from datasets import Dataset
import pandas as pd
from flex_ai.common.logger import get_logger
from transformers import PreTrainedTokenizerBase

logger = get_logger(__name__)


def validate_dataset(train_path:str, eval_path:str | None, type: enums.DatasetType, tokenizer: PreTrainedTokenizerBase):
    train_data = read_jsonl(train_path)
    eval_data = read_jsonl(eval_path)

    if type == enums.DatasetType.CHAT:
        train_data = ensure_spaces_between_chat_completions_separators(train_data)
        eval_data = ensure_spaces_between_chat_completions_separators(eval_data)

        transformed_train_data = [
            {"text": tokenizer.apply_chat_template(chat, tokenize=False)}
            for chat in train_data
        ]
        transformed_eval_data = [
            {"text": tokenizer.apply_chat_template(chat, tokenize=False)}
            for chat in eval_data
        ]
        train_dataset = Dataset.from_pandas(pd.DataFrame(transformed_train_data)).shuffle(seed=42)
        eval_dataset = Dataset.from_pandas(pd.DataFrame(transformed_eval_data)).shuffle(seed=42)
    elif type == enums.DatasetType.INSTRUCTION:
        train_data, eval_data = convert_instruction_dataset_to_chat_dataset(train_data, eval_data)
        train_data = ensure_spaces_between_chat_completions_separators(train_data)
        eval_data = ensure_spaces_between_chat_completions_separators(eval_data)

        transformed_train_data = [
            {"text": tokenizer.apply_chat_template(chat, tokenize=False)}
            for chat in train_data
        ]
        transformed_eval_data = [
            {"text": tokenizer.apply_chat_template(chat, tokenize=False)}
            for chat in eval_data
        ]
        train_dataset = Dataset.from_pandas(pd.DataFrame(transformed_train_data)).shuffle(seed=42)
        eval_dataset = Dataset.from_pandas(pd.DataFrame(transformed_eval_data)).shuffle(seed=42)
    elif type == enums.DatasetType.TEXT:
        train_dataset = Dataset.from_pandas(pd.DataFrame(train_data)).shuffle(seed=42)
        eval_dataset = Dataset.from_pandas(pd.DataFrame(eval_data)).shuffle(seed=42)

    return train_dataset, eval_dataset

def _log_transformed_dpo_examples(train_data, eval_data):
        logger.info("Train dataset chat example after model template for DPO:")
        logger.info("prompt:")
        logger.info(train_data[0]["prompt"])
        logger.info("")
        logger.info("chosen:")
        logger.info(train_data[0]["chosen"])
        logger.info("")
        logger.info("rejected:")
        logger.info(train_data[0]["rejected"])
        logger.info("")
        logger.info("Eval dataset chat example after model template for DPO:")
        logger.info("prompt:")
        logger.info(eval_data[0]["prompt"])
        logger.info("")
        logger.info("chosen:")
        logger.info(eval_data[0]["chosen"])
        logger.info("")
        logger.info("rejected:")
        logger.info(eval_data[0]["rejected"])
        logger.info("")

def _log_transformed_sft_examples(train_data, eval_data):
        logger.info("Train dataset chat example after model template:")
        logger.info(train_data[0]["text"])
        logger.info("")
        logger.info("Eval dataset chat example after model template:")
        logger.info(eval_data[0]["text"])
        logger.info("")