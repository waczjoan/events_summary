"""Utility functions."""
import importlib
from typing import List
from transformers import AutoTokenizer

MAX_LENGTH = 127


def load_model_from_config(cfg: dict):
    """Create an object based on the specified module path and kwargs."""
    module_name, class_name = cfg["module"].rsplit(".", maxsplit=1)

    cls = getattr(
        importlib.import_module(module_name),
        class_name,
    )

    return cls(**cfg["kwargs"])


def truncate_texts(texts_batch: List[str], add_eos_token: bool = True) -> str:
    """Truncates text to max length of the model.

    Set the add EOS token parameter to True to add separator between texts.
    Receives a batch of 4 texts,
    truncates the length of the input to 512 tokens as encoded by T5.
    Returns a concatenated string with or without separators.
    """
    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    result = []
    if add_eos_token:
        for t in texts_batch:
            t = tokenizer.encode(t)
            if len(t) > MAX_LENGTH:
                t = t[:-(len(t) - MAX_LENGTH)]
            decoded = tokenizer.decode(t)
            result.append(decoded)
            result.append("</s>")
        result = ''.join(result[:-1])
    else:
        for t in texts_batch:
            t = tokenizer.encode(t)
            if len(t) > MAX_LENGTH:
                t = t[:-(len(t) - MAX_LENGTH)]
            decoded = tokenizer.decode(t)
            result.append(decoded)
        result = ' '.join(result)
    return result
