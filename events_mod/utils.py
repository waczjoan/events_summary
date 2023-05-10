"""Utility functions."""
import importlib
import textwrap
import logging
from typing import List


def load_model_from_config(cfg: dict):
    """Create an object based on the specified module path and kwargs."""
    module_name, class_name = cfg["module"].rsplit(".", maxsplit=1)

    cls = getattr(
        importlib.import_module(module_name),
        class_name,
    )

    return cls(**cfg["kwargs"])


def split_text_into_paragraphs(
    text: str,
    strategy: str = "empty_line",
    splits_number: int = 3
) -> List[str]:
    """The function that splits the article into paragrapahs."""
    if strategy == "empty_line":
        return text.split("\n\n")
    elif strategy == "equally":
        return textwrap.wrap(text, len(text) // (splits_number - 1))
    else:
        logging.error(f"Unknown split strategy: {strategy}")
