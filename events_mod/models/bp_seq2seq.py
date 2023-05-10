"""Bullet point summarization module."""
from events_mod.models.seq2seq import Seq2Seq
import torch
from typing import List
import textwrap
import logging


class BulletPointSeq2Seq(Seq2Seq):
    """Bullet point seq2seq model."""

    splits_number: int = 3

    def __init__(
        self,
        experiment_name,
        model_name: str = "snrspeaks/t5-one-line-summary",
        split_strategy: str = "empty_line"
    ):
        """Initialize the seq2seq module and set split strategy."""
        super().__init__(experiment_name, model_name)
        self.split_strategy = split_strategy

    def split_text(self, text: str) -> List[str]:
        """The function that splits the article into paragrapahs."""
        if self.split_strategy == "empty_line":
            return text.split("\n\n")
        elif self.split_strategy == "equally":
            return textwrap.wrap(text, len(text) // (self.splits_number - 1))
        else:
            logging.error(f"Unknown split strategy: {self.split_strategy}")

    def tokenize(self, text: str) -> List[torch.Tensor]:
        """Tokenize text."""
        data = self.split_text(text)
        return [
            self.tokenizer.encode(
                "summarize: " + subtext,
                return_tensors="pt",
                add_special_tokens=True
            ) for subtext in data
        ]

    def generate(self, input_ids: List[torch.Tensor]):
        """Generate text (tokens)."""
        return [
            self.model.generate(
                input_ids=ids,
                num_beams=10,
                max_length=100,
                repetition_penalty=2.5,
                length_penalty=1,
                early_stopping=True,
            ) for ids in input_ids
        ]

    def decode(self, generated_ids):
        """Decode generated tokens to text.

        If model generate keyword make sure that input includes
        one keyword only one.

        """
        preds = [
            self.tokenizer.batch_decode(
                ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            for ids in generated_ids
        ]
        return "\n".join([f" - {text[0]}" for text in preds])
