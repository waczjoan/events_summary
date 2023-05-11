"""Bullet point summarization module."""
from events_mod.models.seq2seq import Seq2Seq
import torch
from typing import List
import textwrap


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
        self.split_handler: SplitHandler = SplitHandler(split_strategy)

    def tokenize(self, text: str) -> List[torch.Tensor]:
        """Tokenize text."""
        data = self.split_handler.split_text(text)
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


class SplitHandler:
    """Text split handler for BPSeq2Seq model."""

    def __init__(self, strategy: str):
        """Set the split strategy for handler."""
        self.strategy = strategy

    def split_text(self, text: str, **kwargs) -> List[str]:
        """General function for article splitting, based on the strategy."""
        return {
            "empty_line": self.empty_line_split,
            "equally": self.equal_split,
            "sentence_split": self.sentence_split,
        }[self.strategy](text, **kwargs)

    def empty_line_split(self, text: str, min_length: int = 100) -> List[str]:
        """Split the article by paragraphs marks. Merge if too short."""
        splits: List[str] = text.replace("<p>", "\n\n").split("\n\n")
        processed_splits: List[str] = [splits[0]]
        for split in splits[1:]:
            if len(split) < min_length:
                processed_splits[-1] += split
            else:
                processed_splits.append(split)
        return processed_splits

    def equal_split(self, text: str, splits_count: int = 3) -> List[str]:
        """Split the article in equal chunks."""
        splits: List[str] = textwrap.wrap(text, len(text) // splits_count)
        return splits[:-2] + ["".join(splits[-2:])]

    def sentence_split(
        self,
        text: str,
        splits_count: int = 3,
        min_length: int = 100
    ) -> List[str]:
        """Split the article in chunks by sentencess."""
        splits: List[str] = text.split(".")

        processed_splits: List[str] = [splits[0]]
        for split in splits[1:]:
            if len(split) < min_length:
                processed_splits[-1] += split
            else:
                processed_splits.append(split)

        chunk_size: int = len(processed_splits) // splits_count
        merged_paragraphs: List[str] = [
            "".join(processed_splits[i:i + chunk_size])
            for i in range(0, len(processed_splits), chunk_size)
        ]
        return merged_paragraphs
