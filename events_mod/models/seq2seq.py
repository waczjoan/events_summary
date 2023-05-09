"""AutoModelForSeq2SeqLM from Hugging face, easy uses."""
import collections as c
import operator

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Seq2Seq:
    """Model to change sequence to sequence."""

    def __init__(
            self,
            experiment_name,
            model_name: str = "snrspeaks/t5-one-line-summary",
    ):
        """Init."""
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(self, text) -> torch.Tensor:
        """Tokenize text."""
        return self.tokenizer.encode(
            "summarize: " + text,
            return_tensors="pt",
            add_special_tokens=True
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        num_return_sequences: int = 3
    ):
        """Generate text (tokens)."""
        return self.model.generate(
            input_ids=input_ids,
            num_beams=10,
            max_length=100,
            repetition_penalty=2.5,
            length_penalty=1,
            early_stopping=True,
            num_return_sequences=num_return_sequences
        )

    def decode(self, generated_ids):
        """Decode generated tokens to text.

        If model generate keyword make sure that input includes
        one keyword only one.

        """
        preds = [self.tokenizer.decode(
            g,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ) for g in generated_ids]

        if self.experiment_name == 'key_phrase':
            key_words_list = ' '.join(preds).split(" | ")
            d = dict(c.Counter(key_words_list))
            preds = dict(
                sorted(
                    d.items(),
                    key=operator.itemgetter(1),
                    reverse=True
                )
            )
        return preds
