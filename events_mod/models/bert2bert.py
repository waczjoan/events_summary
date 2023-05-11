"""Pretrained Bert2Bert model for text summarization."""

from transformers import EncoderDecoderModel, AutoTokenizer
from typing import Union, List

MODEL_PATH = 'patrickvonplaten/bert2bert_cnn_daily_mail'


class Bert2Bert:
    """EncoderDecoder pre-fine-tuned model."""

    def __init__(
            self,
            experiment_name: str,
            device: str = 'cuda'):
        """Specify the experiment name and device as per config."""
        self.experiment_name = experiment_name
        self.model = EncoderDecoderModel.from_pretrained(MODEL_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.device = device

    def predict(self, inputs: Union[str, List[str]]):
        """Method taking a text or list of texts and returning summary(-ies)."""
        tokenized_inputs = self.tokenizer(
            inputs,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        input_ids = tokenized_inputs.input_ids.to(self.device)
        attention_mask = tokenized_inputs.attention_mask.to(self.device)
        self.model.to(self.device)
        outputs = self.model.generate(input_ids, attention_mask=attention_mask)

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
