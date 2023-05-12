"""Pretrained Bert2Bert model for text summarization."""

from transformers import EncoderDecoderModel, AutoTokenizer

MODEL_PATH = 'patrickvonplaten/bert2bert_cnn_daily_mail'


class Bert2Bert:
    """EncoderDecoder pre-fine-tuned model."""

    def __init__(
            self,
            experiment_name: str,
            model_name: str,
            device: str = 'cuda', ):
        """Specify the experiment name and device as per config."""
        self.experiment_name = experiment_name
        self.model = EncoderDecoderModel.from_pretrained(MODEL_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.device = device
        self.model_name = model_name

    def tokenize(self, inputs):
        """Tokenize model inputs."""
        return self.tokenizer(
            inputs,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

    def generate(self, inputs_ids, num_return_sequences: int = 3):
        """Method taking tokenized inputs and returning tensor predictions."""
        input_ids = inputs_ids['input_ids'].to(self.device)
        attention_mask = inputs_ids.attention_mask.to(self.device)
        self.model.to(self.device)
        return self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            num_return_sequences=num_return_sequences,
        )

    def decode(self, outputs):
        """Method decoding tensor predictions into string output."""
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
