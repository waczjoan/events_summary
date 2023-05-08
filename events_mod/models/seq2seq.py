from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

"""AutoModelForSeq2SeqLM"""


class Baseline:
    def __init__(
        self,
        experiment_name,
        model_name: str = "google/flan-ul2",
        device: str = 'cuda',
    ):
        """Init."""
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.device = device
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(self, text):
        return self.tokenizer(text, return_tensors="pt")

    def generate(self, tokens):
        return self.model.generate(**tokens)

    def batch_decode(self, generation):
        return self.tokenizer.batch_decode(
            generation, skip_special_tokens=True
        )


inputs = tokenizer("A step by step recipe to make bolognese pasta:", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
['In a large skillet, brown the ground beef and onion over medium heat. Add the garlic']