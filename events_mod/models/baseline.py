"""Baseline"""

class Baseline():
    def __init__(
        self,
        experiment_name,
        model_name,
        device: str = 'cuda',
    ):
        """Init."""
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.device = device
