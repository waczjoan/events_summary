"""A model for semantic similarity evaluation."""
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_PATH = 'sentence-transformers/all-distilroberta-v1'


class SimSemRoberta:
    """RoBERTa-based sentence transformer for NLI and SimSem."""

    def __init__(
            self,
            experiment_name: str,
            model_name: str,
            device: str = 'cuda'):
        """Initialize model and experiment names as per config."""
        self.model = SentenceTransformer(MODEL_PATH)
        self.model_name = model_name
        self.experiment_name = experiment_name,
        self.device = device

    def compare_embeddings(
            self,
            summary: str,
            texts: List[str]
    ) -> np.ndarray:
        """Method comparing contextual embeddings."""
        summary_emb = self.model.encode(summary)
        summary_emb = np.vstack([summary_emb, np.zeros((768,))])
        assert len(texts) > 1, 'Provide at least two texts for comparison'
        texts_embeddings = self.model.encode(texts)
        result = cosine_similarity(X=summary_emb, Y=texts_embeddings)
        return result[0]
