"""Implementation calculation rouge scorer."""
from typing import Dict, List
from rouge_score import rouge_scorer


def calc_rouge(
    text1: str,
    text2: str,
    metrics: List[str] = ['rouge1', 'rouge2', 'rougeLsum']
) -> Dict:
    """Calculation rouge scorer to compare two texts."""
    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
    scores = scorer.score(text1, text2)
    return scores
