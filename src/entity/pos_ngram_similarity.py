from dataclasses import dataclass


@dataclass
class PosNgramSimilarityResult:
    article_id_a: str
    article_id_b: str
    model: str
    ngram_size: int
    embedding_method: str
    ngram_similarity: float