from dataclasses import dataclass


@dataclass
class PosNgramSimilarityResult:
    """Result of POS n-gram similarity calculation between two articles.

    Attributes
    ----------
    article_id_a : str
        ID of the first article.
    article_id_b : str
        ID of the second article.
    model : str
        Name of the model used for similarity calculation.
    ngram_size : int
        Size of the n-grams used.
    embedding_method : str
        Method used for embedding generation.
    ngram_similarity : float
        Calculated similarity score between the articles.
    """

    article_id_a: str
    article_id_b: str
    model: str
    ngram_size: int
    embedding_method: str
    ngram_similarity: float
