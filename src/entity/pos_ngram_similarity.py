from dataclasses import dataclass


@dataclass
class PosNgramSimilarityResult:
    """Result of POS n-gram similarity calculation from one article to another.

    This class represents the similarity relationship from the perspective of
    one article to another article, simplifying the bidirectional relationship
    into a unidirectional one.

    Attributes
    ----------
    other_article_id : str
        ID of the other article being compared to.
    model : str
        Name of the model used for similarity calculation.
    ngram_size : int
        Size of the n-grams used.
    embedding_method : str
        Method used for embedding generation.
    ngram_similarity : float
        Calculated similarity score to the other article.
    """

    other_article_id: str
    model: str
    ngram_size: int
    embedding_method: str
    ngram_similarity: float
