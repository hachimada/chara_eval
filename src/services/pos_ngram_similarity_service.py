from typing import Optional

from src.database import DatabaseManager
from src.entity.pos_ngram_similarity import PosNgramSimilarityResult
from src.repositories.pos_ngram_similarity_repository import PosNgramSimilarityRepository


class PosNgramSimilarityService:
    """Service for managing POS N-gram similarity operations.

    This service provides high-level operations for POS N-gram similarity
    by delegating database operations to the PosNgramSimilarityRepository.
    """

    def __init__(self, db_manager: DatabaseManager):
        self.repository = PosNgramSimilarityRepository(db_manager)

    def save_similarity(self, result: PosNgramSimilarityResult) -> None:
        """Save a single POS N-gram similarity result to the database.

        Parameters
        ----------
        result : PosNgramSimilarityResult
            The similarity result to save
        """
        self.repository.save(result)

    def save_similarities(self, results: list[PosNgramSimilarityResult]) -> None:
        """Save multiple POS N-gram similarity results to the database using bulk insert.

        Parameters
        ----------
        results : list[PosNgramSimilarityResult]
            List of similarity results to save
        """
        self.repository.save_bulk(results)

    def get_similarity(
        self, article_link_a: str, article_link_b: str, model: str, ngram_size: int, embedding_method: str
    ) -> Optional[float]:
        """Get similarity value between two articles with specific parameters.

        Parameters
        ----------
        article_link_a : str
            ID of the first article
        article_link_b : str
            ID of the second article
        model : str
            Model used for similarity calculation
        ngram_size : int
            Size of N-grams used
        embedding_method : str
            Method used for embedding

        Returns
        -------
        Optional[float]
            Similarity value if found, None otherwise
        """
        return self.repository.find_similarity(article_link_a, article_link_b, model, ngram_size, embedding_method)

    def similarity_exists(
        self, article_link_a: str, article_link_b: str, model: str, ngram_size: int, embedding_method: str
    ) -> bool:
        """Check if similarity exists between two articles with specific parameters.

        Parameters
        ----------
        article_link_a : str
            ID of the first article
        article_link_b : str
            ID of the second article
        model : str
            Model used for similarity calculation
        ngram_size : int
            Size of N-grams used
        embedding_method : str
            Method used for embedding

        Returns
        -------
        bool
            True if similarity exists, False otherwise
        """
        return self.repository.exists(article_link_a, article_link_b, model, ngram_size, embedding_method)

    def get_all_similarities(
        self, model: Optional[str] = None, ngram_size: Optional[int] = None, embedding_method: Optional[str] = None
    ) -> list[PosNgramSimilarityResult]:
        """Get all similarity results with optional filtering.

        Parameters
        ----------
        model : Optional[str], default None
            Filter by model if provided
        ngram_size : Optional[int], default None
            Filter by N-gram size if provided
        embedding_method : Optional[str], default None
            Filter by embedding method if provided

        Returns
        -------
        list[PosNgramSimilarityResult]
            List of similarity results matching the filters
        """
        return self.repository.find_all(model, ngram_size, embedding_method)

    def get_similarities_by_pairs(
        self, pairs: list[tuple[str, str]], model: str, ngram_size: int, embedding_method: str
    ) -> list[PosNgramSimilarityResult]:
        """Get similarities for specific article pairs with given parameters.

        Parameters
        ----------
        pairs : list[tuple[str, str]]
            List of article ID pairs to retrieve similarities for
        model : str
            Model used for similarity calculation
        ngram_size : int
            Size of N-grams used
        embedding_method : str
            Method used for embedding

        Returns
        -------
        list[PosNgramSimilarityResult]
            List of similarity results for the specified pairs
        """
        return self.repository.find_by_pairs(pairs, model, ngram_size, embedding_method)
