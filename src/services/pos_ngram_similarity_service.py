from typing import Optional

from src.database import DatabaseManager
from src.entity.pos_ngram_similarity import PosNgramSimilarityResult
from src.repositories.pos_ngram_similarity_repository import PosNgramSimilarityRepository


class PosNgramSimilarityService:
    """Service for managing high-level POS N-gram similarity operations.

    This service provides business logic and high-level operations for POS N-gram
    similarity analysis, abstracting complex operations and ensuring consistent
    bulk processing for optimal performance.
    """

    def __init__(self, db_manager: DatabaseManager) -> None:
        """Initialize service with database manager.

        Parameters
        ----------
        db_manager : DatabaseManager
            Database manager instance for repository initialization.
        """
        self.repository = PosNgramSimilarityRepository(db_manager)

    def save(self, results: PosNgramSimilarityResult | list[PosNgramSimilarityResult]) -> None:
        """Save similarity results to the database using bulk operations.

        This method always uses bulk operations for optimal performance,
        even when saving a single result. It handles deduplication and
        validation automatically.

        Parameters
        ----------
        results : PosNgramSimilarityResult | list[PosNgramSimilarityResult]
            Single result or list of similarity results to save.
        """
        result_list = [results] if isinstance(results, PosNgramSimilarityResult) else results
        if result_list:
            self.repository.save_bulk(result_list)

    def find_similarity(
        self, article_link_a: str, article_link_b: str, model: str, ngram_size: int, embedding_method: str
    ) -> Optional[float]:
        """Find similarity value between two articles with specific parameters.

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

    def has_similarity(
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

    def get_similarities_with_filters(
        self, model: Optional[str] = None, ngram_size: Optional[int] = None, embedding_method: Optional[str] = None
    ) -> list[PosNgramSimilarityResult]:
        """Get similarity results with optional filtering criteria.

        This method provides a high-level interface for retrieving similarity
        results with flexible filtering options.

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

    def get_similarities_for_pairs(
        self, pairs: list[tuple[str, str]], model: str, ngram_size: int, embedding_method: str
    ) -> list[PosNgramSimilarityResult]:
        """Get similarities for specific article pairs with given parameters.

        This method provides efficient batch retrieval of similarity results
        for multiple article pairs using optimized database queries.

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
